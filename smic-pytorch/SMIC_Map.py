import os
import torch.nn as nn
import numpy as np
import torch
import preprocess
from torchvision import models
from PIL import Image
import torch.nn.functional as F
from minepy import MINE
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

mine = MINE(alpha=0.5, c=10)


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()


class GEN_SMIC_MAP(torch.nn.Module):
    def __init__(self):
        super(GEN_SMIC_MAP, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.patchsize = 7
        self.stride = 1
        self.unfold = nn.Unfold(kernel_size=self.patchsize, stride=self.stride)
        self.outchns = 32
        
        self.randpj = nn.Conv2d(256, self.outchns, 1, 1,0, bias = False)
        torch.manual_seed(-3566)
        nn.init.xavier_uniform_(self.randpj.weight, gain=1.0)
        for param in self.randpj.parameters():
                param.requires_grad = False

        self.randpj2 = nn.Conv2d(512, self.outchns, 1, 1,0, bias = False)
        nn.init.xavier_uniform_(self.randpj2.weight, gain=1.0)
        for param in self.randpj2.parameters():
                param.requires_grad = False
        
        self.randpj_layers = [self.randpj, self.randpj2]
        
    def forward_once(self, x):
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        return [h_relu3_3, h_relu4_3]
        
    def forward(self, x, y,name):
        b,c,h,w = x.size()
        with torch.no_grad():
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y) 

        for k in range(2):
            N, C, H, W = feats0[k].size()

            mic_feats0_k, mic_feats1_k = self.randpj_layers[k](feats0[k]), self.randpj_layers[k](feats1[k])
            distparam = self.unfold(mic_feats0_k).view(N, self.outchns, self.patchsize * self.patchsize, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()
            prisparam = self.unfold(mic_feats1_k).view(N, self.outchns, self.patchsize * self.patchsize, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()

            MIC_list = []
            for pt_index in range(distparam.shape[1]):
                MIC_chn = 0
                for cc in range(self.outchns):
                    mine.compute_score(distparam[0, pt_index, cc], prisparam[0, pt_index, cc])
                    MIC_chn += mine.mic()
                MIC_chn = MIC_chn / self.outchns
                MIC_list.append(1 - MIC_chn)

            micmap = np.array(MIC_list).reshape((H - self.patchsize)//self.stride + 1, (W - self.patchsize)//self.stride + 1)
            min_mic = np.min(micmap)
            max_mic = np.max(micmap)
            micmap = (micmap - min_mic) / (max_mic - min_mic)
            

            micmap_resized = zoom(micmap, ((h)/((H - self.patchsize)//self.stride + 1), (w)/((W - self.patchsize)//self.stride + 1)),order = 1)

            cmap = 'magma'

            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            im1 = axes.imshow(micmap_resized, cmap=cmap)

            axes.set_title('SMIC map')
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_xlabel('')
            axes.set_ylabel('')

            plt.tight_layout()

            os.makedirs('./attention_map', exist_ok=True)
            plt.savefig('./attention_map/{}_stage_{}.png'.format(name, k+3))

# this file is used for generating the mic map based on the original image and its corresponding distorted image.
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='generation of attention map')
    parser.add_argument('--ref_path', type=str, default='./imgs/Img1_ref.png')
    parser.add_argument('--dist_path', type=str, default='./imgs//Img1_dist.png')
    args = parser.parse_args()

    model = GEN_SMIC_MAP()
    model.cuda()

    img0 = preprocess.prepare_image(Image.open(args.ref_path).convert("RGB"))
    img1 = preprocess.prepare_image(Image.open(args.dist_path).convert("RGB"))

    img0 = img0.cuda()
    img1 = img1.cuda()

    model(img0, img1, name = args.dist_path.split("/")[-1:])

