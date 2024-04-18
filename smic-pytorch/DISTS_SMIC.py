import preprocess
import os
import numpy as np
import torch.nn as nn
import torch
import preprocess
from torchvision import models
import sys
import torch.nn.functional as F
from PIL import Image
from minepy import MINE

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


class DISTS_SMIC(torch.nn.Module):
    """
    Enhance DISTS by SMIC.
    """
    def __init__(self, load_weights=True):
        super(DISTS_SMIC, self).__init__()
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

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        if load_weights:
            weights = torch.load(os.path.join(sys.prefix,'weights_dists.pt'))
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']

        self.patchsize = 7
        self.stride = 6
        self.unfold = nn.Unfold(kernel_size=self.patchsize, stride=self.stride)
        self.outchns = 32
        # random projections
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
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        
    def forward(self, x, y):
        with torch.no_grad():
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y) 
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)

        for k in [0, 1, 2, 5]:
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        # SMIC-based weighting scheme
        score_s_34 = 0
        for k in [3,4]:
            N, C, H, W = feats0[k].size()
            # implement the random projections by a depth-wise convolutional layer
            mic_feats0_k, mic_feats1_k = self.randpj_layers[k-3](feats0[k]), self.randpj_layers[k-3](feats1[k])
            # divide projected features into several patches
            distparam = self.unfold(mic_feats0_k).view(N, self.outchns, self.patchsize * self.patchsize, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()
            prisparam = self.unfold(mic_feats1_k).view(N, self.outchns, self.patchsize * self.patchsize, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()
            # divide original features into several patches
            feats0_k_patches = self.unfold(feats0[k]).view(N, C, self.patchsize, self.patchsize, -1).transpose(2, 4).transpose(3, 4).transpose(1, 2).contiguous()
            feats1_k_patches = self.unfold(feats1[k]).view(N, C, self.patchsize, self.patchsize, -1).transpose(2, 4).transpose(3, 4).transpose(1, 2).contiguous()
            score_layer = 0
            for pt_index in range(feats0_k_patches.shape[1]):
                x_mean = feats0_k_patches[:, pt_index, :, :, :].mean([2,3], keepdim=True)
                y_mean = feats1_k_patches[:, pt_index, :, :, :].mean([2,3], keepdim=True)
                S1_pt = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
                S1_pt = S1_pt.mean(1,keepdim=True)

                x_var = ((feats0_k_patches[:, pt_index, :, :, :]-x_mean)**2).mean([2,3], keepdim=True)
                y_var = ((feats1_k_patches[:, pt_index, :, :, :]-y_mean)**2).mean([2,3], keepdim=True)
                xy_cov = (feats0_k_patches[:, pt_index, :, :, :]*feats1_k_patches[:, pt_index, :, :, :]).mean([2,3],keepdim=True) - x_mean*y_mean
                S2_pt = (2*xy_cov+c2)/(x_var+y_var+c2)
                S2_pt = S2_pt.mean(1,keepdim=True)

                MIC_chn = 0
                for cc in range(self.outchns):
                    mine.compute_score(distparam[0, pt_index, cc], prisparam[0, pt_index, cc])
                    MIC_chn += mine.mic()
                MIC_chn = MIC_chn / self.outchns
                # SMIC-based weighting
                score_layer += (1-MIC_chn) * (2-S1_pt - S2_pt)

            score_s_34 += score_layer / feats0_k_patches.shape[1] 

        score = (1 - (dist1+dist2)).item() + score_s_34
        return score


if __name__ == '__main__':
    model = DISTS_SMIC()
    model.cuda()

    img0 = preprocess.prepare_image(Image.open('./imgs/Img1_ref.png').convert("RGB"))
    img1 = preprocess.prepare_image(Image.open('./imgs//Img1_dist.png').convert("RGB"))

    img0 = img0.cuda()
    img1 = img1.cuda()

    # Calculate the distance after smic weighting
    score = model(img0, img1).item()

    print(score)