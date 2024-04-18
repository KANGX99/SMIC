import preprocess
import os
import torch.nn as nn
import torch
import preprocess
from torchvision import models
from PIL import Image
import inspect
from collections import namedtuple
from minepy import MINE


mine = MINE(alpha=0.5, c=10)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)


def upsample(in_tens, out_HW=(64,64)):
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out
    

class LPIPS_SMIC(nn.Module):
    def __init__(self, pnet_rand=False, pnet_tune=False, eval_mode=True, net='vgg', version='0.1', use_dropout=True):
        """
        Enhance LPIPS by SMIC.
        """
        super(LPIPS_SMIC, self).__init__()

        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand

        self.chns = [64,128,256,512,512]
        self.L = len(self.chns)

        self.net = vgg16(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)    

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
        self.lins = nn.ModuleList(self.lins)

        model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights_lpips/v%s/%s.pth'%(version,net)))

        print('Loading model from: %s'%model_path)
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)     

        if(eval_mode):
            self.eval()
        
        self.patchsize = 7
        self.stride = 7
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
    
    def forward(self, in0, in1):
        outs0, outs1 = self.net.forward(in0), self.net.forward(in1)
        feats0, feats1, mic_feats0, mic_feats1, diffs = {}, {}, {}, {}, {}
        val = 0
        for kk in [0,1,4]:
            feats0[kk], feats1[kk] = outs0[kk], outs1[kk]
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in [0,1,4]]
        for l in range(3):
            val += res[l]

        # SMIC-based weighting scheme
        for kk in [2,3]:
            feats0[kk], feats1[kk] = outs0[kk], outs1[kk]
            b, c, h, w = feats0[kk].size()
            k = self.patchsize

            # implement the random projections by a depth-wise convolutional layer
            mic_feats0[kk], mic_feats1[kk] = self.randpj_layers[kk-2](feats0[kk]), self.randpj_layers[kk-2](feats1[kk])

            # divide original features into several patches
            tdistparam = self.unfold(feats0[kk]).view(b, c, k * k, -1).permute(0, 3, 1, 2).contiguous()
            tprisparam = self.unfold(feats1[kk]).view(b, c, k * k, -1).permute(0, 3, 1, 2).contiguous()  # b, pt_num, c, k*k

            # divide projected features into several patches
            distparam = self.unfold(mic_feats0[kk]).view(b, self.outchns, k * k, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()
            prisparam = self.unfold(mic_feats1[kk]).view(b, self.outchns, k * k, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()
            pt_num = tdistparam.shape[1]
            mics = 0
            for ii in range(pt_num):
                MIC_chn = 0
                for cc in range(self.outchns):
                    mine.compute_score(distparam[0, ii, cc], prisparam[0, ii, cc])
                    MIC_chn += mine.mic()
                MIC_chn = MIC_chn / self.outchns

                # calculate lpips metric
                diffs = torch.sum((tdistparam[0,ii]-tprisparam[0,ii])**2).item()
                # SMIC-based weighting
                mics += (1.0 - MIC_chn) * diffs
            mics = mics / (pt_num)
            val = val + mics
        return val


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    model = LPIPS_SMIC()
    model.cuda()

    img0 = preprocess.prepare_image(Image.open('./imgs/Img1_ref.png').convert("RGB"))
    img1 = preprocess.prepare_image(Image.open('./imgs//Img1_dist.png').convert("RGB"))

    img0 = img0.cuda()
    img1 = img1.cuda()

    # Calculate the distance after smic weighting
    score = model(img0, img1).item()

    print(score)