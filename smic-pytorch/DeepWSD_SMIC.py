import preprocess
import os
import torch.nn as nn
import numpy as np
import torch
import preprocess
from torchvision import models
import torch.nn.functional as F
from PIL import Image
from preprocess import wsd_downsample
from ot.lp import wasserstein_1d
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


def ws_distance(X,Y,P=2,win=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chn_num = X.shape[1]
    X_sum = X.sum().sum()
    Y_sum = Y.sum().sum()

    X_patch   = torch.reshape(X,[win,win,chn_num,-1])
    Y_patch   = torch.reshape(Y,[win,win,chn_num,-1])
    patch_num = (X.shape[2]//win) * (X.shape[3]//win)

    X_1D = torch.reshape(X_patch,[-1,chn_num*patch_num])
    Y_1D = torch.reshape(Y_patch,[-1,chn_num*patch_num])

    X_1D_pdf = X_1D / (X_sum + 1e-6)
    Y_1D_pdf = Y_1D / (Y_sum + 1e-6)

    interval = np.arange(0, X_1D.shape[0], 1)
    all_samples = torch.from_numpy(interval).to(device).repeat([patch_num*chn_num,1]).t()

    X_pdf = X_1D * X_1D_pdf
    Y_pdf = Y_1D * Y_1D_pdf

    wsd   = wasserstein_1d(all_samples, all_samples, X_pdf, Y_pdf, P)

    L2 = ((X_1D - Y_1D) ** 2).sum(dim=0)
    w  =  (1 / ( torch.sqrt(torch.exp( (- 1/(wsd+10) ))) * (wsd+10)**2))

    final = wsd + L2 * w

    return final.sum()


def ws_distance_faster(X_sum, Y_sum, X,Y,P=2,win=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chn_num = X.shape[1]

    X_patch   = torch.reshape(X,[win,win,chn_num,-1])
    Y_patch   = torch.reshape(Y,[win,win,chn_num,-1])
    patch_num = (X.shape[2]//win) * (X.shape[3]//win)

    X_1D = torch.reshape(X_patch,[-1,chn_num*patch_num])
    Y_1D = torch.reshape(Y_patch,[-1,chn_num*patch_num])

    X_1D_pdf = X_1D / (X_sum + 1e-6)
    Y_1D_pdf = Y_1D / (Y_sum + 1e-6)

    interval = np.arange(0, X_1D.shape[0], 1)
    all_samples = torch.from_numpy(interval).to(device).repeat([patch_num*chn_num,1]).t()

    X_pdf = X_1D * X_1D_pdf
    Y_pdf = Y_1D * Y_1D_pdf

    wsd   = wasserstein_1d(all_samples, all_samples, X_pdf, Y_pdf, P)

    L2 = ((X_1D - Y_1D) ** 2).sum(dim=0)
    w  =  (1 / ( torch.sqrt(torch.exp( (- 1/(wsd+10) ))) * (wsd+10)**2))

    final = wsd + L2 * w

    return final.sum()


class DeepWSD_SMIC(torch.nn.Module):
    """
    Enhance DeepWSD by SMIC.
    """
    def __init__(self, channels=3):
        assert channels == 3
        super(DeepWSD_SMIC, self).__init__()
        self.window = 4

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 4):
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

        self.chns = [3, 64, 128, 256, 512, 512]

        self.patchsize = 4
        self.stride = 4
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
        h = x
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]


    def forward(self, x, y, as_loss=True, resize=True):
        assert x.shape == y.shape
        if resize:
            x, y, window, f = wsd_downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        score = 0

        for k in [0, 1, 2, 5]:
            row_padding = round(feats0[k].size(2) / window) * window - feats0[k].size(2)
            column_padding = round(feats0[k].size(3) / window) * window - feats0[k].size(3)

            pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
            feats0_k = pad(feats0[k])
            feats1_k = pad(feats1[k])

            tmp = ws_distance(feats0_k, feats1_k, win=window)
            score = score + torch.log(tmp+1)

        # SMIC-based weighting scheme
        for k in [3,4]:
            row_padding = round(feats0[k].size(2) / window) * window - feats0[k].size(2)
            column_padding = round(feats0[k].size(3) / window) * window - feats0[k].size(3)

            pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
            feats0_k = pad(feats0[k])
            feats1_k = pad(feats1[k])

            N, C, H, W = feats0_k.size()

            X_sum = feats0_k.sum().sum()
            Y_sum = feats1_k.sum().sum()

            # implement the random projections by a depth-wise convolutional layer
            mic_feats0_k, mic_feats1_k = self.randpj_layers[k-3](feats0_k), self.randpj_layers[k-3](feats1_k)

            # divide projected features into several patches
            distparam = self.unfold(mic_feats0_k).view(N, self.outchns, self.patchsize * self.patchsize, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()
            prisparam = self.unfold(mic_feats1_k).view(N, self.outchns, self.patchsize * self.patchsize, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()

            # divide original features into several patches
            feats0_k_patches = self.unfold(feats0_k).view(N, C, self.patchsize, self.patchsize, -1).transpose(2, 4).transpose(3, 4).transpose(1, 2).contiguous()
            feats1_k_patches = self.unfold(feats1_k).view(N, C, self.patchsize, self.patchsize, -1).transpose(2, 4).transpose(3, 4).transpose(1, 2).contiguous()
            score_layer = 0
            for pt_index in range(feats0_k_patches.shape[1]):
                # calculate wsd
                wsd_score_pt = ws_distance_faster(X_sum, Y_sum, feats0_k_patches[:, pt_index, :, :, :], feats1_k_patches[:, pt_index, :, :, :])
                MIC_chn = 0
                for cc in range(self.outchns):
                    mine.compute_score(distparam[0, pt_index, cc], prisparam[0, pt_index, cc])
                    MIC_chn += mine.mic()
                MIC_chn = MIC_chn / self.outchns
                # SMIC-based weighting
                score_layer += (1.0 - MIC_chn) * wsd_score_pt

            score += score_layer


        score = score / 6

        if as_loss:
            return score
        elif f==1:
            return torch.log(score + 1)
        else:
            return torch.log(score + 1)**2

if __name__ == '__main__':

    model = DeepWSD_SMIC()
    model.cuda()
    
    wsd_img0 = preprocess.wsd_prepare_image(Image.open('./imgs/Img1_ref.png').convert("RGB"))
    wsd_img1 = preprocess.wsd_prepare_image(Image.open('./imgs//Img1_dist.png').convert("RGB"))

    wsd_img0 = wsd_img0.cuda()
    wsd_img1 = wsd_img1.cuda()


    # Calculate the distance after smic weighting
    score = model(wsd_img0, wsd_img1, as_loss = False).item()

    print(score)