import preprocess
from PIL import Image
import os
import torch
import torch.nn as nn
from minepy import MINE
import numpy as np
mine = MINE(alpha=0.5, c=10)
from torchvision import models
from scipy.ndimage import zoom


class L2Pool2d(torch.nn.Module):
    r"""Applies L2 pooling with Hann window of size 3x3
    Args:
        x: Tensor with shape (N, C, H, W)"""
    EPS = 1e-12
    def __init__(self, kernel_size: int = 3, stride: int = 2, padding=1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel is None:
            C = x.size(1)
            self.kernel = self._hann_filter(self.kernel_size).repeat((C, 1, 1, 1)).to(x)

        out = torch.nn.functional.conv2d(
            x ** 2, self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1]
        )
        return (out + self.EPS).sqrt()

    def _hann_filter(self, kernel_size: int) -> torch.Tensor:
        r"""Creates  Hann kernel
        Returns:
            kernel: Tensor with shape (1, kernel_size, kernel_size)
        """
        # Take bigger window and drop borders
        window = torch.hann_window(kernel_size + 2, periodic=False)[1:-1]
        kernel = window[:, None] * window[None, :]
        # Normalize and reshape kernel
        return kernel.view(1, kernel_size, kernel_size) / kernel.sum()
    

class PSNR_SMIC(nn.Module):
    def __init__(self):
        super(PSNR_SMIC, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2Pool2d(kernel_size=3, stride=2, padding=1))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])        
        self.stage3.add_module(str(9), L2Pool2d(kernel_size=3, stride=2, padding=1))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2Pool2d(kernel_size=3, stride=2, padding=1))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.patchsize = 7
        self.unfold_mse = nn.Unfold(kernel_size=self.patchsize, stride=1)

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

    def forward_once(self, x):

        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h

        return [ h_relu3_3, h_relu4_3]
    
    def forward(self, x, y):
        N,C,H,W = x.size()
        with torch.no_grad():
            Ref_fea = self.forward_once(x)
            Dist_fea = self.forward_once(y)
        # distortion map generation
        x_pt = self.unfold_mse(x).view(N, 3, self.patchsize * self.patchsize, -1).permute(0, 3, 1, 2).contiguous().squeeze(0)
        y_pt = self.unfold_mse(y).view(N, 3, self.patchsize * self.patchsize, -1).permute(0, 3, 1, 2).contiguous().squeeze(0)
        mse_map = torch.mean((x_pt - y_pt)**2, dim=(1,2)).view(H - self.patchsize + 1, -1)
        stage_index = 0
        for key in range(2):
            tdistparam = Dist_fea[key]
            tprisparam = Ref_fea[key]
            # implement the random projections by a depth-wise convolutional layer
            if stage_index == 0:
                distparam = self.randpj(tdistparam)
                prisparam = self.randpj(tprisparam)
                stage_index += 1
            else:
                distparam = self.randpj2(tdistparam)
                prisparam = self.randpj2(tprisparam)
            b, c, h, w = tdistparam.size()
            k = self.patchsize
            # divide projected features into several patches
            distparam = self.unfold(distparam).view(b, self.outchns, k * k, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()
            prisparam = self.unfold(prisparam).view(b, self.outchns, k * k, -1).permute(0, 3, 1, 2).contiguous().cpu().numpy()
            
            MIC_chn_list = []
            for pt_index in range(distparam.shape[1]):
                MIC_chn = 0
                for cc in range(self.outchns):
                    mine.compute_score(distparam[0, pt_index, cc], prisparam[0, pt_index, cc])
                    MIC_chn += mine.mic()
                MIC_chn = MIC_chn / self.outchns
                MIC_chn_list.append(1 - MIC_chn)
            # attention map generation
            micmap = np.array(MIC_chn_list).reshape((h - k)//self.stride + 1, (w - k)//self.stride + 1)
            if(key == 0):
                micmap_3 = micmap
                h_3 = (h - k)//self.stride + 1
                w_3 = (w - k)//self.stride + 1
            else:
                micmap_resized_4 = zoom(micmap, (h_3/((h - k)//self.stride + 1), w_3/((w - k)//self.stride + 1)), order=1)
                MIC_final = (micmap_3 + micmap_resized_4) / 2
        # the bilinear interpolation is adopted for the resize operation
        micmap_resized = zoom(MIC_final, ((H-self.patchsize+1)/h_3, (W-self.patchsize+1)/w_3), order=1)
        micmap_resized = torch.tensor(micmap_resized).cuda()

        result = torch.mean(mse_map * micmap_resized)

        # calculate corresponding PSNR
        return 10 * torch.log10(1 / (result+1e-10)) # 1e-10 is to tackle the inf in kadid-10k 


if __name__ == '__main__':

    model = PSNR_SMIC()
    model.cuda()

    img0 = preprocess.prepare_image(Image.open('./imgs/Img1_ref.png').convert("RGB"))
    img1 = preprocess.prepare_image(Image.open('./imgs//Img1_dist.png').convert("RGB"))

    img0 = img0.cuda()
    img1 = img1.cuda()

    # Calculate the distance after smic weighting
    score = model(img0, img1).item()

    print(score)