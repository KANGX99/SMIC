import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

import preprocess
from PIL import Image
import os
import torch
from minepy import MINE
import torch.nn as nn
from torchvision import models
mine = MINE(alpha=0.5, c=10)
import numpy as np
from scipy.ndimage import zoom


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_map = torch.mean(ssim_map,dim = 1).squeeze(0)
    return ssim_map


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_map = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)

    return ssim_map


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
    

class SSIM_SMIC(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ) -> None:

        super(SSIM_SMIC, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

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
    
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        N,C,H,W = X.size()
        with torch.no_grad():
            Ref_fea = self.forward_once(X)
            Dist_fea = self.forward_once(Y)

        stage_index = 0
        key = 0
        # distortion map generation
        ssim_map = ssim(X,Y,data_range=self.data_range,size_average=self.size_average,win=self.win,K=self.K,nonnegative_ssim=self.nonnegative_ssim,)
        ssim_map = 1 - ssim_map

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

            pt_num = distparam.shape[1]
            MIC_chn_list = []
            for ii in range(pt_num):
                MIC_chn = 0
                for cc in range(self.outchns):
                    mine.compute_score(distparam[0, ii, cc], prisparam[0, ii, cc])
                    MIC_chn += mine.mic()
                MIC_chn = MIC_chn / self.outchns
                MIC_chn_list.append(1.0 - MIC_chn)
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
        micmap_resized = zoom(MIC_final, ((H-self.win_size+1)/h_3, (W-self.win_size+1)/w_3), order=1)
        micmap_resized = torch.tensor(micmap_resized).cuda()
        # calculate weighted SSIM
        return torch.mean(ssim_map*micmap_resized)


if __name__ == '__main__':
    model = SSIM_SMIC(data_range=1,win_size = 7)
    model.cuda()

    img0 = preprocess.prepare_image(Image.open('./imgs/Img1_ref.png').convert("RGB"))
    img1 = preprocess.prepare_image(Image.open('./imgs//Img1_dist.png').convert("RGB"))

    img0 = img0.cuda()
    img1 = img1.cuda()

    # Calculate the distance after smic weighting
    score = model(img0, img1).item()

    print(score)