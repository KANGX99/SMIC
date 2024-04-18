import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

def prepare_image(image):
    msize = min(image.size)
    if msize > 128:
        tar_size = max(int(msize / (1.0 * 48)) * 32, 128)
        image = transforms.functional.resize(image, tar_size)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0) 

def wsd_prepare_image(image, repeatNum = 1):
    H, W = image.size
    if max(H,W)>512 and max(H,W)<1000:
        image = transforms.functional.resize(image,[256,256])
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)

# Process input of VGG16 to make it close to 256
def wsd_downsample(img1, img2, maxSize = 256):
    _,channels,H,W = img1.shape
    f = int(max(1,np.round(max(H,W)/maxSize)))

    aveKernel = (torch.ones(channels,1,f,f)/f**2).to(img1.device)
    img1 = F.conv2d(img1, aveKernel, stride=f, padding = 0, groups = channels)
    img2 = F.conv2d(img2, aveKernel, stride=f, padding = 0, groups = channels)
    # For an extremely Large image, the larger window will use to increase the receptive field.
    if f >= 5:
        win = 16
    else:
        win = 4
    return img1, img2, win, f