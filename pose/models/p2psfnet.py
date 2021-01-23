import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from utils.torch import *
from models.handy_lct_net import HandyLCTNet
# from models.unet3d_new import UNet3D
import time
from scipy.sparse import lil_matrix, csr_matrix
from numpy.fft import fft2, ifft2, ifftshift, fftn, ifftn
import math
from numpy import linalg
# import matplotlib.pyplot as plt


class ResNet(nn.Module):

    def __init__(self, out_dim, fix_params=False, running_stats=False):
        super().__init__()
        self.out_dim = out_dim
        self.resnet = models.resnet18(pretrained=True)
        if fix_params:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_dim)
        self.bn_stats(running_stats)

    def forward(self, x):
        return self.resnet(x)

    def bn_stats(self, track_running_stats):
        for m in self.modules():
            if type(m) == nn.BatchNorm2d:
                m.track_running_stats = track_running_stats


"""
Components used by V2PsfNet
"""


class Conv3DBlock(nn.Module):
    """(conv => BN => ReLU)"""
    def __init__(self, in_ch, out_ch):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.inconv = nn.Sequential(
            Conv3DBlock(in_ch=in_ch, out_ch=out_ch),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.inconv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            Conv3DBlock(in_ch=in_ch, out_ch=out_ch)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            Conv3DBlock(in_ch=out_ch, out_ch=out_ch)
        )

    def forward(self, x):
        return self.up(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2),
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=1)
        )

    def forward(self, x):
        return self.outconv(x)


"""
V2PsfNet
Input: 3D transient image (32*32*64)
Output: 3D psf or psf helper matrix (128*64*64)
"""


class V2PsfNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(V2PsfNet, self).__init__()
        self.inc = InConv(input_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.up3 = Up(16, 1)
        self.outc = OutConv(1, output_channels)

        self._initialize_weights()

    def forward(self, x):               # batch_size ×  1 ×  32 × 32 × 64
        x = x.permute(0, 1, 4, 3, 2)    # batch_size ×  1 ×  64 × 32 × 32
        x1 = self.inc(x)                # batch_size × 16 ×  32 × 16 × 16
        x2 = self.down1(x1)             # batch_size × 32 ×  16 ×  8 ×  8
        x3 = self.down2(x2)             # batch_size × 64 ×   8 ×  4 ×  4
        x = self.up1(x3)                # batch_size × 32 ×  16 ×  8 ×  8
        x = self.up2(x + x2)            # batch_size × 16 ×  32 × 16 × 16
        x = self.up3(x + x1)            # batch_size ×  1 ×  64 × 32 × 32
        x = self.outc(x)                # batch_size ×  1 × 128 × 64 × 64
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


"""
Input: batchsize x 1 x 32 x 32 x 64
Output: batch x 128
optimizes SNR usd for invpsf
"""


class P2PsfNet(nn.Module):

    def __init__(self, out_dim, device=None, fix_params=False, running_stats=False, spec='resnet18'):
        super().__init__()
        print('P2PsfNet')
        self.out_dim = out_dim
        self.device = device

        self.v2psf_layer = V2PsfNet(input_channels=1, output_channels=1)
        weight = torch.tensor([2.4, 1.0], device=self.device)
        self.weight = torch.nn.Parameter(weight)

        print('defining NLOS...')
        self.lct = HandyLCTNet(device=self.device)
        print('---------- done.')

        self.upsample = torch.nn.Upsample(mode='nearest', size=64)

        self.fix_params = fix_params
        self.running_stats = running_stats
        self.spec = spec
        self.resnet = ResNet(self.out_dim, fix_params=False, running_stats=False)

    def forward(self, x):

        if self.device is not None:
            vol = torch.zeros((x.shape[0], 3, 32, 32), device=self.device)
            psf = torch.ones((x.shape[0], 1, 128, 64, 64), device=self.device)
        else:
            vol = torch.zeros((x.shape[0], 3, 32, 32))
            psf = torch.ones((x.shape[0], 1, 128, 64, 64))

        """
        Get optimal invpsf matrix for each sample.
        Residual connection sums up
        - initialized invpsf with conventional value (defined by LCT class)
        - optimized volume matrix by network
        """
        softmax_weight = F.softmax(self.weight)
        for i in range(x.shape[0]):
            psf[i, 0, :, :, :] = self.lct.invpsf
        psf = softmax_weight[0] * psf + softmax_weight[1] * self.v2psf_layer(x)

        """ Compute LCT result (3D depth vol. for each sample) """
        for i in range(x.shape[0]):                         # batch_size ×   1 × 32 × 32 × 64
            transient_tmp = x[i, 0, :, :, :]
            invpsf_tmp = psf[i, 0, :, :, :]
            albedo = self.lct.transient_to_albedo_net(transient_tmp, invpsf_tmp)
            vol[i, 0, :, :] = torch.max(albedo, 2)[0]

        """ Multiply constant value """
        vol *= 1e05

        """ upsample results to 64*64 for ResNet """
        x = self.upsample(vol)                              # batch_size ×  64 × 64

        return self.resnet(x)                               # batch_size × 128


if __name__ == '__main__':
    print(torch.__version__)

    gpu_index = 0
    device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)

    net = P2PsfNet(out_dim=128, device=device)
    input = torch.ones(5, 1, 32, 32, 64).to(device)
    t0 = time.time()
    out = net(input)

    print(time.time() - t0)
    print(out.shape)
