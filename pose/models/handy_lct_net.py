import os
import numpy as np
from numpy import matlib
import scipy.io as sio
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift, fftn, ifftn
import math
from numpy import linalg
import torch
import time


class HandyLCTNet:

    def __init__(self, spatial_dim=32, temporal_dim=64, device=None, grid_z=False):

        self.m = torch.nn.Upsample(mode='nearest', scale_factor=2)

        """ parameters for pytorch """
        self.device = device
        if self.device is None:
            print('No GPU is assigned.')
        else:
            print(self.device)

        """ parameters for NLOS """
        self.snr = 8e-1
        self.bin_resolution = 32e-12 * (1024 / 64)
        self.c = 3e8
        self.z_trim = 0
        self.width = 1.0
        self.N = spatial_dim
        self.M = temporal_dim
        self.range = self.M * self.c * self.bin_resolution
        self.isdiffuse  = False
        self.isbackprop = False

        """ grid """
        if grid_z:
            self.grid_z_flag = True
            self.grid_z = np.tile(np.linspace(0, 1, self.M), (self.N, self.N, 1))  # 64*32*32
            self.grid_z = self.grid_z.transpose(2, 1, 0)
            if self.isdiffuse:
                self.grid_z = self.grid_z ** 4
            else:
                self.grid_z = self.grid_z ** 2
        else:
            self.grid_z_flag = False

        """ for PSF """
        self.psf, self.fpsf = self.define_psf()
        if self.isbackprop:
            self.invpsf = np.conj(self.fpsf)
        else:
            self.invpsf = np.conj(self.fpsf) / (abs(self.fpsf) ** 2 + 1 / self.snr)
        self.conj_fpsf = np.conj(self.fpsf)
        self.abs_fpsf = abs(self.fpsf) ** 2

        """ sampling operator """
        self.mtx, self.mtxi = self.resampling_operator()

        """ convert matrix from numpy to pytorch """
        if grid_z:
            self.grid_z = torch.from_numpy(self.grid_z).to(self.device)

        self.invpsf = torch.from_numpy(self.invpsf.astype(np.double)).to(self.device)
        self.conj_fpsf = torch.from_numpy(self.conj_fpsf.astype(np.double)).to(self.device)
        self.abs_fpsf = torch.from_numpy(self.abs_fpsf.astype(np.double)).to(self.device)

        self.mtx = torch.from_numpy(self.mtx.toarray()).to(self.device)
        self.mtxi = torch.from_numpy(self.mtxi.toarray()).to(self.device)

        """ variables useful for faster initialization """
        if self.device is not None:
            self.tdata_ = torch.zeros((2 * self.M, 2 * self.N, 2 * self.N), device=self.device) # 2048*64*64
            self.ttdata_fft_ = torch.zeros((2 * self.M, 2 * self.N, 2 * self.N, 2), device=self.device)  # 2048*64*64
        else:
            self.tdata_ = torch.zeros((2 * self.M, 2 * self.N, 2 * self.N))
            self.ttdata_fft_ = torch.zeros((2 * self.M, 2 * self.N, 2 * self.N, 2))

    def define_psf(self):
        slope = self.width / self.range
        x = np.linspace(-1, 1, 2 * self.N)
        y = np.linspace(-1, 1, 2 * self.N)
        z = np.linspace( 0, 2, 2 * self.M)
        grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')

        psf = np.abs(((4*slope)**2) * (grid_x**2 + grid_y**2) - grid_z)

        psf = psf == np.tile(np.min(psf, axis=0, keepdims=True), (2 * self.M, 1, 1))
        psf = psf.astype(np.float32)
        psf = psf / np.sum(psf[:, self.N, self.N])
        psf = psf / linalg.norm(np.ravel(psf), 2)

        psf = np.roll(psf, self.N, axis=1)
        psf = np.roll(psf, self.N, axis=2)

        fpsf = fftn(psf)

        return psf, fpsf

    def resampling_operator(self):
        mtx = lil_matrix((self.M ** 2, self.M))             # set sparse matrix
        mtx_tmp = lil_matrix((self.M ** 2, self.M ** 2))
        x = np.linspace(1, self.M ** 2, self.M ** 2)

        # set non-zero elements
        for i in range(self.M ** 2):
            mtx[int(x[i])-1, math.ceil(math.sqrt(x[i]))-1] = 1
            mtx_tmp[i, i] = 1 / math.sqrt(x[i])

        # convert lil_matrix to csr_matrix
        mtx = mtx.tocsr()
        mtx_tmp = mtx_tmp.tocsr()

        mtx = mtx_tmp @ mtx
        mtxi = mtx.T

        K = math.log(self.M) / math.log(2)
        for k in range(0, int(K)):
            mtx  = 0.5 * (mtx[::2, :]  + mtx[1::2, :])
            mtxi = 0.5 * (mtxi[:, ::2] + mtxi[:, 1::2])

        return mtx, mtxi

    def transient_to_albedo(self, transient, invpsf_updated=None):

        data = transient.permute(2, 1, 0).double()  # 32*32*64

        """ Skip step 1 (scale radiometric component) to reduce memory """
        if self.grid_z_flag:
            data = torch.mul(data.double(), self.grid_z_small.double())

        """ Step 2: Resampling along t axis """
        tdata = self.tdata_
        tdata[:self.M, :self.N, :self.N] = \
            torch.reshape(torch.matmul(self.mtx, torch.reshape(data, (self.M, self.N*self.N))), (self.M, self.N, self.N))

        """ Step 3: Convolve with inverse filter and unpad result (mul is element-wise) """
        tdata_fft = torch.rfft(tdata, 3, onesided=False)
        ttdata_fft = self.ttdata_fft_
        if invpsf_updated is not None:
            ttdata_fft[:, :, :, 0] = torch.mul(tdata_fft[:, :, :, 0].double(), invpsf_updated.double())
            ttdata_fft[:, :, :, 1] = torch.mul(tdata_fft[:, :, :, 1].double(), invpsf_updated.double())
        else:
            ttdata_fft[:, :, :, 0] = torch.mul(tdata_fft[:, :, :, 0].double(), self.invpsf.double())
            ttdata_fft[:, :, :, 1] = torch.mul(tdata_fft[:, :, :, 1].double(), self.invpsf.double())
        ttvol = torch.irfft(ttdata_fft, signal_ndim=3, signal_sizes=tdata.shape, onesided=False)
        tvol = ttvol[0:self.M, 0:self.N, 0:self.N]

        """ Step 4: Resample depth axis and clamp results (R_z^{-1}?) """
        talbedo = torch.reshape(torch.matmul(self.mtxi.double(), torch.reshape(tvol, (self.M, self.N * self.N)).double()), (self.M, self.N, self.N))
        albedo = talbedo.permute(2, 1, 0)

        return albedo

    def transient_to_albedo_net(self, transient, invpsf_updated=None):

        data = transient.permute(2, 1, 0).double()  # 32*32*64

        """ Skip step 1 (scale radiometric component) to reduce memory """
        # if self.grid_z_flag:
        #     data = torch.mul(data.double(), self.grid_z_small.double())

        """ Step 2: Resampling along t axis """
        tdata = torch.zeros((2 * self.M, 2 * self.N, 2 * self.N), device=self.device)
        tdata_reshaped0 = torch.reshape(data, (self.M, self.N*self.N))
        tdata_resampled = torch.matmul(self.mtx, tdata_reshaped0)
        tdata_reshaped1 = torch.reshape(tdata_resampled, (self.M, self.N, self.N))
        tdata[:self.M, :self.N, :self.N] = tdata_reshaped1

        """ Step 3: Convolve with inverse filter and unpad result (mul is element-wise) """
        tdata_fft = torch.rfft(tdata, 3, onesided=False)
        ttdata_fft = torch.zeros((2 * self.M, 2 * self.N, 2 * self.N, 2), device=self.device)  # 2048*64*64
        if invpsf_updated is not None:
            ttdata_fft[:, :, :, 0] = torch.mul(tdata_fft[:, :, :, 0].double(), invpsf_updated.double())
            ttdata_fft[:, :, :, 1] = torch.mul(tdata_fft[:, :, :, 1].double(), invpsf_updated.double())
        else:
            ttdata_fft[:, :, :, 0] = torch.mul(tdata_fft[:, :, :, 0].double(), self.invpsf.double())
            ttdata_fft[:, :, :, 1] = torch.mul(tdata_fft[:, :, :, 1].double(), self.invpsf.double())
        ttvol = torch.irfft(ttdata_fft, signal_ndim=3, signal_sizes=tdata.shape, onesided=False)
        tvol = ttvol[0:self.M, 0:self.N, 0:self.N]

        """ Step 4: Resample depth axis and clamp results """
        talbedo_reshaped0 = torch.reshape(tvol, (self.M, self.N * self.N)).double()
        talbedo_resampled = torch.matmul(self.mtxi.double(), talbedo_reshaped0)
        talbedo_reshaped1 = torch.reshape(talbedo_resampled, (self.M, self.N, self.N))

        albedo = talbedo_reshaped1.permute(2, 1, 0)

        return albedo