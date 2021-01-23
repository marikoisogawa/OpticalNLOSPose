"""
Function for differentiable argmax.
Note that this function takes 1-d argmax along z(temporal)-axis,
for every spatial points (x, y).

Input: 3D volume (H, W, D)
Output: 2D plane (H, W)
"""

from utils.torch import *
import torch.nn.functional as F


"""
Numpy implementation for soft-argmax for check
"""


def soft_argmax_numpy(input):
    beta = 12
    y_est = input
    a = np.exp(beta * y_est)
    b = np.sum(np.exp(beta * y_est))
    softmax = a / b
    print('softmax numpy', softmax)
    max = np.sum(softmax * y_est)
    print(max)
    pos = range(y_est.size)
    softargmax = np.sum(softmax * pos)
    print(softargmax)


"""
Note that this soft argmax is for z-axis.
Recommended to multiply some large number (like *1000)
to make the output stable.
"""


def soft_argmax(input, device=None):
    soft_max = F.softmax(input, dim=2)
    indices = torch.arange(start=0, end=input.shape[2], device=device)
    # softargmax = torch.sum(soft_max * indices.double(), dim=2).round()
    softargmax = torch.sum(soft_max * indices.double(), dim=2)
    return softargmax


"""
1-D soft-argmax with pytorch implementation
"""


def soft_argmax_1d(input, device=None):
    soft_max = F.softmax(input)
    print('softmax pytprch', soft_max)
    indices = torch.arange(start=0, end=input.shape[1], device=device)
    softargmax = torch.sum(soft_max * indices.float(), dim=1)
    return softargmax
