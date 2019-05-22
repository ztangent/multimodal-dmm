import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn.functional as F

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def anneal(min_val, max_val, t, anneal_len):
    """"Anneal linearly from min_val to max_val over anneal_len."""
    if t >= anneal_len:
        return max_val
    else:
        return (max_val - min_val) * t/anneal_len

def plot_grad_flow(named_parameters, fignum=10):
    """Plots the gradients flowing through different layers in the net
    during training. Can be used for checking for possible gradient vanishing
    / exploding problems.
    
    Usage: Plug this function in after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" 
    to visualize the gradient flow

    src: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    """

    plt.figure(fignum)
    ave_grads = []
    max_grads= []
    nan_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            nan_grads.append(torch.isnan(p.grad).any())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.bar(np.arange(len(max_grads)), nan_grads, alpha=1.0, lw=1, color="r")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max', 'mean', 'nan', 'zero'])
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


# SSIM code below is adapted from https://github.com/VainF/pytorch-msssim/
    
def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    # make it contiguous in y direction for memory efficiency
    out = out.transpose(2, 3).contiguous()
    out = F.conv2d(out, win, stride=1, padding=0, groups=C)
    return out.transpose(2, 3).contiguous()


def _ssim(X, Y, win, data_range=1.0, size_average=False, full=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): batch of images
        Y (torch.Tensor): batch of images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. 
        size_average (bool, optional): if True, average across batch
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    #####################################
    # the 5 convs (blurs) can be combined
    concat_input = torch.cat([X, Y, X*X, Y*Y, X*Y], dim=1)
    concat_win = win.repeat(5, 1, 1, 1).to(X.device, dtype=X.dtype)
    concat_out = gaussian_filter(concat_input, concat_win)

    # unpack from conv output
    mu1, mu2, sigma1_sq, sigma2_sq, sigma12 = (
        concat_out[:, idx*channel:(idx+1)*channel, :, :] for idx in range(5))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def eval_ssim(X, Y, win_size=11, win_sigma=1.5, win=None,
         data_range=1.0, size_average=False, full=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel.
        data_range (float or int, optional): value range of input images. 
        size_average (bool, optional): if True, average across batch
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val
