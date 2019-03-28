import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
