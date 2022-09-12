
import torch
from torch.distributions import Uniform, TransformedDistribution, SigmoidTransform, AffineTransform



class LogisticDistribution(TransformedDistribution):
    def __init__(self, loc, scale):
        base_distribution = Uniform(torch.zeros_like(loc), torch.ones_like(loc))
        transforms = [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]
        super(LogisticDistribution, self).__init__(base_distribution, transforms)

    def log_prob(self, value):
        return super(LogisticDistribution, self).log_prob(value).sum(-1)


def log_sum_exp(x):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x, axis= 0):
    """numerically stable log_softmax implementation that prevents overflow""" #Looks like he is softmaxing on the channel distribution
    # TF ordering
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))