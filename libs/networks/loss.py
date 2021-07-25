import sys
import torch
from torch import nn

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp/torch.sum(inp)    # input
        trg = trg/torch.sum(trg)    # target
        eps = sys.float_info.epsilon    # sys.float_info.epsilon是机器可以区分出的两个浮点数的最小区别,约等于0

        return torch.sum(trg*torch.log(eps+torch.div(trg,(inp+eps))))   # torch.div是pytorch中的除法，所以这里算出来的就是KL原公式

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)
