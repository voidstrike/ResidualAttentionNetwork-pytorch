import torch
from torch.autograd import Variable
from model.attn_module import residual_attn_module


def main():
    tm = residual_attn_module(4, 4, (16, 16), (8, 8))
    tin = Variable(torch.tensor(1).new_full((10, 4, 16, 16), 1)).float()
    out = tm(tin)
    print(out.shape)

main()
