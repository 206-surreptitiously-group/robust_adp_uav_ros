import numpy as np
import matplotlib.pyplot as plt
import torch

# dx = 2x + u
if __name__ == '__main__':
    x1 = torch.Tensor([1])
    x2 = torch.Tensor([2])
    x3 = torch.Tensor([3])
    print(torch.cat((x1, x2, x3)).unsqueeze(1).shape)
