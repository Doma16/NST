import torch
import numpy as np
import matplotlib.pyplot as plt

def GramMatrix(img):
    b, c, h, w = img.shape
    assert b == 1

    imgt = torch.transpose(img.squeeze(0), 1, 2).unsqueeze(0)
    gm = torch.matmul(img,imgt)

    return gm
