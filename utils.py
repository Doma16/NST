import torch

def GramMatrix(img):
    b, c, h, w = img.shape
    imgt = img.clone()

    assert b == 1
    for i in range(c):
        imgt[0][i] = imgt[0][i].t()

    gm = torch.matmul(img,imgt)
    
    return gm
