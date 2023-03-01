import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.utils import save_image

from time import perf_counter

from PIL import Image

from model import NST


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

IMG_SIZE = 256
CHANNELS = 3

LR = 5e-3
STEPS = 5000

ALPHA = 1
BETA = 0.04

PATH_REAL = f'./pics/myself.jpg'
PATH_STYLE = f'./pics/Vincent_van_Gogh_100.jpg'

transform = transforms.Compose(
    [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    ]
)

real_img = Image.open(PATH_REAL)
style_img = Image.open(PATH_STYLE)

real_img = transform(real_img).view(1,CHANNELS,IMG_SIZE,IMG_SIZE)
style_img = transform(style_img).view(1,CHANNELS,IMG_SIZE,IMG_SIZE)

gen_img = real_img.clone().requires_grad_(True)
#gen_img = torch.randn(real_img.shape).requires_grad_(True)

model = NST()

nst_opt = optim.Adam([gen_img], lr=LR)

start = perf_counter()
for step in range(STEPS):


    real_part = model(real_img)
    style_part = model(style_img)
    gen_part = model(gen_img)    

    content_loss = 0
    style_loss = 0

    for r_c, s_c, g_c in zip(real_part, style_part, gen_part):

        content_loss += torch.mean((r_c - g_c)**2)
        
        batch, channels, height, width = g_c.shape


        s_c = s_c.view(channels, height*width)
        g_c = g_c.view(channels, height*width)

        ss_c = torch.matmul(s_c,s_c.t())
        gg_c = torch.matmul(g_c,g_c.t())

        style_loss += torch.mean((ss_c - gg_c)**2)

    loss = ALPHA * content_loss + BETA * style_loss

    nst_opt.zero_grad()
    loss.backward()
    nst_opt.step()

    if step % 200 == 0:
        end = perf_counter()
        print(f'Loss: {loss:.3f}, Time: {end-start:.3f} ')
        save_image(gen_img,f'./gen_pics/step{step}_generated.png')
        start = perf_counter()
    
