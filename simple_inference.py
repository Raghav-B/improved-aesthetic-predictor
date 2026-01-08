import webdataset as wds
from PIL import Image
import io
import sys
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
# from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip

from mlp import MLP


from PIL import Image, ImageFile


#####  This script will predict the aesthetic score for this image file:

img_path = sys.argv[1]  # provide path to image here



def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   


pil_image = Image.open(img_path)

image = preprocess(pil_image).unsqueeze(0).to(device)



with torch.no_grad():
   image_features = model2.encode_image(image)

im_emb_arr = normalized(image_features.cpu().detach().numpy() )

prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

print( "Aesthetic score predicted by the model:")
print( prediction )


