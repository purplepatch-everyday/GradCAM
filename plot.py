#%%
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import json

##### distribution #####

# ID data: ImageNet
IDdir = "/home/juyoung/GradCAM/pytorch-grad-cam/imagenet/imagenet_gradcam_resize.json"
with open(IDdir,'r') as IDdata: 
    json_data = json.load(IDdata)

imagenet = []
for i in range(len(json_data['max'])):
    imagenet.append(json_data['max'][i])

# OOD data1: ImageNet-O 
ood1dir = "/home/juyoung/GradCAM/pytorch-grad-cam/imageneto/resize_gradcam_imageneto.csv"
ood1 = pd.read_csv(ood1dir,names =["values"],sep=" ")

# OOD data2: Describable Texture Dataset(DTD) 
ood2dir = "/home/juyoung/GradCAM/pytorch-grad-cam/DTD/resize_texture.csv"
ood2 = pd.read_csv(ood2dir,names=["values"],sep = " ")

# plot histogram

plt.hist(imagenet,bins = 1000, alpha=0.5, label="ID: ImageNet")
plt.hist(ood1, bins= 1000, alpha = 0.5, label = "OOD1: ImageNet-o")
plt.hist(ood2, bins= 1000, alpha = 0.5, label = "OOD2: Describable Texture")

plt.xlim([0,1.5])

plt.legend()
plt.show()

# %%
