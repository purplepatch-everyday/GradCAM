#%%
import argparse
import pandas as pd           
import matplotlib as mpl         
import matplotlib.pyplot as plt  
import os
import numpy as np
import json
from torch import alpha_dropout
import cv2
import torch
import torchvision.transforms as transforms

###################DTDmax#######################
# df1 = pd.read_csv('DTDmax.csv', names=['value'])

# plt.hist(df1,bins=800)
# plt.title("DTD")
# plt.xlim([0,7])
# plt.ylim([0,40])
# plt.xlabel("Max value")
# plt.ylabel("# of data")
# # plt.show()

# # print(df.head(5))
# # print("max:" ,df.max())
# # print("min:",df.min())
# print(df1['value'].dtype)
# print(df1['value'].describe(include='all'))

################### Imagenetomax #################
# df2 = pd.read_csv('NEWimagenetomax.csv', names=['dir','value'], sep=" ")
# print("ImageNet-O")
# print(df2['value'].dtype)
# print(df2['value'].describe(include='all'))
# print(df2.head())
# plt.hist(df2['value'],bins=1000,alpha=0.5)
# maxindex=df2["value"].idxmax()
# print(df2.iloc[maxindex+1])

# df2=df2.sort_values(by=['value'], axis=0,ascending=False)
# print(df2.head())
# print("tail:",df2.tail())
# print(df2)
# 2000rows * 2 columns

#first5 = [i["dir"] for i in df2][:5]

# top5= df2[:5]
# print(top5)
# print("=========")
# top5= top5["dir"]
# top5 = top5.tolist()
#print(top5)
#print(type(top5))
#print("first value : ",top5[:1])

# last5=df2[-5:]
# last5=last5["dir"]
# print(last5)
################ imagenet.json ######################
# with open('imagenet.json','r') as f:
#     json_data=json.load(f)

# imagenetmax =[]
# for i in range(len(json_data['max'])):
#     imagenetmax.append(json_data['max'][i])

# df3 = pd.DataFrame({"max":imagenetmax})
# print("ImageNet")
# # print(df3['max'].dtype)
# print(df3['max'].describe(include='all'))

# plt.hist(df3,bins=5000,alpha=0.5)

######################### plot and show ###############################
# plt.xlim([0,2.5])
# # plt.ylim([0,40])
# plt.xlabel("Max value")
# plt.title("Grad CAM")
# plt.show()
#######################################################################
curImgPath = '/home/juyoung/GradCAM/pytorch-grad-cam/imageneto_result_image/original19.863325.jpg'
rgb_img = cv2.imread(curImgPath, 1)
rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
rgb_img = np.float32(rgb_img) / 255

transform = transforms.ToTensor()
rgb_img = transform(rgb_img)
print(rgb_img)
# rgb_img = np.float32(rgb_img) / 255

# print(rgb_img.shape)
# rgb_img - torch.from_numpy(rgb_img).float()
# print(type(rgb_img))
        



#%%
