import argparse
import cv2
from matplotlib.pyplot import gray
import numpy as np
import torch
from torchvision import models,transforms
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import pandas as pd

###################################################################### 
# df2 = pd.read_csv('NEWimagenetomax.csv', names=['dir','value'], sep=" ")
df2 = pd.read_csv('/home/juyoung/GradCAM/pytorch-grad-cam/imageneto/resize_gradcam_imageneto.csv', names=['dir','value'], sep=" ")
df2=df2.sort_values(by=['value'], axis=0,ascending=False)

###### BEST 3 #####
top3= df2[:3]
top3= top3["dir"]
top3 = top3.tolist()

# ##### LAST 5 #####
# last5=df2[-5:]
# last5=last5["dir"]
# last5= last5.tolist()

for curImgPath in top3:
# for curImgPath in last5:
    print(curImgPath)
    model = models.resnet50(pretrained=True)
    target_layers = [model.layer4]
    ##### Read Image #####
    test= cv2.imread(curImgPath)
    rgb_img = cv2.imread(curImgPath,1)
    rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(rgb_img) / 255

    transform = transforms.ToTensor()
    rgb_img = transform(rgb_img)

    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    to_plot = input_tensor.squeeze().permute(1,2,0)*torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    to_plot = to_plot.numpy()
    targets = None

    cam_algorithm = GradCAM
    with cam_algorithm(model=model,
                    target_layers=target_layers,
                    use_cuda=True) as cam:

        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=None,
                            aug_smooth=False,
                            eigen_smooth=False)
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    dir = "/home/juyoung/GradCAM/pytorch-grad-cam/imageneto/results"
    # dir = "/home/juyoung/GradCAM/pytorch-grad-cam/dtd_result_image"
    maxvalue=str(np.max(grayscale_cam))
    cv2.imwrite(f'{dir}/best'+ maxvalue +'.jpeg', cam_image)
    # cv2.imwrite(f'{dir}/gb'+maxvalue+'.jpg', gb)
    # cv2.imwrite(f'{dir}/cam_gb'+maxvalue+'.jpg', cam_gb)
    # cv2.imwrite(f'{dir}/'+ maxvalue +'.jpg',test)
    # df2['calculated'] = np.max(grayscale_cam)
    # print(curImgPath + " max: ", np.max(grayscale_cam))
    print(test.shape)
    
################################################################################