import torch
from torchvision.models.resnet import resnet50
from torchvision import transforms
import os
import cv2
import numpy as np
from scipy.special import entr, xlogy

imgpath = "/home/juyoung/ImageNet-O/imagenet-o"
# imgpath = "/home/juyoung/DTD/dtd/images"
testfile = open("resizencrop_resnet_imagenet-o.csv",'w')
device = torch.device('cuda:0')

transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(   
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                                ])

y = resnet50(pretrained=True)
y.eval()
y.to(device)


for (path,dir,files) in os.walk(imgpath):
    for file in files: 
        if '.JPEG' in file: 
        # if '.jpg' in file: 
            Img = path + "/"+ file
            
            rgb_img = cv2.imread(Img, 1)
        
            rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
            
            rgb_img = np.float32(rgb_img) / 255
            transforming = transforms.ToTensor()
    
            rgb_img = transforming(rgb_img)
            img = transform(rgb_img)
            imgtensor = img.unsqueeze(0)
            imgtensor = imgtensor.to(device)
            out = y(imgtensor)
            softmaxout= torch.nn.functional.softmax(out, dim =-1)
            

            entropy = softmaxout*torch.log(softmaxout+1e-8)*(-1)
            entropysum = entropy.sum()
            entropydata = float(entropysum.detach().cpu().numpy())
            print(entropydata)

            dirdata = Img + " "
            testfile.write(str(dirdata))

            testfile.write(str(entropydata) + "\n")