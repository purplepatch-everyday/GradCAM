import argparse
import cv2
from matplotlib.pyplot import gray
import numpy as np
import torch
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
  
    # Create file
    # maxfile = open ("resize_texture.csv",'w')
    maxfile = open("resizencrop_texture.csv",'w')
    # maxfile = open("NEWdtdmax.csv",'w')

    # for (path, dir, files) in os.walk("/home/juyoung/ImageNet-O/imagenet-o"):
    for (path, dir, files) in os.walk("/home/juyoung/DTD/dtd/images"):
        for file in files:
            # ext = os.path.splitext(file)[-1]
            # if '.JPEG' in file:
            if '.jpg' in file:
                curImgPath = path + "/" + file 
                print(curImgPath)

###################################################################### 
                args = get_args()
                methods = GradCAM

                model = models.resnet50(pretrained=True)

                target_layers = [model.layer4]
                
                rgb_img = cv2.imread(curImgPath, 1)
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
                                use_cuda=args.use_cuda) as cam:
             
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        targets=targets,
                                        aug_smooth=False,
                                        eigen_smooth=False)

                 
                    grayscale_cam = grayscale_cam[0, :]

                    cam_image = show_cam_on_image(to_plot, grayscale_cam, use_rgb=True)

                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
                gb = gb_model(input_tensor, target_category=None)

                cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
                cam_gb = deprocess_image(cam_mask * gb)
                gb = deprocess_image(gb)

                # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
                # cv2.imwrite(f'{args.method}_gb.jpg', gb)
                # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
                
                #test
                
                # # min value
                # print('min:',np.min(grayscale_cam))
                # mindata = str(np.min(grayscale_cam)) + " "
                # minfile.write(mindata)
                
                # # mean value
                # print('mean:',np.mean(grayscale_cam))
                # meandata = str(np.mean(grayscale_cam)) +" "
                # meanfile.write(meandata)
                
                # max value
                print('max:',np.max(grayscale_cam))
                dirdata = curImgPath +" "
                maxfile.write(dirdata)
                maxdata = str(np.max(grayscale_cam)) + "\n"
                maxfile.write(maxdata)