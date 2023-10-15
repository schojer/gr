
import os
import cv2
import glob
import torch
import numpy as np
'''

pip install grad-cam==1.3.6
'''
from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from torchvision.models import resnet50, \
                               densenet161, \
                               mnasnet1_0, \
                               vgg16

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings("ignore")

device="cuda" if torch.cuda.is_available() else "cpu"


model = resnet50(pretrained=True) 


# target_layer = model.layer4[-1]
target_layer = [model.layer4]

'''
Resnet18 and 50: [model.layer4]
VGG and densenet161: model.features[-1]
mnasnet1_0: model.layers[-1]
ViT: model.blocks[-1].norm1

'''

'''

'''
img_dir = r"/home/gpu_user2/ext_storage/zhiyihe/classfication/data/test/"
save_dir = r"/home/gpu_user2/ext_storage/zhiyihe/classfication/data/test_GradCAM/"
# image_size = 128
img_list=glob.glob(img_dir+os.sep+"*.png")

i = 0
print('Processing...   Please Wait.')
for imgpath in img_list:
    rgb_img = cv2.imread(imgpath, 1)[:, :, ::-1]   
                                                 
    rgb_img = cv2.imread(imgpath, 1) 
    rgb_img = np.float32(rgb_img) / 255
 
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])   # torch.Size([1, 3, 128, 128])
    # print(input_tensor.shape)
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    img_full_name = imgpath.split('/')[-1]
    img_name = img_full_name.split('.')[0]

    '''

    '''
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

    '''

    '''
    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.


    target_category = None

    '''

    '''
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    '''

    '''
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
    # print(visualization.shape)
    cv2.imwrite(save_dir + str(img_name) + "_" + "GradCAM.png", visualization)

    i += 1
print("total_num is ", i)
print(input_tensor.shape)
print(visualization.shape)


# CUDA_VISIBLE_DEVICES=0 python cam_visualisation.py
