

import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import glob
import argparse
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from pytorch_grad_cam import    GradCAM, \
                                HiResCAM, \
                                ScoreCAM, \
                                GradCAMPlusPlus, \
                                AblationCAM, \
                                XGradCAM, \
                                EigenCAM, \
                                FullGrad, \
                                LayerCAM, \
                                EigenGradCAM, \
                                GradCAMElementWise
                                
                                
from efficientnet_pytorch import EfficientNet


base_model = EfficientNet.from_name('efficientnet-b0') 

import warnings
warnings.filterwarnings("ignore")

device="cuda" if torch.cuda.is_available() else "cpu"


def parser_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predict_dir",type=str,default=r"/home/gpu_user2/ext_storage/zhiyihe/classfication/data/test/")
    parser.add_argument("--save_dir",type=str,default=r"./visualisation_results/")
    parser.add_argument("--weights",type=str,default="./output/1219/best.pth",help="model path")
    parser.add_argument("--imgsz",type=int,default=128,help="test image size")
    opt=parser.parse_known_args()[0]
    return opt


class cam_Vissualisation_model():

    def __init__(self,opt):
        self.imgsz=opt.imgsz 
        self.img_dir=opt.predict_dir 
        self.save_dir=opt.save_dir 
        ########################################################
        self.model=(torch.load(opt.weights)).to(device) 
        ########################################################
        self.model.eval()
        self.class_name=['1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s'] 
    
    def __call__(self):

        data_transorform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop((128,128)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        img_list=glob.glob(self.img_dir+os.sep+"*.png")
        
        i = 0

        # target_layers = self.model._blocks[-1]._project_conv 
        # target_layers = [self.model._blocks[-1]._bn2] 
        # target_layers = self.model._blocks[-1]._swish 
        # target_layers = getattr(self.model, target_layers)
        # print('target_layers is ', target_layers)
        '''
        Resnet18 and 50: [model.layer4]
        VGG and densenet161: model.features[-1]
        mnasnet1_0: model.layers[-1]
        ViT: model.blocks[-1].norm1

        '''
        
        for imgpath in img_list:
            img=cv2.imread(imgpath)
            img_full_name = imgpath.split('/')[-1]
            img_name = img_full_name.split('.')[0]


            img=Image.fromarray(img)
            img=data_transorform(img) #
            img=torch.reshape(img,(-1,3,self.imgsz,self.imgsz)).to(device) 
            print('img type is ', type(img))
            print('img shape is ', img.shape)

            # cam = GradCAM(model=self.model, 
            #               target_layers=target_layers, 
            #               use_cuda=True)
            

            # target_category = '1990s' # 
            # # 6.
            # grayscale_cam = cam(input_tensor=img,
            #                     target_category=target_category,
            #                     aug_smooth=True,
            #                     eigen_smooth=True)  # [batch, 128, 128]



            # cam.remove_handlers()
            features = []

            k=0
            for name, layer in self.model.named_modules():
                x = layer(img)
                features.append(x)
                k=k+1
                print('layer',k, 'name is ', name)

            # features = self.get_layers_features(img)
            # print(features.shape)
            i += 1
        print("total_num is ", i)
        

    def get_layers_features(self, img):
        # target_layers = self.model.get('blocks')
        # print('target_layers is ', target_layers)
        features = []
        x = img
        # net_depth = len(target_layers)
        for name, layer in self.model.named_modules():
            x = layer(x)
            if name in ['0', '1', '2', '3', '4']:
            # for i in range(net_depth):
                features.append(x)
                print('layer name is ', name)
        return features

if __name__ == '__main__':
    opt=parser_opt()
    test_img=cam_Vissualisation_model(opt)
    test_img()



# CUDA_VISIBLE_DEVICES=0 python cam_visualisation.py 
