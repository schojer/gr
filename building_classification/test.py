import torch
import os
import torchvision
import glob
from PIL import Image
import cv2
import argparse
device="cuda" if torch.cuda.is_available() else "cpu"
#
def parser_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--test-dir",type=str,default=r"./dataset/val/2000s")
    parser.add_argument("--weights",type=str,default="./output/1219/best.pth",help="model path")
    # parser.add_argument("--weights",type=str,default="./pre_models/efficientnet-b0-355c32eb.pth",help="model path")
    parser.add_argument("--imgsz",type=int,default=128,help="test image size")
    opt=parser.parse_known_args()[0]
    return opt
#
class Test_model():

    def __init__(self,opt):
        self.imgsz=opt.imgsz #
        self.img_dir=opt.test_dir #

        self.model=(torch.load(opt.weights)).to(device) #
        self.model.eval()
        self.class_name=['1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s'] #
    
    def __call__(self):
        #
        data_transorform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop((128,128)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        img_list=glob.glob(self.img_dir+os.sep+"*.png")
      
        for imgpath in img_list:
            img=cv2.imread(imgpath)
            # new_img=self.expend_img(img) #
            img=Image.fromarray(img)
            img=data_transorform(img) #
            img=torch.reshape(img,(-1,3,self.imgsz,self.imgsz)).to(device) #[B,C,H,W]
            pred, feature = self.model(img)
            _,pred=torch.max(pred,1)
            outputs = self.class_name[pred]
            print("Image path:",imgpath," pred:",outputs)

    #
    def expend_img(self,img,fill_pix=122):
        '''
        :param img: 
        :param fill_pix: 
        :return:
        '''
        h,w=img.shape[:2] #
        if h>=w: #
            padd_width=int(h-w)//2
            padd_h,padd_b,padd_l,padd_r=0,0,padd_width,padd_width #

        elif h<w: #
            padd_high=int(w-h)//2
            padd_h,padd_b,padd_l,padd_r=padd_high,padd_high,0,0

        new_img = cv2.copyMakeBorder(img, padd_h, padd_b, padd_l, padd_r, borderType=cv2.BORDER_CONSTANT,
                                     value=[fill_pix,fill_pix,fill_pix])
        return new_img

if __name__ == '__main__':
    opt=parser_opt()
    test_img=Test_model(opt)
    test_img()

# CUDA_VISIBLE_DEVICES=0 python test.py 
