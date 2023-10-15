import torch
import os
import torchvision
import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
import argparse
import shutil
device="cuda" if torch.cuda.is_available() else "cpu"
#
def parser_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predict_dir",type=str,default=r"/home/gpu_user2/ext_storage/zhiyihe/classfication/data/category_train_jpg/9")
    parser.add_argument("--save_dir",type=str,default=r"/home/gpu_user2/ext_storage/zhiyihe/classfication/data/infer_classfication_results/category_cls4/")
    parser.add_argument("--weights",type=str,default="./output/category_cls4/0110_128_0.8560/best.pth",help="model path")
    parser.add_argument("--imgsz",type=int,default=128,help="test image size")
    opt=parser.parse_known_args()[0]
    return opt
#测试图片
class Test_model():

    def __init__(self,opt):
        self.imgsz=opt.imgsz #
        self.img_dir=opt.predict_dir #
        self.save_dir=opt.save_dir #

        self.model=(torch.load(opt.weights)).to(device) #

        self.model.eval()
        # self.class_name=['1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s'] #
        # self.class_name=['1970s', '1980s', '1990s', 'after2000', 'before1970']
        self.class_name=['1', '2', '3+4', '5']
    def __call__(self):
        #
        data_transorform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop((128,128)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        img_list=glob.glob(self.img_dir+os.sep+"*.jpg")
        total_num = 0
        print('Processing... Please wait a moment')
        for imgpath in img_list:
            img=cv2.imread(imgpath)
            img_full_name = imgpath.split('/')[-1]
            # print(img_full_name)
            img_name = img_full_name.split('.')[0]
            # print(img_name)
            # new_img=self.expend_img(img) #
            img=Image.fromarray(img)
            img=data_transorform(img) #
            img=torch.reshape(img,(-1,3,self.imgsz,self.imgsz)).to(device) #[B,C,H,W]
            ##########################
            pred, feature = self.model(img)
            ##########################
            _,pred=torch.max(pred,1)
            pred_cls_name = self.class_name[pred]
            # print("Image path:",imgpath," pred:",pred_cls_name) # 
            save_classdir = self.save_dir + pred_cls_name 
            shutil.copyfile(imgpath, os.path.join(save_classdir, img_name + ".jpg"))

            #######
            # self.draw_cls_name(save_classdir, img_name, pred_cls_name)
            # print class_name on images 
            # saved_image = os.path.join(save_classdir, img_name + ".jpg")
            # tp = Image.open(saved_image)
            # draw = ImageDraw.Draw(tp)
            # draw.text((0, 0), pred_cls_name ,(255,255,0))
            # tp.save(saved_image)

            total_num += 1
        print("total_num: ", total_num)

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

    def draw_cls_name(img_save_path, img_name, class_name):
        '''
        '''
        saved_image = os.path.join(img_save_path, img_name + ".jpg")
        tp = Image.open(saved_image)
        draw = ImageDraw.Draw(tp)
        draw.text((0, 0), class_name ,(255,255,0))
        tp.save(saved_image)


if __name__ == '__main__':
    opt=parser_opt()
    test_img=Test_model(opt)
    test_img()
    
# CUDA_VISIBLE_DEVICES=0 python predict.py 
