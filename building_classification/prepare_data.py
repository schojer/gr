
#
import os
import glob
import cv2
import random
from pathlib import Path


# 
def expend_img(img):
    '''
    :param img: 
    :return:
    '''
    fill_pix = [122, 122, 122]  # 
    h, w = img.shape[:2]
    if h >= w:  # 
        padd_width = int(h-w)//2
        padd_top, padd_bottom, padd_left, padd_right = 0, 0, padd_width, padd_width  # 
    elif h < w:  # 
        padd_high = int(w-h)//2
        padd_top, padd_bottom, padd_left, padd_right = padd_high, padd_high, 0, 0  # 

    new_img = cv2.copyMakeBorder(
        img, padd_top, padd_bottom, padd_left, padd_right, cv2.BORDER_CONSTANT, value=fill_pix)
    return new_img


# 
# def split_train_test(img_dir,save_dir,train_val_num):
#     '''
#     :param img_dir: 
#     :param save_dir:
#     :param train_val_num: 
#     :return:
#     '''
#     img_dir_list=glob.glob(img_dir+os.sep+"*")#
#     for class_dir in img_dir_list:
#         class_name=class_dir.split(os.sep)[-1] #
#         img_list=glob.glob(class_dir+os.sep+"*") #
#         all_num=len(img_list) #
#         train_list=random.sample(img_list,int(all_num*train_val_num)) #
#         save_train=save_dir+os.sep+"train"+os.sep+class_name
#         save_val=save_dir+os.sep+"val"+os.sep+class_name
#         os.makedirs(save_train,exist_ok=True)
#         os.makedirs(save_val,exist_ok=True) #
#         print(class_name+" trian num",len(train_list))
#         print(class_name+" val num",all_num-len(train_list))
#         #
#         for imgpath in img_list:
#             imgname=Path(imgpath).name #
#             if imgpath in train_list:
#                 img=cv2.imread(imgpath)
#                 new_img=expend_img(img)
#                 cv2.imwrite(save_train+os.sep+imgname,new_img)
#             else: #
#                 img = cv2.imread(imgpath)
#                 new_img = expend_img(img)
#                 cv2.imwrite(save_val + os.sep + imgname, new_img)

#     print("split train and val finished !")

def split_train_test(img_dir, save_dir, train_val_num):
    '''
    :param img_dir: 
    :param save_dir: 
    :param train_val_num: 
    :return:
    '''
    img_dir_list = glob.glob(img_dir+os.sep+"*")  # 
    for class_dir in img_dir_list:
        class_name = class_dir.split(os.sep)[-1]  # 
        img_list = glob.glob(class_dir+os.sep+"*")  # 
        all_num = len(img_list)  # 
        # train_list=random.sample(img_list,int(all_num*train_val_num)) #
        with open("/home/gpu_user2/ext_storage/zhiyihe/classfication/Awesome-Backbones/datas/train.txt", "r") as f:
            train_list = f.readlines()
            # print(train_list)
        train_list = [line.strip().split()[0]
                      for line in train_list if line.strip() != ""]
        # print(train_list)
        train_list = [line for line in train_list if class_name in line]

        save_train = save_dir+os.sep+"train"+os.sep+class_name
        save_val = save_dir+os.sep+"val"+os.sep+class_name
        os.makedirs(save_train, exist_ok=True)
        os.makedirs(save_val, exist_ok=True)  # 
        print(class_name+" trian num", len(train_list))
        print(class_name+" val num", all_num-len(train_list))
        # 
        for imgpath in img_list:
            imgname = Path(imgpath).name  #
            if imgpath in train_list:
                img = cv2.imread(imgpath)
                new_img = expend_img(img)
                cv2.imwrite(save_train+os.sep+imgname, new_img)
            else:  # 
                img = cv2.imread(imgpath)
                new_img = expend_img(img)
                cv2.imwrite(save_val + os.sep + imgname, new_img)

    print("split train and val finished !")


if __name__ == '__main__':
    img_dir = "/home/gpu_user2/ext_storage/zhiyihe/classfication/data/11classes/"
    save_dir = "./dataset/"
    train_val_num = 0.8
    split_train_test(img_dir, save_dir, train_val_num)

# python prepare_data.py

