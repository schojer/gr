

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
# import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings("ignore")

device="cuda" if torch.cuda.is_available() else "cpu"

#
def parser_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predict_dir",type=str,default=r"dataset/11classes/32x32/val/industrial/")
    parser.add_argument("--save_dir",type=str,default=r"./visualisation_results/32x32/")
    parser.add_argument("--weights",type=str,default="output/11classes/128x128/0421/best.pth",help="model path")
    parser.add_argument("--imgsz",type=int,default=128,help="test image size")
    opt=parser.parse_known_args()[0]
    return opt

#
class Vissualisation_model():

    def __init__(self,opt):
        self.imgsz=opt.imgsz #
        self.img_dir=opt.predict_dir #
        self.save_dir=opt.save_dir #
        ########################################################
        self.model=(torch.load(opt.weights)).to(device) #加载模型
        ########################################################
        self.model.eval()
        self.class_name=['commercia+office', 
                         'education_institution', 
                         'hight_private', 'industrial', 
                         'low_private', 
                         'medical', 'mixed-used', 
                         'public_house', 
                         'public_services', 
                         'recreation', 
                         'religion'] #
        # self.class_name=['1', '2', '3', '4', '5', '9'] #
    def __call__(self):
        #
        data_transorform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop((128,128)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        # img_list=glob.glob(self.img_dir+os.sep+"*.png")
        img_list = []
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith('.png'):
                    img_list.append(os.path.join(root, file))
        # print(img_list)
        i = 0
        classes_images = (0, self.imgsz * self.imgsz)
        classes_features = np.empty((0, 1281))
        
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
            ###################################################
            # num_class = 51
            # fc = getattr(model, 'fc')
            # feature_dim = fc.in_features
            # setattr(model,'fc',nn.Linear(feature_dim,num_class))
            # print(model)

            pred, feature = self.model(img)
            
            _,pred=torch.max(pred,1)
            pred_cls_name = self.class_name[pred]
            # print("pre is:",pred," pred data type is:",pred.dtype)
        
            pred1 = pred.cpu().detach().numpy()
            feature1 = feature.cpu().detach().numpy()
            
            #
            class_feature = np.insert(feature1, 0, pred1) 
            class_feature = np.expand_dims(class_feature, axis=0) #  [1, 1281]

            ## [img_num, 1281]
            classes_features = np.concatenate((classes_features, class_feature), axis=0) 

            ###################################################
            print("Image path:",imgpath," pred:",pred_cls_name)

            i += 1
        print("total_num is : ", i)             
        print("classes_features shape is ", classes_features.shape)

        # pca 
        principalComponents, classes= self.pca_visialisation(classes_features)
        # 
        plt.scatter(principalComponents[:, 0], 
                    principalComponents[:, 1], 
                    s=5, # 
                    c=classes, 
                    cmap='Spectral')
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(10)-0.5).set_ticks(np.arange(10)) # 
        plt.title('Visualizing Features through PCA', fontsize=18)
        plt.savefig(self.save_dir + str(classes_features.shape[0]) + "_" + str(classes_features.shape[1]-1) + "_" + "pca.png",
                    # bbox_inches = 'tight',
                    # pad_inches = 0,
                    dpi=600)   
        
        # t-SNE
        tsne, classes= self.tSNE_visialisation(classes_features)
        # 
        plt.scatter(tsne[:, 0], 
                    tsne[:, 1], 
                    s=5, # 
                    c=classes, 
                    cmap='Spectral')
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(9)-0.5).set_ticks(np.arange(8)) #
        plt.title('Visualizing Features through t-SNE', fontsize=18)
        plt.savefig(self.save_dir + str(classes_features.shape[0]) + "_" + str(classes_features.shape[1]-1) + "_" + "t-SNE.png",
                    # bbox_inches = 'tight',
                    # pad_inches = 0,
                    dpi=300)

        # UMAP
        # reducer, embedding, classes= self.umap_visialisation(classes_features)
        # # 
        # plt.scatter(reducer.embedding_[:, 0], 
        #             reducer.embedding_[:, 1], 
        #             s=5, 
        #             c=classes, 
        #             cmap='Spectral')
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.colorbar(boundaries=np.arange(9)-0.5).set_ticks(np.arange(8))
        # plt.title('Visualizing Features through UMAP', fontsize=18)
        # plt.savefig(self.save_dir + str(classes_features.shape[0]) + "_" + str(classes_features.shape[1]-1) + "_" + "t-SNE.png",
        #             # bbox_inches = 'tight',
        #             # pad_inches = 0,
        #             dpi=300)
    

    def pca_visialisation(self, classes_features):
        classes = classes_features[:, 0]
        features = classes_features[:, 1:]
        pca = PCA(n_components=2)  # project from 1280 to 2 dimensions
        principalComponents = pca.fit_transform(features)
        return principalComponents, classes


    def tSNE_visialisation(self, classes_features):
        classes = classes_features[:, 0]
        features = classes_features[:, 1:]
        pca_50 = PCA(n_components=10)
        pca_result_50 = pca_50.fit_transform(features)
        tsne = TSNE(random_state = 42, 
                    n_components=2, 
                    verbose=0, 
                    perplexity=40, 
                    n_iter=300
                    ).fit_transform(pca_result_50) 
        return tsne, classes  


    # def umap_visialisation(self, classes_features):
    #     classes = classes_features[:, 0]
    #     features = classes_features[:, 1:]
    #     reducer = umap.UMAP(random_state=10)
    #     embedding = reducer.fit_transform(features)
    #     return reducer, embedding, classes

    #     embedding = pd.DataFrame(embedding, index=classes_features.index)
        

    def expend_img(self,img,fill_pix=122):
        '''

        :return:
        '''
        h,w=img.shape[:2] 
        if h>=w: 
            padd_width=int(h-w)//2
            padd_h,padd_b,padd_l,padd_r=0,0,padd_width,padd_width 

        elif h<w: 
            padd_high=int(w-h)//2
            padd_h,padd_b,padd_l,padd_r=padd_high,padd_high,0,0

        new_img = cv2.copyMakeBorder(img, padd_h, padd_b, padd_l, padd_r, borderType=cv2.BORDER_CONSTANT,
                                     value=[fill_pix,fill_pix,fill_pix])
        return new_img


if __name__ == '__main__':
    opt=parser_opt()
    test_img=Vissualisation_model(opt)
    test_img()
    

