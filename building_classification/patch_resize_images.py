import os
import cv2
from pathlib import Path

def resize_replace_images(folder_path, downsample_scale=3):
    '''
    :param folder_path: 
    :param downsample_scale: 
    :return:
    '''
    img_extensions = ('.jpg', '.jpeg', '.png') # 
    for root, dirs, files in os.walk(folder_path):
        num = 0
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(img_extensions):
                img = cv2.imread(file_path) # 
                h, w, c = img.shape
                # 
                img_downsampled = cv2.resize(img, (w//downsample_scale, h//downsample_scale))
                # 
                img_resized = cv2.resize(img_downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
                # 
                cv2.imwrite(file_path, img_resized)
                num += 1
        print("resize_replace_images: %d images resized in %s" % (num, root))

if __name__ == '__main__':
    resize_replace_images(r"/home/gpu_user2/ext_storage/zhiyihe/classfication/my_EffifientNet/dataset/11classes/32x32") #
