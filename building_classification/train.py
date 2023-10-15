
from torchvision import datasets,transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import os
import time
import argparse

device="cuda" if torch.cuda.is_available() else "cpu"

class Efficientnet_train():
    def __init__(self,opt):
        self.epochs=opt.epochs #
        self.batch_size=opt.batch_size #batch_size
        self.class_num=opt.class_num #
        self.imgsz=opt.imgsz #
        self.img_dir=opt.img_dir #
        self.weights=opt.weights #
        self.save_dir=opt.save_dir #
        self.lr=opt.lr #
        self.moment=opt.m #
        base_model = EfficientNet.from_name('efficientnet-b0') #
        '''
        efficientnet-b0: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
        efficientnet-b1: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth
        efficientnet-b2: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth
        efficientnet-b3: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth
        efficientnet-b4: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth
        efficientnet-b5: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth
        efficientnet-b6: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth
        efficientnet-b7: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth
        '''
        state_dict = torch.load(self.weights)
        base_model.load_state_dict(state_dict)
        print(base_model)
        # 
        feature_dim = base_model._fc.in_features  # 
        base_model._fc = nn.Linear(feature_dim, self.class_num) # 
        self.model = base_model.to(device)
        
        # 
        self.cross = nn.CrossEntropyLoss()
        #
        self.optimzer = optim.SGD((self.model.parameters()), 
                                   lr=self.lr, momentum=self.moment, 
                                   weight_decay=0.0004)

        #
        self.trainx, self.valx, self.b=self.process()
        print(self.b)

    def __call__(self):
        best_acc = 0
        self.model.train(True)
        for ech in range(self.epochs):
            optimzer1 = self.lrfn(ech, self.optimzer)

            print("----------Start Train Epoch %d----------" % (ech + 1))
            # 
            run_loss = 0.0  # 
            run_correct = 0.0  # 
            count = 0.0  # 

            for i, data in enumerate(self.trainx):

                inputs, label = data
                inputs, label = inputs.to(device), label.to(device)

                # 
                optimzer1.zero_grad()
                output, feature = self.model(inputs)

                loss = self.cross(output, label)
                loss.backward()
                optimzer1.step()

                run_loss += loss.item()  # 
                _, pred = torch.max(output.data, 1)
                count += label.size(0)  # 
                run_correct += pred.eq(label.data).cpu().sum()  # 
                #
                if (i+1)%100==0:
                    print('[Epoch:{}__iter:{}/{}] | Acc:{}'.format(ech + 1,i+1,len(self.trainx), run_correct/count))

            train_acc = run_correct / count
            # 
            print('Epoch:{} | Loss:{} | Acc:{}'.format(ech + 1, run_loss / len(self.trainx), train_acc))

            # 
            print("----------Waiting Test Epoch {}----------".format(ech + 1))
            with torch.no_grad():
                correct = 0.  # 
                total = 0.  # 
                for inputs, labels in self.valx:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, features = self.model(inputs)

                    # 
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += pred.eq(labels).cpu().sum()
                test_acc = correct / total
                print("epoch%d Val_data ACC" % (ech + 1), correct / total)
            if best_acc < test_acc:
                best_acc = test_acc
                start_time=(time.strftime("%m%d",time.localtime()))
                save_weight=self.save_dir+os.sep+start_time #
                os.makedirs(save_weight,exist_ok=True)
                torch.save(self.model, save_weight + os.sep + "best.pth")

    #
    def process(self):
        #
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),  # resize
                transforms.CenterCrop((self.imgsz, self.imgsz)),  # 
                transforms.RandomRotation(10),  # 【-10,10】
                transforms.RandomHorizontalFlip(p=0.2),  #
                transforms.ToTensor(),  # 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 
            ]),
            "val": transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),  # resize
                transforms.CenterCrop((self.imgsz, self.imgsz)),  # 
                transforms.ToTensor(),  # 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        # 
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.img_dir, x), data_transforms[x]) for x in
                          ['train', 'val']}
        # 
        trainx = DataLoader(image_datasets["train"], batch_size=self.batch_size, shuffle=True, drop_last=True)
        valx = DataLoader(image_datasets["val"], batch_size=self.batch_size, shuffle=True, drop_last=True)

        b = image_datasets["train"].class_to_idx  # 

        return trainx,valx,b
       

    # 
    def lrfn(self,num_epoch, optimzer):
        lr_start = 1e-5  #
        max_lr = 4e-4  # 
        lr_up_epoch = 20  # 
        lr_sustain_epoch = 20  # 
        lr_exp = 0.8  # 
        if num_epoch < lr_up_epoch:  # 
            lr = (max_lr - lr_start) / lr_up_epoch * num_epoch + lr_start
        elif num_epoch < lr_up_epoch + lr_sustain_epoch:  # 
            lr = max_lr
        else:  # 
            lr = (max_lr - lr_start) * lr_exp ** (num_epoch - lr_up_epoch - lr_sustain_epoch) + lr_start
        for param_group in optimzer.param_groups:
            param_group['lr'] = lr
        return optimzer

#
def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights",type=str,default="./pre_models/efficientnet-b0-355c32eb.pth",help='initial weights path')#
    parser.add_argument("--img-dir",type=str,default="dataset/11classes/32x32", 
                        help="train image path") #
    parser.add_argument("--imgsz",type=int,default=128,help="image size") 
    parser.add_argument("--epochs",type=int,default=500,help="train epochs")#
    parser.add_argument("--batch-size",type=int,default=24,help="train batch-size") #batch-size
    parser.add_argument("--class_num",type=int,default=11,help="class num") #
    parser.add_argument("--lr",type=float,default=0.0001,help="Init lr") #
    parser.add_argument("--m",type=float,default=0.9,help="optimer momentum") #
    parser.add_argument("--save-dir",type=str,default="./output/11classes/32x32",help="save models dir")#
    opt=parser.parse_known_args()[0]
    return opt

if __name__ == '__main__':
    opt=parse_opt()
    models=Efficientnet_train(opt)
    models()

