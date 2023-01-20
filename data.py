from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np
import torch

import os
from PIL import Image

class horse2zebraDataset(Dataset):
    def __init__(self, path="train", imgSize = 64):
        super(horse2zebraDataset, self).__init__()
        self.path = path
        self.imgSize = imgSize

        self.imgs_A = os.listdir(f'./datasets/horse2zebra/{path}A')
        self.imgs_B = os.listdir(f'./datasets/horse2zebra/{path}B')

    def __len__(self):
        return min(len(self.imgs_A), len(self.imgs_B))

    def __getitem__(self, idx):
        self.imgA = Image.open(f'./datasets/horse2zebra/{self.path}A/'+self.imgs_A[idx]).convert("RGB")
        self.imgB = Image.open(f'./datasets/horse2zebra/{self.path}B/'+self.imgs_B[idx]).convert("RGB")

        if self.path == "train":
            self.imgA = self.train_transform(self.imgA)
            self.imgB = self.train_transform(self.imgB)
        else :
            self.imgA = self.test_transform(self.imgA)
            self.imgB = self.test_transform(self.imgB)

        return (self.imgA, self.imgB)

    def train_transform(self, x):
        transform = transforms.Compose([transforms.Resize(self.imgSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        return transform(x)

    def test_transform(self, x):
        transform = transforms.Compose([transforms.Resize(self.imgSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        return transform(x)


class AAPMDataset(Dataset):
    def __init__(self, path="train", imgSize = 512):
        super(AAPMDataset, self).__init__()
        self.path = path
        self.imgSize = imgSize

        if self.path == 'train':
            self.plus = 500
        else:
            self.plus = 50

        if path == "train":
            self.length = 3839 # train data set cnt
        else:
            self.length = 421 # test data set cnt

        self.imgs_A = os.listdir(f'./AAPM_data/{path}/full_dose')
        self.imgs_B = os.listdir(f'./AAPM_data/{path}/quarter_dose')

    def __len__(self):
        return min(len(self.imgs_A), len(self.imgs_B))

    def __getitem__(self, idx): # length = self.__len__() : 매번 길이 계산하는 함수 불러오면 많이 느려질 것 같아서
        self.imgA = torch.tensor(np.load(f'./AAPM_data/{self.path}/full_dose/{idx+1}.npy')).view(1,512,512)
        self.imgB = torch.tensor(np.load(f'./AAPM_data/{self.path}/quarter_dose/{(idx+self.plus)%self.length+1}.npy')).view(1,512,512) # full & quarter not aligned (= not paired) <- Unsupervised

        return (self.imgA, self.imgB)

    def train_transform(self, x): # post-process ( 원상 복구 = 시각화 용 )
        transform = transforms.Compose([transforms.Resize(self.imgSize),
                                        transforms.Normalize((-0.01,),(0.1,))]) # 데이터가 이미 정규화 됨. mean = 0.009 = std(?) 라서, mean은 -0.01, std = 0.1로 다시 정규화하면 원상 복구.
                                                                                # 보통 min = -0.3 max = 0.5 이정도? -> 분포를 봤을 때 generator out layer로 sigmoid() or tanh() 쓰기 적합하지 않아보임
                                                                                # out layer를 conv로 그냥 끝내버리자

        return transform(x)

    def test_transform(self, x): # post-process
        transform = transforms.Compose([transforms.Resize(self.imgSize),
                                        transforms.Normalize((-0.01,),(0.1,))])
        
        return transform(x)