import torch
import torch.nn as nn

import numpy as np

# PatchGANs discriminator -> 70x70 patch input ( Do on Training stage ) -> Conv연산을 통해 이미지를 30x30으로 줄이면 각 pixel은 본래 이미지의 70x70 크기의 patch가 축소된 것이라고 볼 수 있다. -> 70x70 patch discriminator 완성!
class Discriminator(nn.Module):
    def __init__(self, in_channel, ndf = 64):
        super(Discriminator,self).__init__()
        
        self.sequence = []
        # layer 1
        self.sequence += [nn.Conv2d(in_channels=in_channel, out_channels=ndf, 
                                                kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU(0.2,True)]

        for i in range(3):
            self.sequence += [nn.Conv2d(in_channels=ndf*(2**i), out_channels=ndf*(2**(i+1)), 
                                        kernel_size=4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(ndf*(2**(i+1))),
                            nn.LeakyReLU(0.2,True)]

        # layer 5
        self.sequence += [nn.Conv2d(in_channels=ndf*8, out_channels=ndf*8, 
                                kernel_size=4, stride=1, padding=2, bias=False),
                            nn.BatchNorm2d(ndf*8),
                            nn.LeakyReLU(0.2,True)]

        # out
        self.sequence += [nn.Conv2d(in_channels=ndf*8, out_channels=1, 
                                kernel_size=4, stride=1, padding=1, bias=False)]

        self.layers = nn.Sequential(*self.sequence)

    def forward(self, x):
        y = self.layers(x)

        return y

# ResNet Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=0, bias=True)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.seq1 = nn.Sequential(self.pad, self.conv1, self.bn1, self.relu)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=0, bias=True)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.seq2 = nn.Sequential(self.pad, self.conv2, self.bn2)
        
        self.down_flag = False
        if in_channels != out_channels: self.down_flag = True

        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=2, padding=0, bias=True)
    
    def forward(self, x):
        y = self.seq1(x)
        y = self.seq2(y)

        if self.down_flag:
            x = self.downsample(x)
        
        y = self.relu(y)

        y = y + x # shortcut

        return y
        

# Encoder: ResNet with 9 Blocks, Decoder : Upsampling with ResNet Block
class ResNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, n_block, ngf=64):
        '''
        n_block = 6 (<= 128x128) | 9 (>= 256x256) : # of Residual Blocks
        '''
        super(ResNetGenerator,self).__init__()

        self.sequence = []
        
        # Initial Conv
        self.relu = nn.ReLU()
        self.pad = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=ngf, kernel_size=(7,7), stride=1, padding=0)
        self.norm1 = nn.InstanceNorm2d(ngf)
        self.sequence += [self.pad, self.conv1, self.norm1, self.relu]

        # Down Sampling
        n_downsample = 2
        for i in range(n_downsample):
            self.sequence += [nn.Conv2d(in_channels=ngf*(2**i), out_channels=ngf*(2**(i+1)), kernel_size=(3,3), stride=2, padding=1),
                            nn.InstanceNorm2d(ngf),
                            nn.ReLU(inplace=True)]

        # Residual Blocks
        for i in range(n_block):
            self.sequence += [BasicBlock(ngf*(2**n_downsample),ngf*(2**n_downsample))]
        
        # Up Sampling
        for i in range(n_downsample): # Do up_sample as like n_downsample
            self.sequence += [nn.ConvTranspose2d(ngf*(2**(n_downsample-i)),ngf*(2**(n_downsample-(i+1))), 
                                                kernel_size=3, stride=2, padding=1, output_padding=1),
                              nn.InstanceNorm2d(ngf*(2**(n_downsample-(i+1)))),
                              nn.ReLU()]
                            #   nn.Conv2d(ngf*(2**(n_downsample-(i+1))), ngf*(2**(n_downsample-(i+1))),
                            #             kernel_size=3, stride=1, padding=1),
                            #   nn.InstanceNorm2d(ngf*(2**(n_downsample-(i+1)))),
                            #   nn.ReLU()]
        self.sequence += [nn.ReflectionPad2d(3),
                          nn.Conv2d(ngf, out_channels, 7, padding=0), 
                          nn.Tanh()]
        
        self.layers = nn.Sequential(*self.sequence)

    def forward(self, x):
        return self.layers(x)


# encoding block
def CBR2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False):
    layers = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                            nn.BatchNorm2d(out_channel))

    return layers


# decoding block
def CBR(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False):
    layers = nn.Sequential(nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(in_channel*2, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                            nn.BatchNorm2d(out_channel))

    return layers


class UNetGenerator(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNetGenerator,self).__init__()
        self.n_classes = in_channel
        
        # encoding
        self.enc1 = nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1, bias=False) # 256
        self.enc2 = CBR2d(64, 128) # 128
        self.enc3 = CBR2d(128, 256) # 64
        self.enc4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False))

        # decoding
        self.dec4 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(256)) # 64
        self.dec3 = CBR(256,128) # 128
        self.dec2 = CBR(128,64) # 256
        self.dec1 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(64*2,out_channel,kernel_size=4,stride=2,padding=1)) # 512

    def forward(self,x):
        # constracting
        x1 = self.enc1(x) # 1->64, 256
        x2 = self.enc2(x1) # 64->128, 128
        x3 = self.enc3(x2) # 128->256, 64
        x4 = self.enc4(x3) # 256->512, 32

        # expanding
        y3 = self.dec4(x4) 
        cat = torch.cat((x3,y3),dim=1)
        y2 = self.dec3(cat)
        cat = torch.cat((x2,y2),dim=1)
        y1 = self.dec2(cat)
        cat = torch.cat((x1,y1),dim=1)
        y = self.dec1(cat)

        return y

