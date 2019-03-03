import torch 
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):   
    def __init__(self,input_c,output_c,size,stride=3,padding=0):
        super(ConvBlock,self).__init__()
        self.net=nn.Sequential(nn.Conv2d(input_c,output_c,size,stride=stride,padding=padding),
                               nn.BatchNorm2d(output_c),
                               nn.ReLU())
    def forward(self,x):
        return self.net(x)

class LinearBlock(nn.Module):
    def __init__(self,input_c,output_c):
        super(LinearBlock,self).__init__()
        self.net=nn.Sequential(nn.Linear(input_c,output_c),
                               nn.BatchNorm2d(output_c),
                               nn.ReLU()) 
    def forward(self,x):
        return self.net(x)


class HNet(nn.Module):
    def __init__(self):
        super(HNet,self).__init__()
        self.conv=nn.Sequential(ConvBlock(3,16,3),
                                ConvBlock(16,16,3),
                                nn.AdaptiveMaxpool2d([64,32]),
                                ConvBlock(16,32,3),
                                ConvBlock(32,32,3),
                                nn.AdaptiveMaxpool2d([32,16]),
                                ConvBlock(32,64),
                                ConvBlock(64,64),
                                nn.AdaptiveMaxpool2d([16,8]))

        self.linear=nn.Sequential(LinearBlock(16*8*64,1024),
                                  nn.Linear(1024,6))

    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size[0],-1)
        x=self.linear(x)
        return x