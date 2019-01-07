'''input image size 512*256'''
import torch
import os
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import cv2

class Initial(nn.Module):

    def __init__(self):
        super(Initial,self).__init__()
        self.net=nn.ModuleList([nn.Sequential(nn.Conv2d(3,13,3,stride=2,padding=1),
                                              nn.PReLU(),
                                              nn.BatchNorm2d(13)),
                                nn.AdaptiveMaxPool2d(256)])

    def forward(self,x):
        y=x
        x=self.net[0](x)
        y=self.net[1](y)
        return torch.cat([x,y],dim=1)

class Bottleneck(nn.Module):

    def __init__(self,input_c,output_c,P,Type='downsampling',pool_size=128,ratio=2):
        super(Bottleneck,self).__init__()
        self.Type=Type
        if self.Type=='downsampling':
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,output_c,ratio,stride=ratio),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'block2':nn.Sequential(nn.Conv2d(output_c,output_c,3),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'block3':nn.Sequential(nn.Conv2d(output_c,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'dropout':nn.Dropout2d(p=P),
                                    'Pooling':nn.AdaptiveMaxPool2d(pool_size)})

        elif self.Type.split()[0]=='asymmetric':
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c//ratio,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block2':nn.Sequential(nn.Sequential(nn.Conv2d(input_c//ratio,input_c//ratio,(5,1)),
                                                           nn.Conv2d(input_c//ratio,input_c//ratio,(1,5))),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block3':nn.Sequential(nn.Conv2d(input_c//ratio,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'dropout':nn.Dropout2d(p=P)})

        elif self.Type.split()[0]=='dilated':
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c//ratio,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block2':nn.Sequential(nn.Conv2d(input_c//ratio,input_c//ratio,3,dilation=int(Type.split()[1])),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block3':nn.Sequential(nn.Conv2d(input_c//ratio,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'dropout':nn.Dropout2d(p=P)})
        elif self.Type=='normal':
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c//ratio,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block2':nn.Sequential(nn.Conv2d(input_c//ratio,input_c//ratio,3),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block3':nn.Sequential(nn.Conv2d(input_c//ratio,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'dropout':nn.Dropout2d(p=P)})
        elif self.Type=='upsampling':
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c//ratio,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block2':nn.Sequential(nn.ConvTranspose2d(input_c,input_c//ratio,3),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block3':nn.Sequential(nn.Conv2d(input_c//ratio,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'dropout':nn.Dropout2d(p=P)})

    def forward(self,x):
        y=x
        if self.Type=='donwsampling':
            x=self.net['block1'](x)
            x=self.net['block2'](x)
            x=self.net['block3'](x)
            x=self.net['dropout'](x)

            y=self.net['Pooling'](y)
            zero_pads=torch.zeros(x.size[0],x.size[1]-y.size[1],x.size[ratio],x.size[3])
            y=torch.cat([y,zero_pads],dim=1)
            return x+y 
        else:
            x=self.net['block1'](x)
            x=self.net['block2'](x)
            x=self.net['block3'](x)
            x=self.net['dropout'](x)

            return x+y

'''Repeat Block'''
class RepeatBlock(nn.Sequential):

    def __init__(self,input_c,output_c):
        super(RepeatBlock,self).__init__()
        self.add_module('Bottleneck_1',Bottleneck(input_c,output_c,0.1,Type='Normal'))
        self.add_module('Bottleneck_2',Bottleneck(output_c,output_c,0.1,Type='dilated 2'))
        self.add_module('Bottleneck_3',Bottleneck(output_c,output_c,0.1,Type='asymmetric'))
        self.add_module('Bottleneck_4',Bottleneck(output_c,output_c,0.1,Type='dilated 4'))
        self.add_module('Bottleneck_5',Bottleneck(output_c,output_c,0.1,Type='normal'))
        self.add_module('Bottleneck_6',Bottleneck(output_c,output_c,0.1,Type='dilated 8'))
        self.add_module('Bottleneck_7',Bottleneck(output_c,output_c,0.1,Type='asymmetric'))
        self.add_module('Bottleneck_8',Bottleneck(output_c,output_c,0.1,Type='dilated 16'))

class Decoder(nn.Sequential):

    def __init__(self,input_c,mid_c,output_c):
        super(Decoder).__init__()
        self.add_module('Bottleneck_1',Bottleneck(input_c,mid_c,0.1,Type='upsampling'))
        self.add_module('Bottleneck_2',Bottleneck(mid_c,mid_c,0.1,Type='normal'))
        self.add_module('Bottleneck_3',Bottleneck(mid_c,mid_c,0.1,Type='normal'))
        self.add_module('Bottleneck_4',Bottleneck(mid_c,output_c,0.1,Type='upsampling'))
        self.add_module('Bottleneck_5',Bottleneck(output_c,output_c,0.1,Type='normal'))


'''Shared Encoder'''
class SharedEncoder(nn.Module):

    def __init__(self):
        super(SharedEncoder,self).__init__()
        self.initial=Initial()
        self.net=nn.Sequential(Bottleneck(16,64,0.01),
                               Bottleneck(64,64,0.01,Type='normal'),
                               Bottleneck(64,64,0.01,Type='normal'),
                               Bottleneck(64,64,0.01,Type='normal'),
                               Bottleneck(64,64,0.01,Type='normal'),
                               Bottleneck(64,128,0.1),
                               )
        self.tail=RepeatBlock(128,128)

    
    def forward(self,x):
        x=self.initial(x)
        x=self.net(x)
        x=self.tail(x)
        return x
        
'''Embedding'''
class Embedding(nn.Module):
    
    def __init__(self,embed_size):
        super(Embedding,self).__init__()
        self.net=nn.Sequential(RepeatBlock(128,128),
                               Decoder(128,64,embed_size))
        

    def forward(self,x):
        return self.net(x)

'''Segmentation'''
class Segmentation(nn.Module):
    
    def __init__(self):
        super(Segmentation,self).__init__()
        self.net=nn.Sequential(RepeatBlock(128,128),
                               Decoder(128,1))

    def forward(self,x):
        return self.net(x)


'''LaneNet'''
class LaneNet(nn.Module):

    def __init__(self):
        super(LaneNet,self).__init__()
        self.net=nn.ModuleDict({'Shared_Encoder':SharedEncoder(),
                                'Embedding':Embedding(4),
                                'Segmentation':Segmentation()})
        
    def forward(self,x):
        x=self.net['Shared_Encoder'](x)
        x1=self.net['Embedding'](x)
        x2=self.net['Segmentation'](x)
        return x1,x2


if __name__=='__main__':

    model=LaneNet()



        