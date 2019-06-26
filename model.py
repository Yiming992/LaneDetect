import torch
import os
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import cv2
'''
---------------------------------
Build deep neural network as described in 

"ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation",ArXiv2017,
Adam PaszkeAbhishek ChaurasiaSangpil KimEugenio Culurciello

 "Towards End-to-End Lane Detection: an Instance Segmentation Approach",ArXiv2018,
 Davy Neven, Bert De Brabandere, Stamatios Georgoulis, Marc Proesmans, Luc Van Gool
---------------------------------
'''
class Initial(nn.Module):
    def __init__(self):
        super(Initial,self).__init__()
        self.net=nn.ModuleList([nn.Sequential(nn.Conv2d(3,13,2,stride=2),
                                              nn.PReLU(),
                                              nn.BatchNorm2d(13)),
                                nn.MaxPool2d(2)])

    def forward(self,x):
        y=x
        x=self.net[0](x)
        y=self.net[1](y)
        return torch.cat([x,y],dim=1)

class Bottleneck(nn.Module):
    def __init__(self,input_c,output_c,P,
                 Type='downsampling',pad=0,ratio=2):
        super(Bottleneck,self).__init__()
        self.Type=Type
        if self.Type=='downsampling':#downsampling block
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,output_c,2,stride=2),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'block2':nn.Sequential(nn.Conv2d(output_c,output_c,3,padding=pad),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'block3':nn.Sequential(nn.Conv2d(output_c,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'dropout':nn.Dropout2d(p=P),
                                    'Pooling':nn.MaxPool2d(2),
                                    '1*1':nn.Conv2d(input_c,output_c,1)})

        elif self.Type.split()[0]=='asymmetric':#asymmetric block
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c//ratio,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block2':nn.Sequential(nn.Sequential(nn.Conv2d(input_c//ratio,input_c//ratio,(5,1),padding=(2,0)),
                                                           nn.Conv2d(input_c//ratio,input_c//ratio,(1,5),padding=(0,2))),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block3':nn.Sequential(nn.Conv2d(input_c//ratio,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'dropout':nn.Dropout2d(p=P)})

        elif self.Type.split()[0]=='dilated':#dilated block
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c//ratio,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block2':nn.Sequential(nn.Conv2d(input_c//ratio,input_c//ratio,3,dilation=int(Type.split()[1]),padding=pad),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block3':nn.Sequential(nn.Conv2d(input_c//ratio,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'dropout':nn.Dropout2d(p=P)})
        elif self.Type=='normal':#normal block
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c//ratio,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block2':nn.Sequential(nn.Conv2d(input_c//ratio,input_c//ratio,3,padding=pad),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block3':nn.Sequential(nn.Conv2d(input_c//ratio,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'dropout':nn.Dropout2d(p=P)})
        elif self.Type=='upsampling':#upsampling block
            self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c//ratio,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block2':nn.Sequential(nn.ConvTranspose2d(input_c//ratio,input_c//ratio,2,stride=2,padding=pad),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(input_c//ratio)),
                                    'block3':nn.Sequential(nn.Conv2d(input_c//ratio,output_c,1),
                                                           nn.PReLU(),
                                                           nn.BatchNorm2d(output_c)),
                                    'block4':nn.Conv2d(input_c,output_c,1),
                                    'dropout':nn.Dropout2d(p=P)})

    def forward(self,x,prev=None):
        y=x
        if self.Type=='downsampling':
            x=self.net['block1'](x)
            x=self.net['block2'](x)
            x=self.net['block3'](x)
            x=self.net['dropout'](x)
            y=self.net['Pooling'](y)
            y=self.net['1*1'](y)

            return x+y
        else:
            x=self.net['block1'](x)
            x=self.net['block2'](x)
            x=self.net['block3'](x)
            x=self.net['dropout'](x)
            if self.Type=='upsampling':
                return x+prev
            return x+y

class RepeatBlock(nn.Sequential):
    def __init__(self,input_c,output_c):
        super(RepeatBlock,self).__init__()
        self.add_module('Bottleneck_1',Bottleneck(input_c,output_c,0.1,Type='normal',pad=1))
        self.add_module('Bottleneck_2',Bottleneck(output_c,output_c,0.1,Type='dilated 2',pad=2))
        self.add_module('Bottleneck_3',Bottleneck(output_c,output_c,0.1,Type='asymmetric'))
        self.add_module('Bottleneck_4',Bottleneck(output_c,output_c,0.1,Type='dilated 4',pad=4))
        self.add_module('Bottleneck_5',Bottleneck(output_c,output_c,0.1,Type='normal',pad=1))
        self.add_module('Bottleneck_6',Bottleneck(output_c,output_c,0.1,Type='dilated 8',pad=8))
        self.add_module('Bottleneck_7',Bottleneck(output_c,output_c,0.1,Type='asymmetric'))
        self.add_module('Bottleneck_8',Bottleneck(output_c,output_c,0.1,Type='dilated 16',pad=16))

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder,self).__init__()
        self.initial=Initial()
        self.downsample=nn.ModuleDict({'downsample_1':Bottleneck(16,64,0.01,pad=1),
                                       'downsample_2':Bottleneck(64,128,0.1,pad=1)})
        self.net=nn.Sequential(Bottleneck(64,64,0.01,Type='normal',pad=1),
                               Bottleneck(64,64,0.01,Type='normal',pad=1),
                               Bottleneck(64,64,0.01,Type='normal',pad=1),
                               Bottleneck(64,64,0.01,Type='normal',pad=1)
                               )
        self.tail=RepeatBlock(128,128)

    def forward(self,x):
        prevs=[]
        x=self.initial(x)
        prevs.append(x)
        x=self.downsample['downsample_1'](x)
        prevs.append(x)
        x=self.net(x)
        x=self.downsample['downsample_2'](x)
        x=self.tail(x)
        return x,prevs

class Decoder(nn.Module):
    def __init__(self,input_c,mid_c,output_c,final):
        super(Decoder,self).__init__()
        self.upsample=nn.ModuleDict({'upsample_1':Bottleneck(input_c,mid_c,0.1,Type='upsampling'),
                                     'upsample_2':Bottleneck(mid_c,output_c,0.1,Type='upsampling')})
        self.net=nn.ModuleDict({'normal_1':Bottleneck(mid_c,mid_c,0.1,Type='normal',pad=1),
                                'normal_2':Bottleneck(mid_c,mid_c,0.1,Type='normal',pad=1)})
        self.end=nn.Sequential(nn.ConvTranspose2d(output_c,final,2,stride=2))                             
    
    def forward(self,x,y):
        x=self.upsample['upsample_1'](x,y[1])
        x=self.net['normal_1'](x)
        x=self.net['normal_2'](x)
        x=self.upsample['upsample_2'](x,y[0])
        x=self.end(x)
        return x

        

class Embedding(nn.Module):  
    def __init__(self,embed_size):
        super(Embedding,self).__init__()
        self.net=nn.ModuleDict({'repeat':RepeatBlock(128,128),
                                'decoder':Decoder(128,64,16,embed_size)})
        
    def forward(self,x,prevs):
        '''
        定义Embedding模块前向传播过程
        '''
        x=self.net['repeat'](x)
        x=self.net['decoder'](x,prevs)
        return x

class Segmentation(nn.Module):   
    def __init__(self):
        super(Segmentation,self).__init__()
        self.net=nn.ModuleDict({'repeat':RepeatBlock(128,128),
                                'decoder':Decoder(128,64,16,2)})

    def forward(self,x,prevs):
        x=self.net['repeat'](x)
        x=self.net['decoder'](x,prevs)
        return x

'''LaneNet'''
class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet,self).__init__()
        self.net=nn.ModuleDict({'Shared_Encoder':SharedEncoder(),
                                'Embedding':Embedding(4),
                                'Segmentation':Segmentation()})
        
    def forward(self,x):
        x,prevs=self.net['Shared_Encoder'](x)
        embeddings=self.net['Embedding'](x,prevs)
        segmentation=self.net['Segmentation'](x,prevs)
        return segmentation,embeddings

    
        
    


        