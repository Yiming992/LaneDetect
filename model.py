'''input image size 512*256'''
import torch
import os
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import cv2
'''
---------------------------------------------------------------------------------------------
LaneNet based on paper 'Towards End-to-End Lane Detection: an Instance Segmentation Approach'
---------------------------------------------------------------------------------------------
'''
class Initial(nn.Module):
    def __init__(self):
        super(Initial,self).__init__()
        self.net=nn.ModuleList([nn.Sequential(nn.Conv2d(3,13,2,stride=2),
                                              nn.PReLU(),
                                              nn.BatchNorm2d(13)),
                                nn.AdaptiveMaxPool2d((256,128))])

    def forward(self,x):
        y=x
        x=self.net[0](x)
        y=self.net[1](y)
        return torch.cat([x,y],dim=1)

class Bottleneck(nn.Module):
    def __init__(self,input_c,output_c,P,
                 Type='downsampling',pool_size=(128,64),
                 pad=0,ratio=2):
        super(Bottleneck,self).__init__()
        self.Type=Type
        if self.Type=='downsampling':
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
                                    'Pooling':nn.AdaptiveMaxPool2d(pool_size,return_indices=True)})

        elif self.Type.split()[0]=='asymmetric':
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

        elif self.Type.split()[0]=='dilated':
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
        elif self.Type=='normal':
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
        elif self.Type=='upsampling':
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
                                    'unpooling':nn.MaxUnpool2d(2,stride=2),
                                    'dropout':nn.Dropout2d(p=P)})

    def forward(self,x,pool_index=None):
        y=x
        if self.Type=='downsampling':
            x=self.net['block1'](x)
            x=self.net['block2'](x)
            x=self.net['block3'](x)
            x=self.net['dropout'](x)
            y,index=self.net['Pooling'](y)
            zero_pads=torch.zeros(x.size(0),x.size(1)-y.size(1),y.size(2),y.size(3)).cuda()
            y=torch.cat([y,zero_pads],dim=1)

            return x+y,index
        else:
            x=self.net['block1'](x)
            x=self.net['block2'](x)
            x=self.net['block3'](x)
            x=self.net['dropout'](x)
            if self.Type=='upsampling':
                y=self.net['block4'](y)
                y=self.net['unpooling'](y,pool_index)

            return x+y


'''Shared Encoder'''
class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder,self).__init__()
        self.initial=Initial()
        self.downsample=nn.ModuleDict({'downsample_1':Bottleneck(16,64,0.01,pad=1),
                                       'downsample_2':Bottleneck(64,128,0.1,pool_size=(64,32),pad=1)})
        self.net=nn.Sequential(Bottleneck(64,64,0.01,Type='normal',pad=1),
                               Bottleneck(64,64,0.01,Type='normal',pad=1),
                               Bottleneck(64,64,0.01,Type='normal',pad=1),
                               Bottleneck(64,64,0.01,Type='normal',pad=1)
                               )
        self.tail=RepeatBlock(128,128)

    def forward(self,x): 
        pool_indices={}
        x=self.initial(x)
        x,index=self.downsample['downsample_1'](x)
        pool_indices['downsample_1']=index
        x=self.net(x)
        x,index=self.downsample['downsample_2'](x)
        pool_indices['downsample_2']=index
        x=self.tail(x)
        return x,pool_indices


'''Repeat Block'''
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


class Decoder(nn.Module):
    def __init__(self,input_c,mid_c,output_c,final):
        super(Decoder,self).__init__()
        self.upsample=nn.ModuleDict({'upsample_1':Bottleneck(input_c,mid_c,0.1,Type='upsampling'),
                                     'upsample_2':Bottleneck(mid_c,output_c,0.1,Type='upsampling')})
        self.net=nn.ModuleDict({'normal_1':Bottleneck(mid_c,mid_c,0.1,Type='normal',pad=1),
                                'normal_2':Bottleneck(mid_c,mid_c,0.1,Type='normal',pad=1)})
        self.end=nn.Sequential(nn.ConvTranspose2d(output_c,final,2,stride=2))                             
        #self.add_module('Bottleneck_5',Bottleneck(output_c,output_c,0.1,Type='normal'))
    
    def forward(self,x,pool_indices):
        x=self.upsample['upsample_1'](x,pool_index=pool_indices['downsample_2'])
        x=self.net['normal_1'](x)
        x=self.net['normal_2'](x)
        x=self.upsample['upsample_2'](x,pool_index=pool_indices['downsample_1'])
        x=self.end(x)
        return x

        
'''Embedding'''
class Embedding(nn.Module):  
    def __init__(self,embed_size):
        super(Embedding,self).__init__()
        self.net=nn.ModuleDict({'repeat':RepeatBlock(128,128),
                                'decoder':Decoder(128,64,16,embed_size)})
        
    def forward(self,x,pool_indices=None):
        x=self.net['repeat'](x)
        x=self.net['decoder'](x,pool_indices)
        return x

'''Segmentation'''
class Segmentation(nn.Module):   
    def __init__(self):
        super(Segmentation,self).__init__()
        self.net=nn.ModuleDict({'repeat':RepeatBlock(128,128),
                                'decoder':Decoder(128,64,16,2)})

    def forward(self,x,pool_indices=None):
        x=self.net['repeat'](x)
        x=self.net['decoder'](x,pool_indices)
        return x

'''LaneNet'''
class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet,self).__init__()
        self.net=nn.ModuleDict({'Shared_Encoder':SharedEncoder(),
                                'Embedding':Embedding(5),
                                'Segmentation':Segmentation()})
        
    def forward(self,x):
        x,pool_indices=self.net['Shared_Encoder'](x)
        embeddings=self.net['Embedding'](x,pool_indices)
        segmentation=self.net['Segmentation'](x,pool_indices)
        return segmentation,embeddings

if __name__=='__main__':
    model=LaneNet()
    
        
    


        