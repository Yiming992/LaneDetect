'''input image size 512*512'''
import torch
import os
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import cv2

class Initial(nn.Module):

	def __init__(self):
		super(Initial).__init__()
		self.net=nn.ModuleList([nn.Sequential(nn.Conv2d(3,13,3,stride=ratio,padding=1),
                                              nn.PReLu(),
                                              nn.BatchNorm2d(13)),
			                    nn.AdaptiveMaxPool2d(256)])

	def forward(self,x):
		y=x
		x=self.net[0](x)
        y=self.net[1](y)
        return torch.cat([x,y],dim=1)

class Bottleneck(nn.Module):

	def __init__(self,input_c,output_c,P,Type='downsampling',pool_size=128,ratio=2):
		super(Bottleneck).__init__()
		self.Type=Type
		if Type=='downsampling':
		    self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,output_c,ratio,stride=ratio),
		    	                                           nn.PReLu(),
		    	                                           nn.BatchNorm2d(output_c)),
		    	                    'block2':nn.Sequential(nn.Conv2d(output_c,output_c,3),
		    	                                           nn.PReLu(),
		    	                                           nn.BatchNorm2d(output_c)),
		    	                    'block3':nn.Sequential(nn.Conv2d(output_c,output_c,1),
		    	                                           nn.PReLu(),
		    	                                           nn.BatchNorm2d(output_c)),
		    	                    'dropout':nn.Dropout2d(p=P),
		    	                    'Pooling':nn.AdaptiveMaxPool2d(pool_size)})

		elif Type.split()[0]=='asymmetric':
			self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c/ratio,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_c/ratio)),
				                    'block2':nn.Sequential(nn.Sequential(nn.Conv2d(input_c/ratio,input_c/ratio,(5,1)),
				                    	                   nn.Conv2d(input_c/ratio,input_c/ratio,(1,5))),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_c/ratio)),
				                    'block3':nn.Sequential(nn.Cov2d(input_c/ratio,output_c,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(output_c)),
				                    'dropout':nn.Dropout2d(p=P)})

		elif Type.split()[0]=='dilated':
			self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c/ratio,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_c/ratio)),
				                    'block2':nn.Sequential(nn.Conv2d(input_c/ratio,input_c/ratio,3,dilation=int(Type.split()[1])),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_c/ratio)),
				                    'block3':nn.Sequential(nn.Conv2d(input_c/ratio,output_c,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(output_c)),
				                    'dropout':nn.Dropout2d(p=P)})
		elif Type=='normal':
			self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c/ratio,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_c/ratio)),
				                    'block2':nn.Sequential(nn.Conv2d(input_c/ratio,input_c/ratio,3),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_c/ratio)),
				                    'block3':nn.Sequential(nn.Conv2d(input_c/ratio,output_c,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(output_c)),
				                    'dropout':nn.Dropout2d(p=P)})
		elif Type='upsampling':
			self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_c,input_c/ratio,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_c/ratio)),
			                        'block2':nn.Sequential(nn.ConvTranspose2d(input_c,input_c/ratio,3),
			                       	                       nn.PReLu(),
			                       	                       nn.BatchNorm2d()),
			                        'block3':nn.Sequential(nn.Conv2d(input_c/ratio,output_c,1),
			                       	                       nn.PReLu(),
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
		super(RepeatBlock).__init__()
		self.add_module(Bottleneck(Type='Normal',input_c,output_c,0.1))
		self.add_module(Bottleneck(Type='dilated ratio',output_c,output_c,0.1))
		self.add_module(Bottleneck(Type='asymmetric',output_c,output_c,0.1))
		self.add_module(Bottleneck(Type='dilated 4',output_c,output_c,0.1))
		self.add_module(Bottleneck(Type='normal',output_c,output_c,0.1))
		self.add_module(Bottleneck(Type='dilated 8',output_c,output_c,0.1))
		self.add_module(Bottleneck(Type='asymmetric',output_c,output_c,0.1))
		self.add_module(Bottleneck(Type='dilated 16',output_c,output_c,0.1))

class Decoder(nn.Sequential):

	def __init__(self,input_c,mid_c,output_c):
		super(Decoder).__init__()
        self.add_module(Bottleneck(Type='upsampling',input_c,mid_c,0.1))
        self.add_module(Bottleneck(Type='normal',mid_c,mid_c,0.1))
        self.add_module(Bottleneck(Type='normal',mid_c,mid_c,0.1))
        self.add_module(Bottleneck(Type='upsampling',mid_c,output_c,0.1))
        self.add_module(Bottleneck(Type='normal',output_c,output_c,0.1))


'''Shared Encoder'''
class SharedEncoder(nn.Module):

	def __init__(self):
		super(SharedEncoder).__init__()
		self.initial=Initial()
		self.net=nn.Sequential(Bottleneck(16,64,0.01),
			                   Bottleneck(Type='normal',64,64,0.01),
			                   Bottleneck(Type='normal',64,64,0.01),
			                   Bottleneck(Type='normal',64,64,0.01),
			                   Bottleneck(Type='normal',64,64,0.01),
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
		super(Embedding).__init__()
        self.net=nn.Sequential(RepeatBlock(128,128),
                               Decoder(128,64,embed_size))
		

	def forward(self,x):
		return self.net(x)

'''Segmentation'''
class Segmentation(nn.Module):
	
	def __init__(self):
		super(Segmentation).__init__()
		self.net=nn.Sequential(RepeatBlock(128,128),
                               Decoder(128,2))

	def forward(self,x):
		return self.net(x)


'''LaneNet'''
class LaneNet(nn.Module):

	def __init__(self):
		super(LaneNet).__init__()
		self.net=nn.ModuleDict({'Shared_Encoder':SharedEnocer(),
			                    'Embedding':Embedding(),
			                    'Segmentation':Segmentation()})
		
	def forward(self,x):
		x=self.net['Shared_Encoder'](x)
		x=self.net['Embedding'](x)
		x=self.net['Segmentation'](x)
		return x


		