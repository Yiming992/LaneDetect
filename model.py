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
		self.net=nn.ModuleList([nn.Conv2d(3,13,3,stride=2,padding=1),
			                    nn.AdaptiveMaxPool2d(256)])

	def forward(self,x):
		y=x
		x=self.net[0](x)
        y=self.net[1](y)
        return torch.cat([x,y],dim=1)

class Bottleneck(nn.Module):

	def __init__(self,Type='downsampling',input_num,output_num,P):
		super(Bottleneck).__init__()
		self.Type=Type
		if Type=='downsampling':
		    self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_num,output_num,2,stride=2),
		    	                                           nn.PReLu(),
		    	                                           nn.BatchNorm2d(ouput_num)),
		    	                    'block2':nn.Sequential(nn.Conv2d(input_num,output_num,3),
		    	                                           nn.PReLu(),
		    	                                           nn.BatchNorm2d(output_num)),
		    	                    'block3':nn.Sequential(nn.Conv2d(input_num,output_num,1),
		    	                                           nn.PReLu(),
		    	                                           nn.BatchNorm2d(output_num)),
		    	                    'dropout':nn.Dropout2d(p=P),
		    	                    'Pooling':nn.AdaptiveMaxPool2d()})

		elif Type.split()[0]=='asymmetric':
			self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_num,input_num/2,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_num/2)),
				                    'block2':nn.Sequential(nn.Sequential(nn.Conv2d(input_num/2,input_num/2,(5,1)),
				                    	                   nn.Conv2d(input_num/2,input_num/2,(1,5))),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_num/2)),
				                    'block3':nn.Sequential(nn.Cov2d(input_num/2,output_num,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(output_num)),
				                    'dropout':nn.Dropout2d(p=P)})

		elif Type.split()[0]=='dilated':
			self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_num,input_num/2,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_num/2)),
				                    'block2':nn.Sequential(nn.Conv2d(input_num/2,input_num/2,3,dilation=int(Type.split()[1])),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_num/2)),
				                    'block3':nn.Sequential(nn.Conv2d(input_num/2,output_num,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(output_num)),
				                    'dropout':nn.Dropout2d(p=P)})
		elif Type=='normal':
			self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_num,input_num/2,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_num/2)),
				                    'block2':nn.Sequential(nn.Conv2d(input_num/2,input_num/2,3),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_num/2)),
				                    'block3':nn.Sequential(nn.Conv2d(input_num/2,output_num,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(output_num)),
				                    'dropout':nn.Dropout2d(p=P)})
		elif Type='upsampling':
			self.net=nn.ModuleDict({'block1':nn.Sequential(nn.Conv2d(input_num,input_num/2,1),
				                                           nn.PReLu(),
				                                           nn.BatchNorm2d(input_num/2)),
			                        'block2':nn.Sequential(nn.ConvTranspose2d(input_num,input_num/2,3),
			                       	                       nn.PReLu(),
			                       	                       nn.BatchNorm2d()),
			                        'block3':nn.Sequential(nn.Conv2d(input_dims_num/2,output_num,1),
			                       	                       nn.PReLu(),
			                       	                       nn.BatchNorm2d(output_dim)),
			                        'dropout':nn.Dropout2d(p=P)})

	def forward(self,x):
		y=x
		if self.Type=='donwsampling':
			x=self.net['block1'](x)
			x=self.net['block2'](x)
			x=self.net['block3'](x)
			x=self.net['dropout'](x)


			y=self.net['Pooling'](y)
			zero_pads=torch.zeros(x.size[0],x.size[1]-y.size[1],x.size[2],x.size[3])
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

	def __init__(self,input_dims,output_dims):
		super(RepeatBlock).__init__()
		self.add_module(Bottleneck(Type='Normal',input_dims,output_dims,0.1))
		self.add_module(Bottleneck(Type='dilated 2',input_dims,output_dims,0.1))
		self.add_module(Bottleneck(Type='asymmetric',input_dims,output_dims,0.1))
		self.add_module(Bottleneck(Type='dilated 4',input_dims,output_dims,0.1))
		self.add_module(Bottleneck(Type='normal',input_dims,output_dims,0.1))
		self.add_module(Bottleneck(Type='dilated 8',input_dims,output_dims,0.1))
		self.add_module(Bottleneck(Type='asymmetric',input_dims,output_dims,0.1))
		self.add_module(Bottleneck(Type='dilated 16',input_dims,output_dims,0.1))

class Decoder(nn.Sequential):

	def __init__(self):
		super(Decoder).__init__()



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


		
	def forward(self,x):
		x=self.net['Initial'](x)
		return self.net[Block_1](x)
		

'''Embedding'''
class Embedding(nn.Module):
	
	def __init__(self,embed_dims):
		super(Embedding).__init__()
		

	def forward(self):
		pass

'''Segmentation'''
class Segmentation(nn.Module):
	
	def __init__(self):
		super(Segmentation).__init__()
		pass

	def forward(self):
		pass


'''LaneNet'''
class LaneNet(nn.Module):

	def __init__(self):
		super(LaneNet).__init__()
		self.net=nn.ModuleDict({'Shared_Encoder':,
			                    'Embedding':,
			                    'Segmentation':})
		
	def forward(self,x):
		x=self.net['Shared_Encoder'](x)
		x=self.net['Embedding'](x)
		x=self.net['Segmentation'](x)
		return x


'''H-Net'''

class HNet(nn.Module):

	def __init__(self):
		super(HNet).__init__()


	def forward(self):
		pass
		