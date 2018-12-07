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
        return torch.cat([x,y],dim=0)

class Bottleneck(nn.Module):

	def __init__(self,Type='downsampling',input_num,output_num,P):
		super(Bottleneck).__init__()
		self.Type=Type
		if Type=='downsampling':
		    self.net=nn.ModuleDict({'conv1':nn.Conv2d(input_num,output_num,2,stride=2),
		    	                    'act1':nn.PReLu(),
		    	                    'batch1':nn.BatchNorm2d(ouput_num),
		    	                    'conv2':nn.Conv2d(input_num,output_num,3),
		    	                    'act2':nn.PReLu(),
		    	                    'batch2':nn.BatchNorm2d(output_num),
		    	                    'conv3':nn.Conv2d(input_num,output_num,1),
		    	                    'act3':nn.PReLu(),
		    	                    'batch3':nn.BatchNorm2d(output_num),
		    	                    'dropout':nn.Dropout2d(p=P),
		    	                    'Pooling':nn.AdaptiveMaxPool2d()})

		elif Type.split()[0]=='asymmetric':
			self.net=nn.ModuleDict({'conv1':nn.Conv2d(input_num,input_num/2,1),
				                    'act1':nn.PReLu(),
				                    'batch1':nn.BatchNorm2d(input_num/2),
				                    'conv2':nn.Sequential(nn.Conv2d(input_num/2,input_num/2,(5,1)),
				                    	                  nn.Conv2d(input_num/2,input_num/2,(1,5))),
				                    'act2':nn.PReLu(),
				                    'batch2':nn.BatchNorm2d(input_num/2),
				                    'conv3':nn.Conv2d(input_num/2,output_num,1),
				                    'act3':nn.PReLu(),
				                    'batch3':nn.BatchNorm2d(output_num),
				                    'dropout':nn.Dropout2d(p=P)})

		elif Type.split()[0]=='dilated':
			self.net=nn.ModuleDict({'conv1':nn.Conv2d(input_num,input_num/2,1),
				                    'act1':nn.PReLu(),
				                    'batch1':nn.BatchNorm2d(input_num/2),
				                    'conv2':nn.Conv2d(input_num/2,input_num/2,3,dilation=int(Type.split()[1])),
				                    'act2':nn.PReLu(),
				                    'batch2':nn.BatchNorm2d(input_num/2),
				                    'conv3':nn.Conv2d(input_num/2,output_num,1),
				                    'act3':nn.PReLu(),
				                    'batch3':nn.BatchNorm2d(output_num),
				                    'dropout':nn.Dropout2d(p=P)})
		elif Type=='Normal':
			self.net=nn.ModuleDict({'conv1':nn.Conv2d(input_num,input_num/2,1),
				                    'act1':nn.PReLu(),
				                    'batch1':nn.BatchNorm2d(input_num/2),
				                    'conv2':nn.Conv2d(input_num/2,input_num/2,3),
				                    'act2':nn.PReLu(),
				                    'batch2':nn.BatchNorm2d(input_num/2),
				                    'conv3':nn.Conv2d(input_num/2,output_num,1),
				                    'act3':nn.PReLu(),
				                    'batch3':nn.BatchNorm2d(output_num)})

	def forward(self,x):
		if self.Type=='donwsampling':
			x=self.net['conv1']


'''Shared Encoder'''
class SharedEncoder(nn.Module):

	def __init__(self):
		self.net=nn.ModuleDict({'Initial':Initial(),
			                    'Block_1':Bottleneck()})


		
	def forward(self,x):
		x=self.net['Initial'](x)
		return self.net[Block_1](x)
		

'''Embedding'''
class Embedding(nn.Module):
	
	def __init__(self,embed_dims):
		super(Embedding).__init__()
		pass

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
		