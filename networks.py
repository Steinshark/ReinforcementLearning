from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import torch.nn.functional as fun 



class SnakeNetwork (torch.nn.Module):
    
	def __init__(self,input_dim,loss_fn,activation_fn):

		#Init Super
		super(SnakeNetwork,self).__init__()

		#Set These 
		self.activation_fn 	= activation_fn
		self.loss_fn 		= loss_fn()


	
	def forward(self,x:torch.Tensor) -> torch.Tensor:
		print(f"Not implemented yet!")


class SnakeConvNetSm (SnakeNetwork):

	def __init__(self,input_dim,loss_fn,activation_fn):

		super(SnakeConvNetSm,self).__init__(input_dim,loss_fn,activation_fn)


		#Create model 
		self.conv1 			= torch.nn.Conv2d(input_dim[0],16,3,1,1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		self.conv2 			= torch.nn.Conv2d(16,64,5,1,1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		self.avgpool 		= torch.nn.AdaptiveAvgPool2d(4).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		self.lin1 			= torch.nn.Linear(64*4*4,128).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		self.lin2 			= torch.nn.Linear(128,4).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	
	def forward(self,x:torch.Tensor) ->torch.Tensor:

		x 					= self.activation_fn(self.conv1(x))
		x 					= self.activation_fn(self.conv2(x))
		#x 					= self.activation_fn(self.conv3(x))
		x 					= self.avgpool(x)
		x 					= x.view(x.shape[0],-1)
		#print(f"x is shape {x.shape}")
		x 					= self.activation_fn(self.lin1(x))
		x 					= self.lin2(x)

		return 				x 


class SnakeConvNetMd (SnakeNetwork):

	def __init__(self,input_dim,loss_fn,activation_fn):
		print(f"act is {activation_fn}")
		for x,y in [(fun.relu,nn.ReLU),(fun.relu6,nn.ReLU6),(fun.leaky_relu,nn.LeakyReLU),(fun.gelu,nn.GELU),(fun.tanh,nn.Tanh),(fun.elu,nn.ELU),(fun.celu,nn.CELU)]:
			if activation_fn is x: 
				print(f"activation fun was {activation_fn}")
				activation_fn = y
				print(f"now is {activation_fn}")
		super(SnakeConvNetMd,self).__init__(input_dim,loss_fn,activation_fn)


		#Create model 
		self.downsample_1 	= torch.nn.Sequential(
			torch.nn.Conv2d(input_dim[0],16,3,1,1,bias=False),
			torch.nn.BatchNorm2d(16),
			activation_fn(),

			torch.nn.Conv2d(16,16,3,2,1,bias=False),
			torch.nn.BatchNorm2d(16),
			activation_fn()
		).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))	# 20x20

		self.downsample_2 	= torch.nn.Sequential(
			torch.nn.Conv2d(16,32,3,1,1,bias=False),
			torch.nn.BatchNorm2d(32),
			activation_fn(),

			torch.nn.Conv2d(32,32,3,2,1,bias=False),
			torch.nn.BatchNorm2d(32),
			activation_fn()
		).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))	# 10x10

		self.downsample_3 	= torch.nn.Sequential(
			torch.nn.Conv2d(32,64,3,1,1,bias=False),
			torch.nn.BatchNorm2d(64),
			activation_fn(),

			torch.nn.AdaptiveAvgPool2d(8)
		).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))	#8x8

		self.downsample_4 	= torch.nn.Sequential(
			torch.nn.Conv2d(64,128,3,1,1,bias=False),
			torch.nn.BatchNorm2d(128),
			activation_fn(),

			torch.nn.Conv2d(128,128,3,2,1,bias=False),
			torch.nn.BatchNorm2d(128),
			activation_fn(),
			
		).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))	# 4x4

		self.output 		= torch.nn.Sequential(
			torch.nn.Conv2d(128,4,(4,4),1,0,bias=True),
			torch.nn.Flatten(start_dim=1)
		).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 4


	
	def forward(self,x:torch.Tensor) ->torch.Tensor:

		y 			= self.downsample_1(x)
		y 			= self.downsample_2(y)
		y 			= self.downsample_3(y)
		y 			= self.downsample_4(y)
		y 			= self.output(y)	
		return 				y		


class SnakeConvNetLg (SnakeNetwork):

	def __init__(self,input_dim,loss_fn,activation_fn):

		super(SnakeConvNetLg,self).__init__(input_dim,loss_fn,activation_fn)


		#Create model 
		self.conv1 			= torch.nn.Conv2d(input_dim[0],16,3,1,1)
		self.conv2 			= torch.nn.Conv2d(16,32,5,1,2)
		self.conv3 			= torch.nn.Conv2d(32,64,5,1,2)
		self.conv4 			= torch.nn.Conv2d(64,128,5,1,2)
		self.conv5 			= torch.nn.Conv2d(128,256,5,1,2)
		
		self.avgpool 		= torch.nn.AdaptiveAvgPool2d(8)
		self.lin1 			= torch.nn.Linear(256*8*8,512)
		self.lin2 			= torch.nn.Linear(512,4)

		
	
	def forward(self,x:torch.Tensor) ->torch.Tensor:

		x 					= self.activation_fn(self.conv1(x))
		x 					= self.activation_fn(self.conv2(x))
		x 					= self.activation_fn(self.conv3(x))
		x 					= self.activation_fn(self.conv4(x))
		x 					= self.activation_fn(self.conv5(x))
		x 					= self.avgpool(x)
		x 					= x.view(x.shape[0],-1)
		x 					= self.activation_fn(self.lin1(x))
		x 					= self.lin2(x)

		return 				x 


class SnakeAdaptNet (SnakeNetwork):

	def __init__(self,input_dim,loss_fn,activation_fn):

		super(SnakeAdaptNet,self).__init__(input_dim,loss_fn,activation_fn)

		adapt_1 			= 32
		adapt_2 			= 16
		adapt_3 			= 8
		bias				= False

		#Create model 
		self.conv1 			= torch.nn.Conv2d(input_dim[0],8,5,1,0,bias=bias)
		self.bn1 			= torch.nn.BatchNorm2d(8)

		self.conv2 			= torch.nn.Conv2d(8,16,5,1,2,bias=bias)
		self.bn2 			= torch.nn.BatchNorm2d(16)

		self.conv3 			= torch.nn.Conv2d(16,32,5,1,2,bias=bias)
		self.bn3 			= torch.nn.BatchNorm2d(32)
		self.avgpool1 		= torch.nn.AdaptiveAvgPool2d(adapt_1)

		self.conv4 			= torch.nn.Conv2d(32,32,5,1,2,bias=bias)
		self.bn4 			= torch.nn.BatchNorm2d(32)
		self.avgpool2 		= torch.nn.AdaptiveAvgPool2d(adapt_2)

		self.conv5 			= torch.nn.Conv2d(32,32,5,1,2,bias=bias)
		self.bn5 			= torch.nn.BatchNorm2d(32)
		self.avgpool3 		= torch.nn.AdaptiveAvgPool2d(adapt_3)
	
		self.lin1 			= torch.nn.Linear(32*8*8,128)
		self.lin2 			= torch.nn.Linear(128,4)

		
	
	def forward(self,x:torch.Tensor) ->torch.Tensor:
		x 					= self.activation_fn(self.bn1(self.conv1(x)))

		x 					= self.activation_fn(self.bn2(self.conv2(x)))

		x 					= self.activation_fn(self.bn3(self.conv3(x)))

		x 					= self.avgpool1(x)
		x 					= self.activation_fn(self.bn4(self.conv4(x)))

		x 					= self.avgpool2(x)
		x 					= self.activation_fn(self.bn5(self.conv5(x)))

		x 					= self.avgpool3(x)

		x 					= x.view(x.shape[0],-1)

		x 					= self.activation_fn(self.lin1(x))
		x 					= self.lin2(x)

		return 				x 


class SnakeVarNet (SnakeNetwork):

	def __init__(self,input_dim,loss_fn,activation_fn):

		super(SnakeVarNet,self).__init__(input_dim,loss_fn,activation_fn)

		adapt_1 			= 64#int(input_dim[1]/2)
		adapt_2 			= 32
		adapt_3 			= 16#int(input_dim[1]/4)
		adapt_4 			= 8

		#Create model 
		self.conv1 			= torch.nn.Conv2d(input_dim[0],8,3,1,0)

		self.conv2 			= torch.nn.Conv2d(8,16,3,1,1)

		self.conv3 			= torch.nn.Conv2d(16,64,5,2,2)

		self.conv5 			= torch.nn.Conv2d(64,256,5,2,2)

		self.conv7 			= torch.nn.Conv2d(256,512,5,2,2)

		self.avgpool4		= torch.nn.AdaptiveAvgPool2d(adapt_4)


		self.lin1 			= torch.nn.Linear(512*adapt_4*adapt_4,512)

		self.lin2 			= torch.nn.Linear(512,4)

		
	
	def forward(self,x:torch.Tensor) ->torch.Tensor:

		x 					= self.activation_fn(self.conv1(x))
		x 					= self.activation_fn(self.conv2(x))
		x 					= self.avgpool1(x)

		x 					= self.activation_fn(self.conv3(x))
		x 					= self.avgpool2(x)

		x 					= self.activation_fn(self.conv5(x))
		x 					= self.avgpool3(x)

		x 					= self.activation_fn(self.conv7(x))
		x 					= self.avgpool4(x)

		x 					= x.view(x.shape[0],-1)
		x 					= self.activation_fn(self.lin1(x))
		x 					= self.lin2(x)

		return 				x 

