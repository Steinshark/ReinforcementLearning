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
		self.layer1 		= torch.nn.Sequential(torch.nn.Conv2d(input_dim[0],64,5,1,2),torch.nn.BatchNorm2d(64))
		self.layer2			= torch.nn.Sequential(torch.nn.Conv2d(64,64,5,1,2),torch.nn.BatchNorm2d(64))
		self.layer3			= torch.nn.Sequential(torch.nn.Conv2d(64,64,(input_dim[1],input_dim[2]),1,0))
		self.linear1		= torch.nn.Sequential(torch.nn.Flatten(start_dim=1),torch.nn.Linear(64,64))
		self.output_layer	= torch.nn.Sequential(torch.nn.Linear(64,4))
	

	def forward(self,x:torch.Tensor) ->torch.Tensor:

		x 					= self.activation_fn(self.layer1(x))
		x 					= self.activation_fn(self.layer2(x))
		x 					= self.activation_fn(self.layer3(x))
		x 					= self.activation_fn(self.linear1(x))
		return 				self.output_layer(x)


class SnakeConvNetMd (SnakeNetwork):

	def __init__(self,input_dim,loss_fn,activation_fn):
		super(SnakeConvNetMd,self).__init__(input_dim,loss_fn,activation_fn)


		self.layer1 		= torch.nn.Sequential(torch.nn.Conv2d(input_dim[0],32,3,1,1),torch.nn.BatchNorm2d(32))
		self.layer2 		= torch.nn.Sequential(torch.nn.Conv2d(32,32,3,1,1),torch.nn.BatchNorm2d(32))
		self.layer3 		= torch.nn.Sequential(torch.nn.Conv2d(32,32,3,1,1),torch.nn.BatchNorm2d(32))
		self.layer4 		= torch.nn.Sequential(torch.nn.Conv2d(32,32,3,1,1),torch.nn.BatchNorm2d(32))
		self.layer5 		= torch.nn.Sequential(torch.nn.Conv2d(32,32,3,1,1),torch.nn.BatchNorm2d(32))
		self.layer6 		= torch.nn.Sequential(torch.nn.Conv2d(32,32,3,1,1),torch.nn.BatchNorm2d(32))
		self.layer7 		= torch.nn.Sequential(torch.nn.Conv2d(32,32,3,1,1),torch.nn.BatchNorm2d(32))
		self.layer8 		= torch.nn.Sequential(torch.nn.Conv2d(32,32,3,1,1),torch.nn.BatchNorm2d(32))
		self.output_layer 	= torch.nn.Sequential(torch.nn.Flatten(start_dim=1),torch.nn.Linear(32*input_dim[1]*input_dim[2],4))


	
	def forward(self,x:torch.Tensor) ->torch.Tensor:

		y 			= self.layer1(x)
		y 			= self.layer2(y)
		y 			= self.layer3(y)
		y 			= self.layer4(y)
		y 			= self.layer5(y)
		y 			= self.layer6(y)
		y 			= self.layer7(y)
		y 			= self.layer8(y)
		return 		self.output_layer(y)	


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

		adapt_1 			= 16
		adapt_2 			= 8
		adapt_3 			= 4
		bias				= False

		#Create model 
		self.conv1 			= torch.nn.Conv2d(input_dim[0],16,5,1,0,bias=bias)
		self.bn1 			= torch.nn.BatchNorm2d(16)

		self.conv2 			= torch.nn.Conv2d(16,16,5,1,2,bias=bias)
		self.bn2 			= torch.nn.BatchNorm2d(16)
		self.avgpool1 		= torch.nn.AdaptiveAvgPool2d(adapt_1)


		self.conv3 			= torch.nn.Conv2d(16,64,5,1,2,bias=bias)
		self.bn3 			= torch.nn.BatchNorm2d(64)
		self.avgpool2 		= torch.nn.AdaptiveAvgPool2d(adapt_2)

		self.conv4 			= torch.nn.Conv2d(64,256,5,1,2,bias=bias)
		self.bn4 			= torch.nn.BatchNorm2d(256)
		self.avgpool3 		= torch.nn.AdaptiveAvgPool2d(adapt_3)

		self.conv5 			= torch.nn.Conv2d(256,1024,(adapt_3,adapt_3),1,0,bias=bias)
		self.bn5 			= torch.nn.BatchNorm2d(1024)
	
		self.lin1 			= torch.nn.Linear(1024,128)
		self.lin2 			= torch.nn.Linear(128,4)

		
	
	def forward(self,x:torch.Tensor) ->torch.Tensor:
		x 					= self.activation_fn(self.bn1(self.conv1(x)))
		x 					= self.activation_fn(self.bn2(self.conv2(x)))
		x 					= self.avgpool1(x)


		x 					= self.activation_fn(self.bn3(self.conv3(x)))
		x 					= self.avgpool2(x)

		x 					= self.activation_fn(self.bn4(self.conv4(x)))
		x 					= self.avgpool3(x)

		x 					= self.activation_fn(self.bn5(self.conv5(x)))

		x 					= x.view(x.shape[0],-1)

		x 					= self.activation_fn(self.lin1(x))

		return 				self.lin2(x)


class SnakeMegaNet (SnakeNetwork):

	def __init__(self,input_dim,loss_fn,activation_fn):

		super(SnakeMegaNet,self).__init__(input_dim,loss_fn,activation_fn)

		bias 				= False
		adapt_1 			= 16
		adapt_2 			= 8
		adapt_3 			= 4

		#Create model 
		self.conv1 			= torch.nn.Conv2d(input_dim[0],8,5,1,2,bias=bias)
		self.bn1 			= torch.nn.BatchNorm2d(8)
		self.conv2 			= torch.nn.Conv2d(8,16,5,1,2,bias=bias)
		self.bn2 			= torch.nn.BatchNorm2d(16)
		self.conv3 			= torch.nn.Conv2d(16,32,5,1,2,bias=bias)
		self.bn3 			= torch.nn.BatchNorm2d(32)
		self.conv4 			= torch.nn.Conv2d(32,64,5,1,2,bias=bias)
		self.bn4 			= torch.nn.BatchNorm2d(64)
		self.avgpool1		= torch.nn.AdaptiveAvgPool2d(adapt_1)


		self.conv5 			= torch.nn.Conv2d(64,128,5,1,2,bias=bias)
		self.bn5 			= torch.nn.BatchNorm2d(128)
		self.conv6 			= torch.nn.Conv2d(128,256,5,1,2,bias=bias)
		self.bn6 			= torch.nn.BatchNorm2d(256)
		self.avgpool2 		= torch.nn.AdaptiveAvgPool2d(adapt_2)

		self.conv7 			= torch.nn.Conv2d(256,512,5,1,2,bias=bias)
		self.bn7			= torch.nn.BatchNorm2d(512)
		self.conv8 			= torch.nn.Conv2d(512,1024,5,1,2,bias=bias)
		self.bn8			= torch.nn.BatchNorm2d(1024)
		self.avgpool3 		= torch.nn.AdaptiveAvgPool2d(adapt_3)

		self.conv9 			= torch.nn.Conv2d(1024,1024,(adapt_3,adapt_3),1,0)
		self.flat1 			= torch.nn.Flatten(start_dim=1)


		self.lin1 			= torch.nn.Linear(1024,128)

		self.lin2 			= torch.nn.Linear(128,4)

	
	
	def forward(self,x:torch.Tensor) ->torch.Tensor:
		
		x 					= self.activation_fn(self.bn1(self.conv1(x)))
		x 					= self.activation_fn(self.bn2(self.conv2(x)))
		x 					= self.activation_fn(self.bn3(self.conv3(x)))
		x 					= self.activation_fn(self.bn4(self.conv4(x)))
		x 					= self.avgpool1(x)

		x 					= self.activation_fn(self.bn5(self.conv5(x)))
		x 					= self.activation_fn(self.bn6(self.conv6(x)))
		x 					= self.avgpool2(x)

		x 					= self.activation_fn(self.bn7(self.conv7(x)))
		x 					= self.activation_fn(self.bn8(self.conv8(x)))
		x 					= self.avgpool3(x)

		x 					= self.activation_fn(self.conv9(x))
		x 					= self.flat1(x)

		x 					= self.activation_fn(self.lin1(x))
		return 				self.lin2(x)




if __name__ == "__main__":

	m = SnakeConvNetSm((3,32,32),torch.nn.MSELoss,torch.nn.functional.relu)
	print(f"n params: {sum([p.numel() for p in m.parameters()])}")

	m = SnakeConvNetMd((3,32,32),torch.nn.MSELoss,torch.nn.functional.relu)
	print(f"n params: {sum([p.numel() for p in m.parameters()])}")