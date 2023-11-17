from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import random



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
		self.conv1 			= torch.nn.Conv2d(input_dim[0],16,3,1,1)
		self.conv2 			= torch.nn.Conv2d(16,64,5,1,1)
		self.conv3 			= torch.nn.Conv2d(64,128,5,1,1)
		self.avgpool 		= torch.nn.AdaptiveAvgPool2d(1)
		self.lin1 			= torch.nn.Linear(128,32)
		self.lin2 			= torch.nn.Linear(32,4)

		self.model 			= torch.nn.ModuleList([self.conv1,self.conv2,self.conv3,self.avgpool,self.lin1,self.lin2])
	
	def forward(self,x:torch.Tensor) ->torch.Tensor:

		x 					= self.activation_fn(self.conv1(x))
		x 					= self.activation_fn(self.conv2(x))
		x 					= self.activation_fn(self.conv3(x))
		x 					= self.avgpool(x)
		x 					= x.view(x.shape[0],-1)
		x 					= self.activation_fn(self.lin1(x))
		x 					= self.lin2(x)

		return 				x 
