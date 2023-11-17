from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import random

class FullyConnectedNetwork(nn.Module):
	def __init__(self,input_dim,output_size,loss_fn=None,optimizer_fn=None,lr=1e-6,wd=1e-6,architecture=[512,32,16]):
		super(FullyConnectedNetwork,self).__init__()

		self.model = [nn.Linear(input_dim,architecture[0])]
		self.model.append(nn.LeakyReLU(.2))

		for i,size in enumerate(architecture[:-1]):
			
			self.model.append(nn.Linear(size,architecture[i+1]))
			self.model.append(nn.LeakyReLU(.1))
		self.model.append(nn.Linear(architecture[-1],output_size))

		od = OrderedDict({str(i):self.model[i] for i in range(len(self.model))})
		self.model = nn.Sequential(od)
		self.optimizer = optimizer_fn(self.model.parameters(),lr=lr,weight_decay=wd)
		self.loss = loss_fn()

	def train(self,x_input,y_actual,epochs=1000,verbose=False,show_steps=10,batch_size="online",show_graph=False):
		memory = 3
		prev_loss = [100000000 for x in range(memory)]
		losses = []
		if type(batch_size) is str:
			batch_size = len(y_actual)

		if verbose:
			print(f"Training on dataset shape:\t f{x_input.shape} -> {y_actual.shape}")
			print(f"batching size:\t{batch_size}")

		#Create the learning batches
		dataset = torch.utils.data.TensorDataset(x_input,y_actual)
		dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)


		for i in range(epochs):
			#Track loss avg in batch
			avg_loss = 0

			for batch_i, (x,y) in enumerate(dataloader):

				#Find the predicted values
				batch_prediction = self.forward(x)
				#Calculate loss
				loss = self.loss(batch_prediction,y)
				avg_loss += loss
				#Perform grad descent
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			avg_loss = avg_loss / batch_i 			# Add losses to metric's list
			losses.append(avg_loss.cpu().detach().numpy())

			#Check for rising error
			if not False in [prev_loss[x] > prev_loss[x+1] for x in range(len(prev_loss)-1)]:
				print(f"broke on epoch {i}")
				break
			else:
				prev_loss = [avg_loss] + [prev_loss[x+1] for x in range(len(prev_loss)-1)]

			#Check for verbosity
			if verbose and i % show_steps == 0:
				print(f"loss on epoch {i}:\t{loss}")

		if show_graph:
			plt.plot(losses)
			plt.show()


	def forward(self,x_list):
		x_list = torch.flatten(x_list,start_dim=1)
		return self.model(x_list)
		#	y_predicted.append(y_pred.cpu().detach().numpy())

class ConvolutionalNetwork(nn.Module):
	
	def __init__(self,loss_fn=None,optimizer_fn=None,kwargs={},architecture:list=[],input_shape=(1,3,30,20),device=torch.device("cpu"),verbose=False):
		super(ConvolutionalNetwork,self).__init__()
		self.input_shape 	= input_shape
		through 			= torch.ones(size=input_shape,device=device)
		for module_i in range(len(architecture)):
			architecture[module_i] = architecture[module_i].to(device)

		module  = architecture[0] 
		ch_in   = input_shape[1]
		ch_out  = module.out_channels
		pad     = module.padding
		kernel  = module.kernel_size
		stride  = module.stride
		architecture[0] = torch.nn.Conv2d(ch_in,ch_out,kernel,stride,pad,device=device,bias=False)
		
		for i,module in enumerate(architecture):

			if "Flatten" in str(module):
				through = module(through)
				flat_size = through.size()[1]
				old_outs = architecture[i+1].out_features
				try:
					old_next_outs = architecture[i+3].out_features
				except IndexError:
					architecture[i+1] = torch.nn.Linear(flat_size,4)
					break
				while flat_size <= old_outs*2:
					old_outs /= 2
					old_outs = int(old_outs) 
				architecture[i+1] = torch.nn.Linear(flat_size,old_outs,device=device)
				architecture[i+3] = torch.nn.Linear(old_outs,old_next_outs,device=device)
				break
			else:
				through = module(through)
		o_d 				= OrderedDict({str(i) : n for i,n in enumerate(architecture)})
		self.model 			= nn.Sequential(o_d)
		self.loss = loss_fn()
		self.optimizer = optimizer_fn(self.model.parameters(),**kwargs)
		self.to(device)

		if verbose:
			print(f"generated model with {sum([p.numel() for p in self.model.parameters()])} params")
		
	def train(self,x_input,y_actual,epochs=10,in_shape=(1,6,10,10)):

		#Run epochs
		for i in range(epochs):
			
			#Predict on x : M(x) -> y
			y_pred = self.model(x_input)
			#Find loss  = y_actual - y
			loss = self.loss_function(y_pred,y_actual)
			print(f"epoch {i}:\nloss = {loss}")

			#Update network
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

	def forward(self,x):
		#if len(x.shape) == 3:
			#input(f"received input shape: {x.shape}")
			#x = torch.reshape(x,self.input_shape)
		return self.model(x)

class ConvNetSmO(nn.Module):
	def __init__(self,in_ch=4,loss_fn=nn.MSELoss,optimizer=torch.optim.Adam,lr=.0001,act_fn=torch.nn.LeakyReLU,kernel_size=3,w=0,h=0):
		super(ConvNetSmO,self).__init__()

		flattened_size 		= w * h * 32 
		if flattened_size > 1024:
			lin_n 				= 1024 
			lin_n2 				= 512
		elif flattened_size > 256:
			lin_n 				= 256 
			lin_n2 				= 128
		else:
			lin_n 				= 128 
			lin_n2 				= 64
		
		self.model 	= torch.nn.Sequential(
			torch.nn.Conv2d(in_ch,8,kernel_size,1,1),
			act_fn(),

			torch.nn.Conv2d(8,16,kernel_size,1,1),
			act_fn(),

			torch.nn.Conv2d(16,32,kernel_size,1,1),
			act_fn(),

			torch.nn.Flatten(),

			torch.nn.Linear(flattened_size,lin_n),
			act_fn(),

			torch.nn.Linear(lin_n,lin_n2),
			act_fn(),

			torch.nn.Linear(lin_n2,4)
		)

		self.optimizer	= optimizer(self.model.parameters(),lr=lr)
		self.loss 		= loss_fn()

	def forward(self,x):
		y 		= self.model(x)
		return y

class ConvNetSm(nn.Module):
	def __init__(self,in_ch=4,loss_fn=nn.MSELoss,optimizer=torch.optim.Adam,lr=.0001,act_fn=torch.nn.functional.relu,kernel_size=3,w=0,h=0):
		super(ConvNetSm,self).__init__()

		flattened_size 		= w * h * 32 
		self.act_fn 		= act_fn
		if flattened_size > 1024:
			lin_n 				= 1024 
			lin_n2 				= 512
		elif flattened_size > 256:
			lin_n 				= 256 
			lin_n2 				= 128
		else:
			lin_n 				= 128 
			lin_n2 				= 64
		
		self.conv1 			= torch.nn.Conv2d(in_ch,16,kernel_size,1,1)
		self.conv2			= torch.nn.Conv2d(16,32,kernel_size,1,1)
		self.conv3			= torch.nn.Conv2d(32,64,kernel_size,1,1)
		self.conv4			= torch.nn.Conv2d(64,128,kernel_size,1,1)
		self.avgpool 		= torch.nn.AdaptiveMaxPool2d(1)

		self.lin1 			= torch.nn.Linear(128,4)

		self.model 			= torch.nn.ModuleList([self.conv1,self.conv2,self.conv3,self.conv4,self.avgpool,self.lin1])

		self.optimizer	= optimizer(self.model.parameters(),lr=lr)
		self.loss 		= loss_fn()

	def forward(self,x):
		x 		= self.act_fn(self.conv1(x))
		x 		= self.act_fn(self.conv2(x))
		x 		= self.act_fn(self.conv3(x))
		x 		= self.act_fn(self.conv4(x))
		x 		= self.avgpool(x)
		x 		= x.view(x.size(0), -1)
		x 		= self.lin1(x)
		return x 

class ConvNet(nn.Module):
	def __init__(self,in_ch=4,loss_fn=nn.MSELoss,optimizer=torch.optim.Adam,lr=.0001,act_fn=torch.nn.LeakyReLU,kernel_size=5,w=0,h=0):
		super(ConvNet,self).__init__()
		flattened_size 		= (w-4) * (h-4) * 64 
		if flattened_size > 1024:
			lin_n 				= 1024 
			lin_n2 				= 512
		elif flattened_size > 256:
			lin_n 				= 256 
			lin_n2 				= 128
		else:
			lin_n 				= 128 
			lin_n2 				= 64
		self.model 	= torch.nn.Sequential(
			torch.nn.Conv2d(in_ch,16,kernel_size,1,2),
			act_fn(),

			torch.nn.Conv2d(16,32,kernel_size,1,2),
			act_fn(),

			torch.nn.Conv2d(32,64,kernel_size,1,2),
			act_fn(),

			torch.nn.Conv2d(64,64,kernel_size,1,1),
			act_fn(),

			torch.nn.Conv2d(64,64,kernel_size,1,1),
			act_fn(),

			torch.nn.Flatten(),

			torch.nn.Linear(flattened_size,lin_n),
			act_fn(),

			torch.nn.Linear(lin_n,lin_n2),
			act_fn(),

			torch.nn.Linear(lin_n2,4)
		)
		self.optimizer	= optimizer(self.model.parameters(),lr=lr)
		self.loss 		= loss_fn()

	def forward(self,x):
		y 		= self.model(x)
		return y

class ConvNet20(nn.Module):
	def __init__(self,loss_fn=nn.MSELoss,optimizer=torch.optim.Adam,lr=.0001,act_fn=torch.nn.LeakyReLU,in_ch=4,w=0,h=0):
		super(ConvNet20,self).__init__()

		self.model 	= torch.nn.Sequential(
			torch.nn.Conv2d(in_ch,16,3,1,1),
			torch.nn.BatchNorm2d(16),
			act_fn(),

			torch.nn.Conv2d(16,32,3,1,0),			#w-2 
			torch.nn.BatchNorm2d(32),
			act_fn(),

			torch.nn.Conv2d(32,64,3,1,0),			#w-4
			torch.nn.BatchNorm2d(64),
			act_fn(),

			torch.nn.Conv2d(64,64,3,1,0),			#w-6
			torch.nn.BatchNorm2d(64),
			act_fn(),

			torch.nn.Conv2d(64,64,3,1,0),			#w-8
			torch.nn.BatchNorm2d(64),
			act_fn(),

			torch.nn.Conv2d(64,64,3,1,0),			#w-10
			torch.nn.BatchNorm2d(64),
			act_fn(),

			torch.nn.Flatten(),

			torch.nn.Linear((w-10)*(h-10)*64,64),
			act_fn(),
			
			torch.nn.Linear(64,4),
		)

		self.optimizer	= optimizer(self.model.parameters(),lr=lr)
		self.loss 		= loss_fn()

	def forward(self,x):
		y 		= self.model(x)
		return y


class FullConvNet(nn.Module):
	def __init__(self,loss_fn=nn.MSELoss,optimizer=torch.optim.Adam,lr=.0001,act_fn=torch.nn.LeakyReLU,in_ch=4,w=0,h=0):
		super(FullConvNet,self).__init__()
		final_layer 	= int(w/8) * int(h/8) * 128

		self.model 	= torch.nn.Sequential(

			torch.nn.Conv2d(in_ch,16,3,1,1),	# w * h 
			torch.nn.MaxPool2d(2),
			act_fn(),

			torch.nn.Conv2d(16,32,3,1,1),		# (w/2) * (h/2)
			torch.nn.MaxPool2d(2),
			act_fn(),

			torch.nn.Conv2d(32,64,3,1,1),		# (w/4) * (h/4)
			torch.nn.MaxPool2d(2),
			act_fn(),

			torch.nn.Conv2d(64,128,3,1,1),		# (w/8) * (h/8)
			act_fn(),

			torch.nn.Flatten(),

			torch.nn.Linear(final_layer,4)
		)

		self.optimizer	= optimizer(self.model.parameters(),lr=lr)
		self.loss 		= loss_fn()

	def forward(self,x):
		y 		= self.model(x)
		return y

class FCN(nn.Module):
	def __init__(self,loss_fn=nn.MSELoss,optimizer:torch.optim=torch.optim.Adam,lr=.0001,act_fn=torch.nn.LeakyReLU,in_ch=4,w=0,h=0):
		super(FCN,self).__init__()

		in_layer 		= w*h*3*2

		self.model 	= torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(in_layer,512),
			torch.nn.Dropout(.5),
			act_fn(),

			torch.nn.Linear(512,256),
			torch.nn.Dropout(.2),
			act_fn(),

			torch.nn.Linear(256,4)
		)

		if isinstance(optimizer,torch.optim.SGD):
			self.optimizer	= optimizer(self.model.parameters(),lr=lr,momentum=.75)
		else:	
			self.optimizer	= optimizer(self.model.parameters(),lr=lr)
		self.loss 		= loss_fn()

	def forward(self,x):
		y 		= self.model(x)
		return y

class FCN2(nn.Module):
	def __init__(self,loss_fn=nn.MSELoss,optimizer=torch.optim.Adam,lr=.0001,act_fn=torch.nn.LeakyReLU,in_ch=4,w=0,h=0):
		super(FCN2,self).__init__()

		in_layer 		= w*h*3*2

		self.model 	= torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(in_layer,256),
			torch.nn.Dropout(.5),
			act_fn(),

			torch.nn.Linear(256,64),
			torch.nn.Dropout(.25),
			act_fn(),

			torch.nn.Linear(64,32),
			torch.nn.Dropout(.1),
			act_fn(),

			torch.nn.Linear(32,32),
			torch.nn.Dropout(.05),
			act_fn(),

			torch.nn.Linear(32,4)
		)

		self.optimizer	= optimizer(self.model.parameters(),lr=lr)
		self.loss 		= loss_fn()

	def forward(self,x):
		y 		= self.model(x)
		return y
	
class FCN3(nn.Module):
	def __init__(self,loss_fn=nn.MSELoss,optimizer=torch.optim.Adam,lr=.0001,act_fn=torch.nn.LeakyReLU,in_ch=4,w=0,h=0):
		super(FCN3,self).__init__()

		in_layer 		= w*h*3*2

		self.model 	= torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(in_layer,128),
			torch.nn.Dropout(.5),
			act_fn(),

			torch.nn.Linear(128,64),
			torch.nn.Dropout(.5),
			act_fn(),

			torch.nn.Linear(64,32),
			#torch.nn.Dropout(.25),
			act_fn(),

			torch.nn.Linear(32,32),
			#torch.nn.Dropout(.1),
			act_fn(),

			torch.nn.Linear(32,32),
			#torch.nn.Dropout(.05),
			act_fn(),

			torch.nn.Linear(32,32),
			#torch.nn.Dropout(),
			act_fn(),

			torch.nn.Linear(32,4),
			torch.nn.Tanh()
		)

		self.optimizer	= optimizer(self.model.parameters(),lr=lr)
		self.loss 		= loss_fn()

	def forward(self,x):
		y 		= self.model(x)
		return y
	

class IMG_NET_OG(nn.Module):

	def __init__(self,input_shape=(3,540,960),nf=32,loss_fn=torch.nn.MSELoss,optimizer_fn=torch.optim.Adam,kwargs={"lr":.0001,"betas":(.9,.999)},dropout_p=.25,neg_slope=.2,device=torch.device('cuda')):
		super(IMG_NET,self).__init__()
		self.dropout		= dropout_p
		self.model 			= nn.Sequential(

			# #960x540
			# nn.Conv2d(input_shape[1],32,5,1,2,bias=False),
			# nn.LeakyReLU(negative_slope=.02),
			# nn.BatchNorm2d(32),
			# nn.MaxPool2d(2),		
			#480x270
			nn.Conv2d(3,nf,5,1,2,bias=False),
			nn.BatchNorm2d(nf),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),
			nn.MaxPool2d(2),	
			#240x135
			nn.Conv2d(nf,nf*2,5,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),
			nn.MaxPool2d(2),
			#120x75
			nn.Conv2d(nf*2,nf*4,5,1,1,bias=False),
			nn.BatchNorm2d(nf*4),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),
			nn.MaxPool2d(2),
			#60x37
			nn.Conv2d(nf*4,nf*8,5,1,1,bias=False),
			nn.BatchNorm2d(nf*8),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),
			nn.MaxPool2d(2),
				
			nn.Flatten(1),

			nn.Linear(41472,2048),
			nn.Dropout(p=dropout_p),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),

			nn.Linear(2048,512),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),

			nn.Linear(512,128),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),

			nn.Linear(128,4)
		).to(device)

		self.loss 			= loss_fn()
		self.optimizer		= optimizer_fn(self.model.parameters(),**kwargs)

	def forward(self,x):
		return self.model(x)

class IMG_NET(nn.Module):

	def __init__(self,input_shape=(3,540,960),nf=8,loss_fn=torch.nn.MSELoss,optimizer_fn=torch.optim.Adam,kwargs={"lr":.0001,"betas":(.9,.999)},dropout_p=.25,neg_slope=.05,device=torch.device('cuda')):
		super(IMG_NET,self).__init__()
		self.dropout		= dropout_p
		self.model 			= nn.Sequential(

			#480x270
			nn.Conv2d(3,nf,5,1,2,bias=False),
			nn.BatchNorm2d(nf),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),
			#240x135
			nn.Conv2d(nf,nf*2,5,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),
			nn.MaxPool2d(2),
			#120x75
			nn.Conv2d(nf*2,nf*2,5,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),
			nn.MaxPool2d(2),
			#60x37
			nn.Conv2d(nf*2,nf*2,5,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),
			nn.MaxPool2d(2),

			nn.Conv2d(nf*2,nf*4,5,1,1,bias=False),
			nn.BatchNorm2d(nf*4),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),
			nn.MaxPool2d(2),

				
			nn.Flatten(1),

			nn.Linear(768,512),
			nn.Dropout(p=dropout_p*2),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),

			nn.Linear(512,128),
			nn.Dropout(p=dropout_p),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),

			nn.Linear(128,64),
			nn.Dropout(p=dropout_p/2),
			nn.LeakyReLU(negative_slope=neg_slope),#negative_slope=.02),

			nn.Linear(64,4)
		).to(device)

		self.loss 			= loss_fn()
		self.optimizer		= optimizer_fn(self.model.parameters(),**kwargs)

	def forward(self,x):
		return self.model(x)

class IMG_NET_SIMPLE(nn.Module):

	def __init__(self,input_shape=(3,540,960),nf=8,loss_fn=torch.nn.MSELoss,optimizer_fn=torch.optim.Adam,kwargs={"lr":.0001,"betas":(.9,.999)},act_fn=nn.Tanh,act_kwargs={},dropout_p=.25,neg_slope=.05,device=torch.device('cuda')):
		super(IMG_NET_SIMPLE,self).__init__()
		self.dropout		= dropout_p
		self.act_fn			= act_fn 
		self.act_kwargs 	= act_kwargs
		self.model 			= nn.Sequential(

			#480x270
			nn.Conv2d(3,nf,5,1,2,bias=False),
			#nn.BatchNorm2d(nf),
			self.act_fn(**self.act_kwargs),
			#240x135
			nn.Conv2d(nf,nf*2,5,1,2,bias=False),
			#nn.BatchNorm2d(nf*2),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),
			#120x75
			nn.Conv2d(nf*2,nf*2,5,1,2,bias=False),
			#nn.BatchNorm2d(nf*2),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),
			#60x37
			nn.Conv2d(nf*2,nf*2,5,1,2,bias=False),
			#nn.BatchNorm2d(nf*2),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),

			nn.Conv2d(nf*2,nf*4,5,1,2,bias=False),
			#nn.BatchNorm2d(nf*4),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),

			nn.Conv2d(nf*4,nf*4,5,1,2,bias=False),
			#nn.BatchNorm2d(nf*4),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),
				
			nn.Flatten(1),

			nn.Linear(320,256),
			nn.Dropout(p=dropout_p),
			self.act_fn(**self.act_kwargs),

			nn.Linear(256,128),
			nn.Dropout(p=dropout_p),
			self.act_fn(**self.act_kwargs),

			nn.Linear(128,4)
		).to(device)

		self.loss 			= loss_fn()
		self.optimizer		= optimizer_fn(self.model.parameters(),**kwargs)

	def forward(self,x):
		return self.model(x)

class IMG_NET_COMPLEX(nn.Module):

	def __init__(self,input_shape=(3,540,960),nf=8,loss_fn=torch.nn.MSELoss,optimizer_fn=torch.optim.Adam,kwargs={"lr":.0001,"betas":(.9,.999)},act_fn=nn.LeakyReLU,act_kwargs={},dropout_p=.25,neg_slope=.05,device=torch.device('cuda')):
		super(IMG_NET_COMPLEX,self).__init__()
		self.dropout		= dropout_p
		self.act_fn			= act_fn 
		self.act_kwargs 	= act_kwargs
		self.model 			= nn.Sequential(

			#480x270
			nn.Conv2d(3,nf,5,1,2,bias=False),
			nn.BatchNorm2d(nf),
			self.act_fn(**self.act_kwargs),

			#240x135
			nn.Conv2d(nf,nf*2,5,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			self.act_fn(**self.act_kwargs),

			#120x75
			nn.Conv2d(nf*2,nf*2,5,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),
			#60x37
			nn.Conv2d(nf*2,nf*2,5,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),

			nn.Conv2d(nf*2,nf*4,5,1,1,bias=False),
			nn.BatchNorm2d(nf*4),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),

			nn.Conv2d(nf*4,nf*4,5,1,1,bias=False),
			nn.BatchNorm2d(nf*4),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),

			nn.Conv2d(nf*4,nf*4,5,1,1,bias=False),
			nn.BatchNorm2d(nf*4),
			self.act_fn(**self.act_kwargs),

				
			nn.Flatten(1),

			nn.Linear(192,128),
			nn.Dropout(p=dropout_p),
			self.act_fn(**self.act_kwargs),

			nn.Linear(128,4)
		).to(device)

		self.loss 			= loss_fn()
		self.optimizer		= optimizer_fn(self.model.parameters(),**kwargs)

	def forward(self,x):
		return self.model(x)

class IMG_NET3(nn.Module):

	def __init__(self,input_shape=(3,540,960),nf=16,loss_fn=torch.nn.MSELoss,optimizer_fn=torch.optim.Adam,kwargs={"lr":.0001,"betas":(.9,.999)},dropout_p=.25,neg_slope=.2,device=torch.device('cuda')):
		super(IMG_NET3,self).__init__()
		self.dropout		= dropout_p
		self.act_fn			= nn.Tanh
		self.act_kwargs 	= {}#{"negative_slope":.1}
		self.model 			= nn.Sequential(

			#480x270
			nn.Conv2d(3,nf,3,1,1,bias=False),
			nn.BatchNorm2d(nf),
			self.act_fn(**self.act_kwargs),

			#240x135
			nn.Conv2d(nf,nf*2,3,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			self.act_fn(**self.act_kwargs),

			#120x75
			nn.Conv2d(nf*2,nf*2,5,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			self.act_fn(**self.act_kwargs),
			#nn.MaxPool2d(2),

			#60x37
			nn.Conv2d(nf*2,nf*2,5,1,1,bias=False),
			nn.BatchNorm2d(nf*2),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),
			#30x
			nn.Conv2d(nf*2,nf*4,5,1,1,bias=False),
			nn.BatchNorm2d(nf*4),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),

			nn.Conv2d(nf*4,nf*4,5,1,1,bias=False),
			nn.BatchNorm2d(nf*4),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),

			nn.Conv2d(nf*4,nf*4,5,1,1,bias=False),
			nn.BatchNorm2d(nf*4),
			self.act_fn(**self.act_kwargs),
			nn.MaxPool2d(2),
			

			
				
			nn.Flatten(1),

			nn.Linear(1536,512),
			nn.Dropout(p=dropout_p),
			self.act_fn(**self.act_kwargs),

			nn.Linear(512,128),
			nn.Dropout(p=dropout_p),
			self.act_fn(**self.act_kwargs),

			nn.Linear(128,4)
		).to(device)

		self.loss 			= loss_fn()
		self.optimizer		= optimizer_fn(self.model.parameters(),**kwargs)

	def forward(self,x):
		return self.model(x)

def init_weights(m,conv_av=1.0,conv_var=0.002,bn_av=1.0,bn_var=.002,ln_av=1.0,ln_var=.001):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, conv_av, conv_var)

	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, bn_av, bn_var)
		nn.init.constant_(m.bias.data, 0)

	elif classname.find("Linear") != -1:
		#nn.init.normal_(m.weight.data, ln_av,ln_var)
		#nn.init.normal_(m.bias.data, 0,ln_var)
		pass
		
if __name__ == "__main__":
	inv 	= torch.randn(size=(3,3,540,960),dtype=torch.float,device=torch.device('cuda'))

	model = IMG_NET(960,540,3)

	print(f"{model.forward(inv).shape}")
