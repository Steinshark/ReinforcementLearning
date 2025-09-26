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

        super(SnakeConvNetMd,self).__init__(input_dim,loss_fn,activation_fn)


        #Create model 
        self.conv1 			= torch.nn.Conv2d(input_dim[0],16,3,1,1)
        self.conv2 			= torch.nn.Conv2d(16,32,5,1,1)
        self.conv3 			= torch.nn.Conv2d(32,64,5,1,1)
        self.conv4 			= torch.nn.Conv2d(64,128,5,1,1)
        self.avgpool 		= torch.nn.AdaptiveAvgPool2d(4)
        self.lin1 			= torch.nn.Linear(128*4*4,128)
        self.lin2 			= torch.nn.Linear(128,4)

        
    
    def forward(self,x:torch.Tensor) ->torch.Tensor:

        x 					= self.activation_fn(self.conv1(x))
        x 					= self.activation_fn(self.conv2(x))
        x 					= self.activation_fn(self.conv3(x))
        x 					= self.activation_fn(self.conv4(x))
        x 					= self.avgpool(x)
        x 					= x.view(x.shape[0],-1)
        x 					= self.activation_fn(self.lin1(x))
        x 					= self.lin2(x)

        return 				x 


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

        adapt_1 			= 64

        self.conv_layers    = torch.nn.Sequential(
            
            #Scale image to 64x64
            torch.nn.AdaptiveAvgPool2d((adapt_1,adapt_1)),
            
            #conv layers for 8 ch       
            torch.nn.Conv2d(input_dim[0],8,3,1,1),
            torch.nn.BatchNorm2d(8),
            activation_fn(),

            #Scale image to 32x32
            torch.nn.MaxPool2d(2),

            #conv layers for 16 ch
            torch.nn.Conv2d(8,16,3,1,1),
            torch.nn.BatchNorm2d(16),
            activation_fn(),

            #Scale image to 16x16
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16,32,3,1,1),
            torch.nn.BatchNorm2d(32),
            activation_fn(),

            #Scale image to 8x8
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32,64,3,1,1),
            torch.nn.BatchNorm2d(64),
            activation_fn(),


            #Scale image to 4x4
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64,128,3,1,1),
            torch.nn.BatchNorm2d(128),
            activation_fn(),

            #Scale image to 2x2
            torch.nn.MaxPool2d(2),

            
            torch.nn.Conv2d(128,4,(2,2),1,0),    #Yields 4x1
            torch.nn.Flatten(start_dim=1)
            
            )
            

        
        
    
    def forward(self,x:torch.Tensor) ->torch.Tensor:

        x 					= self.conv_layers(x)
        return x 


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
        #self.avgpool1 		= torch.nn.AdaptiveAvgPool2d(adapt_1)


        self.conv3 			= torch.nn.Conv2d(16,64,5,2,2)
        #self.conv4 			= torch.nn.Conv2d(32,32,5,1,2)
        #self.avgpool2		= torch.nn.AdaptiveAvgPool2d(adapt_2)


        self.conv5 			= torch.nn.Conv2d(64,256,5,2,2)
        #self.conv6 			= torch.nn.Conv2d(64,64,5,1,2)
        #self.avgpool3		= torch.nn.AdaptiveAvgPool2d(adapt_3)

        self.conv7 			= torch.nn.Conv2d(256,512,5,2,2)
        #self.conv8 			= torch.nn.Conv2d(64,64,5,1,2)
        self.avgpool4		= torch.nn.AdaptiveAvgPool2d(adapt_4)


        self.lin1 			= torch.nn.Linear(512*adapt_4*adapt_4,512)
        self.lin2 			= torch.nn.Linear(512,4)

        
    
    def forward(self,x:torch.Tensor) ->torch.Tensor:

        x 					= self.activation_fn(self.conv1(x))
        x 					= self.activation_fn(self.conv2(x))
        x 					= self.avgpool1(x)

        x 					= self.activation_fn(self.conv3(x))
        #x 					= self.activation_fn(self.conv4(x))
        x 					= self.avgpool2(x)

        x 					= self.activation_fn(self.conv5(x))
        #x 					= self.activation_fn(self.conv6(x))
        x 					= self.avgpool3(x)

        x 					= self.activation_fn(self.conv7(x))
        #x 					= self.activation_fn(self.conv6(x))
        x 					= self.avgpool4(x)

        x 					= x.view(x.shape[0],-1)
        x 					= self.activation_fn(self.lin1(x))
        x 					= self.lin2(x)

        return 				x 


if __name__ == '__main__':

    model = SnakeAdaptNet((3,32,32),torch.nn.MSELoss,torch.nn.GELU)

    y = model.forward(torch.randn(8,3,32,32))   
    input(y.shape)