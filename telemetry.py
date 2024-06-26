import random 
import os 
from matplotlib import pyplot as plt
import torch 
from torch.nn import ReLU,MaxPool2d,Conv2d,Linear,Softmax,Flatten, BatchNorm2d, LeakyReLU
from tkinter import BooleanVar
import networks 
#Plot a list of scores and lives from a run of snake trainer

def plot_game(scores_list=[],steps_list=[],series_names="Empty",x_scales=[],graph_name="NoName",f_name="iterations"):

    colors = ["red","green","orange","cyan","black","mediumblue","chocolate","deepskyblue","gold"]
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(19.2,10.8)

    random.shuffle(colors) 
    for i,sc,li,x,na in zip([l for l in range(len(scores_list))],scores_list,steps_list,x_scales,series_names):
        axs[0].plot(x,sc,label=na,color=colors[i])
        axs[1].plot(x,li,label=na,color=colors[i])

    axs[0].legend()
    axs[0].set_title("Average Score")
    axs[1].legend()
    axs[1].set_title("Average Steps")

    axs[0].set_xlabel("Game Number")
    axs[0].set_ylabel("Score")
    axs[1].set_xlabel("Game Number")
    axs[1].set_ylabel("Steps Taken")
    fig.suptitle(graph_name)
    #Save fig to figs directory
    if not os.path.isdir("figs"):
        os.mkdir("figs")

    u = 0
    name = os.path.join("figs",f"{f_name}{u}.png")
    while os.path.exists(name):
        u += 1
        name = os.path.join("figs",f"{f_name}{u}.png")

    fig.savefig(name)
    print(f"saved to {name}")

ARCHITECTURES = {   "conv_sm"   : networks.SnakeConvNetSm,
                    "conv_md"   : networks.SnakeConvNetMd,
                    "conv_lg"   : networks.SnakeConvNetLg,
                    "adaptive"  : networks.SnakeAdaptNet,
                    "mega"      : networks.SnakeMegaNet
}

LOSSES      = { "MSE"       : torch.nn.MSELoss,
                "Huber"     : torch.nn.HuberLoss,
                "L1"        : torch.nn.L1Loss}

OPTIMIZERS  = { "Adam"      : torch.optim.Adam,
                "AdamW"     : torch.optim.AdamW,
                "SGD"       : torch.optim.SGD,
                "RMSProp"   : torch.optim.RMSprop}

ACTIVATIONS = { "ReLU"      : torch.nn.functional.relu, 
                "ReLU6"     : torch.nn.functional.relu6,
                "LeakyReLU" : torch.nn.functional.leaky_relu,
                "Sigmoid"   : torch.nn.functional.sigmoid,
                "Tanh"      : torch.nn.functional.tanh,
                "CELU"      : torch.nn.functional.celu,
                "PReLU"     : torch.nn.functional.prelu,
                "GELU"      : torch.nn.functional.gelu,
                "ELU"       : torch.nn.functional.elu}

DEFAULTS    = { "gameX"     : 10,
                "gameY"     : 10,
                "img_dim"   : (32, 32),
                "iters"     : 8192,
                "te"        : 8,
                "ps"        : 16384,
                "ss"        : 512,
                "bs"        : 128,
                "hs"        : 3,
                "kw"        : "{'lr':.0002}",
                "ep"        : 1,
                "mt"        : .05,
                "mx"        : 100,
                "arch"      : "",
                "lo"        : "",
                "op"        : "",
                "ac"        : "",
                "tr"        : 8,
                "gam"       : .9,
                "gpu"       : False,
                "rew"       : "{'die':-35,'eat':75,'step':-2}",
                "rpick"     : 0,
                "run_name"  : "run1"
                }



POTENTIAL   = { "gameX"     : 14,
                "gameY"     : 14,
                "iters"     : 1024*32,
                "te"        : 8,
                "ps"        : 1024*2,
                "ss"        : 128,
                "bs"        : 16,
                "lr"        : .0001,
                "kw"        : "{'weight_decay':.5e-5}",
                "ll"        : "[(-1,3e-3),(256,5e-5)]",
                "ep"        : 1,
                "mt"        : .03,
                "mx"        : 80,
                "arch"      : "",
                "lo"        : "",
                "op"        : "",
                "tr"        : 12,
                "drop"      : .25,
                "gam"       : .9,
                "gpu"       : True,
                "rew"       : "{'die':-.75,'eat':1,'step':0}",
                "rpick"     : .45
                }


# in_v    = torch.randn(size=(1,2,20,20),dtype=torch.float)

# model = ConvolutionalNetwork(torch.nn.MSELoss,torch.optim.Adam,lr=.0001,architecture=ARCHITECTURES['new']['arch'],input_shape=(1,2,20,20))

# print(f"Model Out: {model.forward(in_v).shape}")