import random 
import math 
import time 
import torch 
import numpy 
from matplotlib import pyplot as plt 
import torchvision.utils as vutils
TIME_MULT   = 0 
VECT_TYPE   = torch.float32
SNAKE_SQ    = torch.ones(size=(1,1))
HEAD_SQ    = torch.ones(size=(1,1))
FOOD_SQ     = torch.ones(size=(1,1))
TOP_L       = (0,0)
BOT_R       = (0,0)
SQUARE_SF   = 0 
DEV         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLORS      = {"snake":(0/255,255/255,0/255),"food":(255/255,0/255,0/255),'head':(0/255,255/255,255/255)}



def tensor_to_img(tensor):
    return tensor.detach().cpu().numpy().transpose(1,2,0).astype(numpy.float32)

def init_utils(board_size,img_w,img_h,vect_type,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    global TOP_L,BOT_R,FOOD_SQ,SNAKE_SQ,HEAD_SQ,VECT_TYPE,SQUARE_SF,DEV

    w                           = board_size[0]
    h                           = board_size[1]
    board_ar                    = w / h 
    DEV                         = device
    VECT_TYPE                   = vect_type

    if board_ar > img_w/img_h:                                     #If too wide, scale to sides 
        SQUARE_SF                   = int(img_w / w)
        TOP_L                       = (0, int((img_h - h*SQUARE_SF) / 2))
        BOT_R                       = (img_w,int(img_h - (img_h - h*SQUARE_SF) / 2))

    else:  
        SQUARE_SF                   = int(img_h / h) 
        TOP_L                       = (int((img_w - w*SQUARE_SF) / 2), 0)
        BOT_R                       = (img_w - int((img_w - w*SQUARE_SF) / 2), img_h)

    SNAKE_SQ                        = torch.zeros(size=(3,SQUARE_SF,SQUARE_SF))
    SNAKE_SQ[0]                     = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=COLORS['snake'][0])                 
    SNAKE_SQ[1]                     = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=COLORS['snake'][1])
    SNAKE_SQ[2]                     = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=COLORS['snake'][2])

    HEAD_SQ                         = torch.zeros(size=(3,SQUARE_SF,SQUARE_SF))
    HEAD_SQ[0]                     = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=COLORS['head'][0])                 
    HEAD_SQ[1]                     = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=COLORS['head'][1])
    HEAD_SQ[2]                     = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=COLORS['head'][2])

    FOOD_SQ                         = torch.zeros(size=(3,SQUARE_SF,SQUARE_SF))
    FOOD_SQ[0]                      = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=COLORS['food'][0])                 
    FOOD_SQ[1]                      = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=COLORS['food'][1])
    FOOD_SQ[2]                      = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=COLORS['food'][2])

def build_snake_img(snake_list,food_loc,board_size,img_w=1280,img_h=720):

    vect_init_type              = torch.float32
    #create base tensor in CxWxH 
    w                           = board_size[0]
    h                           = board_size[1]

    frame_repr_tensor           = torch.zeros(size=(3,img_h,img_w),dtype=vect_init_type)

    #Snake head
    sq = snake_list[0]
    sq_topl_x                       = int(TOP_L[0] + sq[0]*SQUARE_SF) 
    sq_topl_y                       = int(TOP_L[1] + sq[1]*SQUARE_SF) 
    frame_repr_tensor[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]        = HEAD_SQ

    for sq in snake_list[1:]:
        sq_topl_x                       = int(TOP_L[0] + sq[0]*SQUARE_SF) 
        sq_topl_y                       = int(TOP_L[1] + sq[1]*SQUARE_SF) 
        

        sq_topl_x                       = int(TOP_L[0] + sq[0]*SQUARE_SF) 
        sq_topl_y                       = int(TOP_L[1] + sq[1]*SQUARE_SF) 
        frame_repr_tensor[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]        = SNAKE_SQ
    
    sq_topl_x                    = int(TOP_L[0] + food_loc[0]*SQUARE_SF) 
    sq_topl_y                    = int(TOP_L[1] + food_loc[1]*SQUARE_SF) 

    frame_repr_tensor[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]            = FOOD_SQ

    frame_repr_tensor[:,    0:img_h,    0:TOP_L[0]]                                             = torch.ones(size=(3,img_h,img_w-BOT_R[0]))
    frame_repr_tensor[:,    0:img_h,    BOT_R[0]:img_w]                                         = torch.ones(size=(3,img_h,img_w-BOT_R[0]))
    
    return frame_repr_tensor.to(DEV)

def build_snake(snake_list,food_loc,arr_w,arr_h,history):
    repr        = numpy.zeros(shape=(history,2,arr_w,arr_h))

    for seg in snake_list:
        x,y                                     = seg[0],seg[1] 
        repr[0][0][y][x]                        = 1
    
    repr[0][1][food_loc[1],food_loc[0]]     = 1 
    
    return torch.from_numpy(repr).to(DEV).float()

def step_snake_img(game_vector:torch.Tensor,snake_list,food_loc,board_size,img_w=1280,img_h=720,dim_fact=.33,vect_init_type=torch.float32,min_thresh=.03,display_imgs=False):    
    global TIME_MULT,SNAKE_SQ,FOOD_SQ,TOP_L,BOT_R
    w                           = board_size[0]
    h                           = board_size[1]

    MIN     = torch.nn.Threshold(min_thresh,0,True) 

    #Dim playable surface and set threshold of 5 for pixels
    game_vector[:,TOP_L[1]:BOT_R[1],TOP_L[0]:BOT_R[0]] *= dim_fact 
    game_vector = MIN(game_vector)
    
    t0 = time.time()
    #Snake head
    sq = snake_list[0]
    sq_topl_x                       = int(TOP_L[0] + sq[0]*SQUARE_SF) 
    sq_topl_y                       = int(TOP_L[1] + sq[1]*SQUARE_SF) 
    game_vector[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]      = HEAD_SQ
    #Snake body
    for sq in snake_list[1:]:
        sq_topl_x                       = int(TOP_L[0] + sq[0]*SQUARE_SF) 
        sq_topl_y                       = int(TOP_L[1] + sq[1]*SQUARE_SF) 
        game_vector[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]      = SNAKE_SQ
    #Food
    TIME_MULT += time.time()-t0
    sq_topl_x                    = int(TOP_L[0] + food_loc[0]*SQUARE_SF) 
    sq_topl_y                    = int(TOP_L[1] + food_loc[1]*SQUARE_SF) 

    game_vector[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]          = FOOD_SQ


    return game_vector

def step_snake(game_vector,snake_list,food_loc,arr_w,arr_h):
    #input(f"og vect: {game_vector.shape}")
    #reduce 1 dimension of history 
    game_vector     = game_vector[:-1,:,:,:]
    #print(f"game vect slice is {game_vector.shape}")
    new_vector     = numpy.zeros(shape=(1,2,arr_w,arr_h))

    for seg in snake_list:
        x,y                     = seg[0],seg[1]
        new_vector[0,0,y,x]   = 1

    new_vector[0,1,food_loc[1],food_loc[0]]       = 1  

    final_snake     = torch.cat([torch.from_numpy(new_vector).to(DEV),game_vector])
    #input(f"snake dim: {final_snake.shape}")
    return final_snake
    
def reduce_arr(arr,newlen):

    #Find GCF of len(arr) and len(newlen)
    gcf         = math.gcd(len(arr),newlen)
    mult_fact   = int(newlen / gcf) 
    div_fact    = int(len(arr) / gcf) 

    new_arr     = numpy.repeat(arr,mult_fact)


    return [sum(list(new_arr[n*div_fact:(n+1)*div_fact]))/div_fact for n in range(newlen)]





if __name__ == "__main__":
    pass