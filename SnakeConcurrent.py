
# All pytorch modules 
import torch
import torch.nn as nn
from torch.nn import Conv2d,Linear,Flatten,ReLU

#All supporting modules 
import random
import numpy 
import time 
from networks import ConvolutionalNetwork, IMG_NET, IMG_NET3
import copy
from matplotlib import pyplot as plt 
import utilities
import torchvision.utils as vutils
import networks 
#This class interfaces only with NP 
class Snake:


	#	CONSTRUCTOR 
	#	This method initializes the snake games to be played until each are over 
	#	i.e. it allows for all 16, 32, etc... games of a batch to be played at once.
	def __init__(self,w,h,target_model:nn.Module,simul_games=32,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),rewards={"die":-1,'eat':1,"step":-.01},max_steps=200,history=3):


		#Set global Vars
		#	grid_w maintains the width of the game field 
		#	grid_h tracks the height of the game field 
		#	simul_games is how many games will be played at once. 
		# 	- essential this is the batch size  
		self.grid_w 			= w
		self.grid_h 			= h
		self.simul_games 		= simul_games
		self.cur_step			= 0
		self.history			= history

		#	GPU or CPU computation are both possible 
		#	Requires pytorch for NVIDIA GPU Computing Toolkit 11.7
		self.device 			= device


		#	Hopefully is a CNN 
		# 	must be a torch.nn Module
		self.target_model 		= target_model
		self.target_model.to(device)


		#	This list holds information on each game, as well as their experience sets.
		#	Since all games are played simultaneously, we must keep track of which ones
		#	should still be played out and which ones are over.
		#
		#	MAINTAINS:
		#		status: 		if the game is over or not 
		#		experiences: 	the set of (s,a,s`,r,done) tuples 
		#		highscore:		how many foods this snake collected
		#		lived_for:		how many steps this snake survived
		self.game_collection 	= [{"status":"active","experiences":[],"highscore":0,"lived_for":0,"eaten_since":0,"step_reward":None} for _ in range(simul_games)]
		self.active_games 		= list(range(simul_games))


		#Encode games as a history x w x h   
		self.game_vectors	    = torch.zeros(size=(simul_games,history,2,w,h),device=device,dtype=torch.float32)

		#	The game directions are tracked in this list. 
		# 	Each tuple encodes the step taken in (x,y).
		#
		# 	Tuples should only ever have one '1' in it. 
		# 	i.e. (1,1) & (0,0) are NOT legal direction tuples
		self.direction_vectors 	= [ random.randint(0,3) for _ in range(simul_games) ]
		self.prev_dir_vectors 	= self.direction_vectors

		#	The food directions are tracked in this list. 
		# 	Each tuple encodes the food at pos (x,y).
		self.food_vectors 		= [ [random.randint(0,w-1),random.randint(0,h-1)] for _ in range(simul_games) ]

		#	The snake is tracked in this list 
		#	Used to update Numpy Arrays in a more efficient manner.
		self.snake_tracker		= [ [[0,0]] for _ in range(simul_games) ]
		self.full_game_tracker 	= [[] for _ in range(simul_games)]

		#	Store all experiences in a list of 
		#	dictionaries that will be returned to the training class
		self.experiences 		= list()
		
		#	A very important hyper-parameter: the reward made for each action
		self.reward 			= rewards 
		self.move_threshold  	= max_steps
		self.movements 			= [(0,-1),(0,1),(-1,0),(1,0)]

		



	#	GAME PLAYER 
	#	Calling this method will play out all games until completion
	def play_out_games(self,epsilon=.2,debugging=False,display_img=False):

		#	Track all imgs 
		if display_img:
			frame_sc		= torch.zeros(size=(64,self.history,self.grid_w,self.grid_h))
		
		#	Maintain some global parameters 
		self.cur_step 		= 0
		self.graph_pending 	= False 

		#	Spawn the snake in a random location each time
		# 	Create the initial state_imgs for snakes 
		for snake_i in range(self.simul_games):
			game_start_x,game_start_y = random.randint(0,self.grid_w-1),random.randint(0,self.grid_h-1)
			self.snake_tracker[snake_i] = [[game_start_x,game_start_y]]
			t0 = time.time()
			self.game_vectors[snake_i]			= utilities.build_snake(self.snake_tracker[snake_i],self.food_vectors[snake_i],arr_w=self.grid_w,arr_h=self.grid_h,history=self.history)







		#	Game Loop executes while at least one game is still running 
		#	Executes one step of the game and does all work accordingly
		while True:			
			#	GET NEXT DIR  
			#	- an epsilon-greedy implementation 
			#	- choose either to exploit or explore based off of 
			#	  some threshold. At first, P(explore) >> P(exploit).
			#	- decides for ALL GAMES simultaneously
			
			# 	The model for this dropoff will probably change and is 
			#	open to exploration

			# if debugging: 			#SHOW IMAGE 
			# 	print(f"GAME No. {self.active_games[-1]}")
			# 	vect 		= self.game_vectors[self.active_games[-1]].detach().cpu().numpy().transpose(1,2,0).astype(numpy.float32)
			# 	plt.imshow(vect.astype(numpy.float32))
			# 	plt.show()

			for snake_i in self.active_games:
				self.full_game_tracker[snake_i].append({"snake":self.snake_tracker[snake_i],'food':self.food_vectors[snake_i]})

			#Save first game 
			if display_img:
				if self.game_collection[0]['status'] == 'active' and self.cur_step < 64:
					frame_sc[self.cur_step]	=self.game_vectors[0]

			if debugging:
				game_i 		= self.active_games[0]
				print(f"game{game_i} -\nsnake:{self.snake_tracker[game_i]}\nfood:{self.food_vectors[game_i]}\n\n")
				print(f"game{game_i}\n{self.game_vectors[game_i]}")
					
			if random.random() < epsilon:
				self.explore()
			else:
				self.exploit()
			
			#print(f"dir:\t{self.movements[self.direction_vectors[game_i]]}")
			
			#	MAKE NEXT MOVES 
			#	Involves querying head of each game, finding where it will end next,
			#	and applying game logic to kill/reward it 
			
			#	Step
			self.step_snake()

			#input(f"ended at\n\n{self.game_vectors[game_i]}")
			 
			# 	Check if we are done 
			if len(self.active_games) == 0:
				#Display frames
				if display_img and self.game_collection[0]['highscore'] > 0 and self.cur_step > 10:
					ex 			= vutils.make_grid(frame_sc.detach().cpu(),padding=1,normalize=True)
					fig,axs 	= plt.subplots(nrows=1,ncols=1)
					axs.axis 	= 'off'
					axs.imshow(numpy.transpose(ex,(1,2,0)))
					img 		= plt.gcf()
					img.set_size_inches(30,16)
					img.savefig("EPOCH SAVE AFTER SCORE",dpi=100)	
					plt.cla()			
					self.saved_img = True
				else:
					self.saved_img = False
				return self.cleanup()
			else:
				self.cur_step+=1
			
	





	#############################################################
	#															#
	#	HELPER FUNCTIONS TO MAKE TRAINING FUNCTION LOOK NICER   #
	#															#
	 
	#	EXPLORE 
	# 	Update all directions to be random.
	#	This includes illegal directions i.e. direction reversal
	def explore(self):
		self.prev_dir_vectors = copy.copy(self.direction_vectors)

		for snake_i in self.active_games:
			cur_dir = self.direction_vectors[snake_i]

			#Give it only legal moves
			if cur_dir in [0,1]:
				self.direction_vectors[snake_i] = random.randint(2,3)
			elif cur_dir == [2,3]:
				self.direction_vectors[snake_i] = random.randint(0,1)
			
			#self.direction_vectors[snake_i] = random.randint(0,3) 
	


	#	EXPLOIT 
	# 	Sends ALL games into model to be predicted (probably faster than sifting (???))
	def exploit(self,mode='Alive'):

		self.prev_dir_vectors = copy.copy(self.direction_vectors)
		#	Inputs are of shape (#Games,#Channels,Height,Width) 
		#	Model output should be of shape (#Games,4)
		#	model_out corresponds to the EV of taking direction i for each game
		
		if mode == 'All':
			with torch.no_grad():
					model_out = self.target_model.forward(self.game_vectors.type(torch.float))
					#Find the direction for each game with highest EV 
					next_dirs = torch.argmax(model_out,dim=1).tolist()

					#Update direction vectors accordingly 
					self.direction_vectors = next_dirs
		
		#TODO BAD IMPLEMENTATION DO ALL ALIVE AT ONCE 
		elif mode == 'Alive':
			with torch.no_grad():
					for snake_i in self.active_games:
						#input(f"passing in shape {self.game_vectors[snake_i].flatten(1,2).shape}")
						model_out 	= self.target_model.forward(self.game_vectors[snake_i].flatten(0,1).type(torch.float).unsqueeze(0))
						next_dir 	= torch.argmax(model_out)
						#print(f"outputs are\t{model_out}\n\nmax was {next_dir.item()}")
						self.direction_vectors[snake_i] = next_dir.item()

	#	STEP SNAKE 
	#	Move each snake in the direction that dir points 
	#	Ensure we only touch active games
	def step_snake(self):

		mark_del = []
		i = self.active_games[0]
		for snake_i in self.active_games:



			# DEBUG 
			#if snake_i == i and  print(f"snake {i} - {self.snake_tracker[i]}\ninit dir {self.movements[self.direction_vectors[i]]}\ninit food {self.food_vectors[i]}\ninit state:\n{self.game_vectors[snake_i]}"): pass
			

			#	Find next location of snake 
			chosen_action 	= self.direction_vectors[snake_i]
			dx,dy 			= self.movements[chosen_action]
			next_x 			= self.snake_tracker[snake_i][0][0]+dx
			next_y 			= self.snake_tracker[snake_i][0][1]+dy
			next_head 		= [next_x,next_y]
			
			#Check if this snake lost 
			if next_x < 0 or next_y < 0 or next_x == self.grid_w or next_y == self.grid_h or next_head in self.snake_tracker[snake_i] or self.game_collection[snake_i]['eaten_since'] > self.move_threshold or self.check_opposite(snake_i):
				
				#Mark for delete and cleanup
				mark_del.append(snake_i)
				self.game_collection[snake_i]['status'] = "done"
				self.game_collection[snake_i]['highscore'] = len(self.snake_tracker[snake_i])-1
				self.game_collection[snake_i]["lived_for"] = self.cur_step

				#Add final experience
				experience = {"s":self.game_vectors[snake_i].flatten(0,1).clone(),"a":chosen_action,"r":self.reward['die'],'s`':self.game_vectors[snake_i].flatten(0,1).clone(),'done':0}
				
				#Dont penalize fully for threshold
				if self.game_collection[snake_i]['eaten_since'] > self.move_threshold:
					experience['r'] = self.reward['step']

				self.experiences.append(experience)
				continue
			
			#	START EXP CREATION 	
			experience = {"s":self.game_vectors[snake_i].flatten(0,1).clone(),"a":chosen_action,"r":None,'s`':None,'done':1}
			
			#	MOVE SNAKE  
			#	Since the snake has survived, dim the previous snake and add over the new one  
			#	the new snake state .
			# self.game_vectors[snake_i] *= .333

			#check to delete 
			#Check if snake ate food
			if next_head == self.food_vectors[snake_i]:
				
				#Change location of the food
				self.spawn_new_food(snake_i)
				
				self.snake_tracker[snake_i].append(self.snake_tracker[snake_i][-1])

				#Set snake reward to be food 
				experience['r'] = self.reward['eat']
				self.game_collection[snake_i]["eaten_since"] = 0
			
			
			else:
				experience['r'] = self.reward["step"]


			#Update the food location vector 
			self.game_collection[snake_i]["eaten_since"] += 1

			#	Update snake tracker with some finicky magic 
			#	If the snake grew, then snake tracker will have been made artificially longer 
			# 	to account for growth 
			self.snake_tracker[snake_i] 			= [next_head] + self.snake_tracker[snake_i][:-1]

			#Update game_state repr 
			self.game_vectors[snake_i]				= utilities.step_snake(self.game_vectors[snake_i],self.snake_tracker[snake_i],self.food_vectors[snake_i],arr_w=self.grid_w,arr_h=self.grid_h)
			#	Add s` to the experience 
			experience['s`'] 						= self.game_vectors[snake_i].flatten(0,1).clone()
			self.experiences.append(experience)
			

		#Delete all items from mark_del  
		for del_snake_i in mark_del:
			self.active_games.remove(del_snake_i)
	

		return 



	#	SPAWN NEW FOOD 
	#	Place a random food on map.
	#	Check that its not in the snake
	#	Repeat until above is True
	def spawn_new_food(self,snake_i):
		next_x = random.randint(0,self.grid_w-1)
		next_y = random.randint(0,self.grid_h-1)
		food_loc = [next_x,next_y]

		while food_loc in self.snake_tracker[snake_i]:
			next_x = random.randint(0,self.grid_w-1)
			next_y = random.randint(0,self.grid_h-1)
			food_loc = [next_x,next_y] 

		self.food_vectors[snake_i] = food_loc 
		return next_x,next_y



	#	SAVE ALL GAME EXPS
	#	Apply game logic to see which snakes died,
	# 	which eat, and which survive 
	def cache_round(self):
		return


	#	RETURN TO TRAINER
	def cleanup(self):
		return self.game_collection,self.experiences,self.full_game_tracker


	def check_opposite(self,snake_i):
		if self.cur_step == 0 or len(self.snake_tracker[snake_i]) == 1:
			return False
		dir_1 = self.direction_vectors[snake_i]
		dir_2 = self.prev_dir_vectors[snake_i]

		return abs(dir_1-dir_2) == 1 and not dir_1+dir_2 == 3
	
	def display_img(self,snake_i):
		img_repr 	= utilities.step_snake(self.game_vectors[snake_i],self.snake_tracker[snake_i],self.food_vectors[snake_i],(self.grid_w,self.grid_h),img_w=self.img_repr_size[0],img_h=self.img_repr_size[1],min_thresh=self.min_thresh)
		plt.imshow(img_repr.numpy(),interpolation="nearest")
		plt.show()

if __name__ == "__main__":
	w = 4
	h = 4
	from networks import *
	dev 	= torch.device('cpu')
	model 	= networks.ConvNet()
	s 		= Snake(w,h,model,simul_games=1,device=dev,history=2)
	s.play_out_games(debugging=False,display_img=False,epsilon=0)
