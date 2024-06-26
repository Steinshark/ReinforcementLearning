
# All pytorch modules 
import torch
import torch.nn as nn

#All supporting modules 
import random
import numpy 
import time 
import copy
from matplotlib import pyplot as plt 
import utilities
import torchvision.utils as vutils

class Snake2:

	def __init__(self,w,h,rewards,max_step,img_dims,device:torch.device):

		self.grid_w 			= w 
		self.grid_h				= h 
		self.img_dims 			= 0 
		self.max_steps 			= max_step
	
		#Spawn snake anywhere in the playing field and give it a random direction
		self.snake 				= [[random.randint(0,w-1),random.randint(0,h-1)]]
		self.direction 			= random.randint(0,3)

		self.static_gpu_tensor 	= torch.empty(size=(3,img_dims[0],img_dims[1]),device=device)

	def generate_representation(self):
		pass

	def step_snake(self):
		pass
		
	def observe_outcome(self):
		pass

#This class interfaces only with NP 
class Snake:


	#	CONSTRUCTOR 
	#	This method initializes the snake games to be played until each are over 
	#	i.e. it allows for all 16, 32, etc... games of a batch to be played at once.
	def __init__(self,w,h,target_model:nn.Module,simul_games=32,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),rewards={"die":-1,'eat':1,"step":-.01},max_steps=200,img_dims=(160,90),min_thresh=.1):

		#Set variables pertaining to game structure
		#	grid_w,gridh define the size of the snake game (in squares)
		#	simul_games is how many games will be played at once. 
		# 	- (essentially this is the inference batch size)
		self.grid_w 			= w
		self.grid_h 			= h
		self.simul_games 		= simul_games
		self.cur_step			= 0
		self.img_dims 			= img_dims
		self.min_thresh			= min_thresh

		#	GPU or CPU computation are both possible 
		#	Requires pytorch for NVIDIA GPU Computing Toolkit 11.7 or 12
		self.device 			= device

		# 	Must be a torch.nn.Module that accepts an input shape (3,img_dims[0],img_dims[1])
  		#   and output a 1x4 tensor that corresponds to the expected Q value of making each of 
		# 	the 4 possible moves from that state	
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


		#Encode games as a 3-channel m x n imgage 
		self.game_vectors	    = torch.zeros(size=(simul_games,3,img_dims[1],img_dims[0]),device=device,dtype=torch.float32)
		self.template_vectors 	= [None] * simul_games
		
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
		self.snake_tracker		= [ [[random.randint(0,h-1),random.randint(0,w-1)]] for _ in range(simul_games) ]
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
	def play_out_games(self,epsilon=.2,debugging=False,display_img=False,train_dtype=torch.float32):

		#	Track all imgs 
		if display_img:
			frame_sc		= torch.zeros(size=(64,3,self.img_dims[1],self.img_dims[0]))
		
		#	Maintain some global parameters 
		self.cur_step 		= 0
		self.graph_pending 	= False 

		#	Setup utils to make the snake frames 
		#utilities.init_utils((self.grid_w,self.grid_h),self.img_dims[0],self.img_dims[1],torch.float32)
		
		#	Spawn the snake in a random location each time
		# 	Create the initial state_imgs for snakes 
		for snake_i in range(self.simul_games):
			game_start_x,game_start_y = random.randint(0,self.grid_w-1),random.randint(0,self.grid_h-1)
			self.snake_tracker[snake_i] = [[game_start_x,game_start_y]]
			t0 = time.time()

			#OLD
			#self.game_vectors[snake_i]			= utilities.build_snake_img(self.snake_tracker[snake_i],self.food_vectors[snake_i],(self.grid_w,self.grid_h),img_w=self.img_dims[0],img_h=self.img_dims[1])

			#NEW 
			self.game_vectors[snake_i],self.template_vectors[snake_i]	= utilities.build_snake_img_sq(self.snake_tracker[snake_i],self.food_vectors[snake_i],(self.grid_w,self.grid_h),img_w=self.img_dims[0],img_h=self.img_dims[1])
			





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

			if debugging: 			#SHOW IMAGE 
				print(f"GAME No. {self.active_games[-1]}")
				vect 		= self.game_vectors[self.active_games[-1]].detach().cpu().numpy().transpose(1,2,0).astype(numpy.float32)
				plt.imshow(vect.astype(numpy.float32))
				plt.show()

			for snake_i in self.active_games:
				self.full_game_tracker[snake_i].append({"snake":self.snake_tracker[snake_i],'food':self.food_vectors[snake_i]})

			#Save first game 
			if display_img:
				if self.game_collection[0]['status'] == 'active' and self.cur_step < 64:
					frame_sc[self.cur_step]	=self.game_vectors[0]
					
			if random.random() < epsilon:
				self.explore()
			else:
				self.exploit(train_dtype=train_dtype)
			
			#	MAKE NEXT MOVES 
			#	Involves querying head of each game, finding where it will end next,
			#	and applying game logic to kill/reward it 
			
			#	Step
			self.step_snake(train_dtype=train_dtype)
			 
			# 	Check if we are done 
			if len(self.active_games) == 0:
				#Display frames
				
				if display_img and self.game_collection[0]['highscore'] > 0 and self.cur_step > 4:
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

			#Any move is a go
			self.direction_vectors[snake_i] = random.randint(0,3)
				

	#	EXPLOIT 
	# 	Use the model to choose moves
	def exploit(self,mode='Alive',train_dtype=torch.float32):

		self.prev_dir_vectors = copy.copy(self.direction_vectors)	
		#	Inputs are of shape (#Games,#Channels,Height,Width) 
		#	Model output should be of shape (#Games,4)
		#	model_out corresponds to the EV of taking direction i for each game
		
		if mode == 'All':
			with torch.no_grad():
				with torch.autocast('cuda'):
					model_out = self.target_model.forward(self.game_vectors.type(train_dtype))
					#Find the direction for each game with highest EV 
					next_dirs = torch.argmax(model_out,dim=1).tolist()

					#Update direction vectors accordingly 
					self.direction_vectors = next_dirs
		
		#TODO BAD IMPLEMENTATION DO ALL ALIVE AT ONCE 
		elif mode == 'Alive':
			with torch.no_grad():
				live_games 	= torch.stack([self.game_vectors[i] for i in self.active_games]).type(train_dtype)
				model_out	= self.target_model.forward(live_games) 
				next_dirs 	= torch.argmax(model_out,dim=1)

				for snake_i,best_dir in zip(self.active_games,next_dirs):
					self.direction_vectors[snake_i] = best_dir


	#	STEP SNAKE 
	#	Move each snake in the direction that dir points 
	#	Ensure we only touch active games
	def step_snake(self,train_dtype=torch.float32):

		mark_del = []
		i = self.active_games[0]
		for snake_i in self.active_games:



			# DEBUG 
			#if snake_i == i and  print(f"snake {i} - {self.snake_tracker[i]}\ninit dir {self.movements[self.direction_vectors[i]]}\ninit food {self.food_vectors[i]}\ninit state:\n{self.game_vectors[snake_i]}"): pass
			

			#	Find next location of snake 
			chosen_action = self.direction_vectors[snake_i]
			dx,dy = self.movements[chosen_action]
			next_x = self.snake_tracker[snake_i][0][0]+dx
			next_y = self.snake_tracker[snake_i][0][1]+dy
			next_head = [next_x,next_y]
			
			#Check if this snake lost 
			if next_x < 0 or next_y < 0 or next_x == self.grid_w or next_y == self.grid_h or next_head in self.snake_tracker[snake_i] or self.game_collection[snake_i]['eaten_since'] > self.move_threshold or self.check_opposite(snake_i):
				
				#Mark for delete and cleanup
				mark_del.append(snake_i)
				self.game_collection[snake_i]['status'] = "done"
				self.game_collection[snake_i]['highscore'] = len(self.snake_tracker[snake_i])-1
				self.game_collection[snake_i]["lived_for"] = self.cur_step

				#Add final experience
				experience = {"s":self.game_vectors[snake_i,:,:].clone(),"a":chosen_action,"r":self.reward['die'],'s`':self.game_vectors[snake_i,:,:].clone(),'done':0}
				
				#Dont penalize fully for threshold
				if self.game_collection[snake_i]['eaten_since'] > self.move_threshold:
					experience['r'] = self.reward['step']

				self.experiences.append(experience)
				continue
			
			#	START EXP CREATION 	
			experience = {"s":self.game_vectors[snake_i,:,:].clone(),"a":chosen_action,"r":None,'s`':None,'done':1}
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
			self.game_vectors[snake_i],self.template_vectors[snake_i]				= utilities.step_snake_img_sq(self.template_vectors[snake_i],self.snake_tracker[snake_i],self.food_vectors[snake_i],(self.grid_w,self.grid_h),img_w=self.img_dims[0],img_h=self.img_dims[1],min_thresh=self.min_thresh)
			#	Add s` to the experience 
			experience['s`'] 						= self.game_vectors[snake_i,:,:].clone()
			self.experiences.append(experience)
			#input(f"added exp s: {experience['s'].shape}\t{experience['s`'].shape}")

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
		#print(f"returning experiences {self.experiences[0]}")
		return self.game_collection,self.experiences,self.full_game_tracker


	def check_opposite(self,snake_i):
		if self.cur_step == 0 or len(self.snake_tracker[snake_i]) == 1:
			return False
		dir_1 = self.direction_vectors[snake_i]
		dir_2 = self.prev_dir_vectors[snake_i]

		return abs(dir_1-dir_2) == 1 and not dir_1+dir_2 == 3
	

	def display_img(self,snake_i):
		img_repr 	= utilities.step_snake_img(self.game_vectors[snake_i],self.snake_tracker[snake_i],self.food_vectors[snake_i],(self.grid_w,self.grid_h),img_w=self.img_dims[0],img_h=self.img_dims[1],min_thresh=self.min_thresh)
		plt.imshow(img_repr.numpy(),interpolation="nearest")
		plt.show()

if __name__ == "__main__":
	w = 20
	h = 20
	from networks import *
	dev 	= torch.device('cpu')
	model 	= SnakeAdaptNet((3,10,10),torch.nn.MSELoss,torch.nn.functional.relu)
	s 		= Snake(w,h,model,simul_games=1,device=dev,img_dims=(100,100))
	s.saved_img = False 
	s.game_vectors[0],s.template_vectors[0] = utilities.build_snake_img_sq(s.snake_tracker[0],s.food_vectors[0],(20,20),100,100)
	for _ in range(5):
		s.step_snake()
		s.explore()
	
	b = s.game_vectors[0].permute(1,2,0)
	plt.imshow(b.numpy())
	plt.show()
