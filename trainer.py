import networks 
import torch 
import os 
import time 
import SnakeConcurrentIMG
import random 
from matplotlib import pyplot as plt 
import numpy 
import sys 
import tkinter as tk
from telemetry import plot_game
import copy
from utilities import reduce_arr

class Trainer:

	def __init__(	self,
			  		game_w,
			  		game_h,
					model_class:networks.SnakeNetwork,
					input_dim,
					loss_fn			= torch.nn.MSELoss,
					optimizer_fn	= torch.optim.Adam,
					optimizer_kwargs={'lr':.0001},
					activation		= torch.nn.functional.relu,

					#Training Vars
					visible			= True,
					gamma			= .96,
					epsilon			= .2,
					
					#Telemetry Vars 
					steps			= None,
					scored			= None,
					output			= sys.stdout,
					score_tracker	= [],
					step_tracker	= [],
					game_tracker	= [],
					progress_var	= None,

					fname="experiences",
					name="generic",
					save_fig_now=False,
					instance=None,
					lr_threshs=[],
					gui=False):


		

		#Set file handling vars 
		self.PATH 				= "C:/Users/Default/temp/models"
		self.fname 				= fname
		self.name 				= name
		self.save_fig 			= save_fig_now

		#Set model vars  
		self.input_dim 			= input_dim
		self.progress_var 		= progress_var
		self.movement_repr_tuples = [(0,-1),(0,1),(-1,0),(1,0)]
		self.loss_fn 			= loss_fn
		self.activation 		= activation

		#Set runtime vars 
		self.cancelled 			= False
		self.w 					= game_w	
		self.h 					= game_h
		self.visible 			= visible

		#Set telemetry vars 
		self.steps_out 			= steps
		self.score_out			= scored
		self.best_score 		= 0 
		self.all_scores 		= score_tracker
		self.all_lived 			= step_tracker
		self.output 			= output
		self.parent_instance 	= instance
		self.game_tracker 		= game_tracker
		self.gui 				= gui
		self.base_threshs		= [(-1,.00003),(1024+256,.00001),(1024+512+256,3e-6),(2048,1e-6),(4096,5e-7),(4096+2048,2.5e-7),(8192,1e-7),(8192*2,1e-8)] if not lr_threshs else lr_threshs
		self.errors 			= [0,0,0,0,0] 
		
		#Set training vars 
		self.gamma 				= gamma
		self.epsilon 			= epsilon
		self.e_0 				= self.epsilon

		#Enable cuda acceleration if specified 
		self.device 			= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




		#Generate models for the learner agent 
		self.model_class 		= model_class

		self.learning_model 	= self.model_class(self.input_dim,self.loss_fn,self.activation)
		self.target_model		= self.model_class(self.input_dim,self.loss_fn,self.activation)
		self.optimizer_fn 		= optimizer_fn(self.learning_model.parameters(),**optimizer_kwargs)
		print(f"Training with arch:\n{self.learning_model}")
	
		self.target_model.to(self.device)
		self.learning_model.to(self.device)
		self.output.insert(tk.END,f"\tTraing with {self.device}\n")



		#print(self.learning_model)
		#REMOVE 
		

	def train_concurrent(self,iters=1000,train_every=1024,pool_size=32768,sample_size=128,batch_size=32,epochs=1,transfer_models_every=2,verbose=False,rewards={"die":-3,"eat":5,"step":-.01},max_steps=100,random_pick=True,drop_rate=0):
		
		#	Sliding window memory update 
		#	Instead of copying a new memory_pool list 
		#	upon overflow, simply replace the next window each time 

		print(f"\tStart training with {random_pick}")
		memory_pool 	= []
		window_i 		= 0
		threshs 		= copy.deepcopy(self.base_threshs)
		stop_thresh  	= False 
		#	Keep track of models progress throughout the training 
		best_score 		= 0 
		self.error 		= 0.0

		#	Train 
		i = 0 
		while i < iters and not self.cancelled:
			#	Keep some performance variables 
			t0 				= time.time() 

			#	UPDATE EPSILON
			e 				= self.update_epsilon(i/(iters))	

			if self.gui:
				self.progress_var.set(i/iters)


			#	GET EXPERIENCES
			metrics, experiences, new_games = SnakeConcurrentIMG.Snake(self.w,self.h,self.learning_model,simul_games=train_every,device=self.device,rewards=rewards,max_steps=max_steps).play_out_games(epsilon=e)



			#	UPDATE MEMORY POOL 
			#	replace every element of overflow with the next 
			# 	exp instead of copying the list every time 
			#	Is more efficient when history >> len(experiences)
			for exp in experiences:
				if window_i < pool_size:
					memory_pool.append(None)
				memory_pool[window_i%pool_size] = exp 
				window_i += 1

			#	UPDATE METRICS
			#	Add average metrics to telemetry 
			self.all_scores.append(sum([game['highscore'] for game in metrics])/len(metrics))
			self.all_lived.append(sum([game['lived_for'] for game in metrics])/len(metrics))


			#	Find best game
			scores 				= [game['highscore'] for game in metrics]
			round_top_score 	= max(scores)
			round_top_game 		= new_games[scores.index(round_top_score)]
			self.game_tracker.append(round_top_game)


			#	Update local and possibly gui games
			if round_top_score >= self.best_score:
				self.best_score 	= round_top_score

				if self.gui:
					self.parent_instance.best_game 	= copy.deepcopy(round_top_game)
					if round_top_score > self.parent_instance.best_score:
						self.parent_instance.best_score	= round_top_score
						self.output.insert(tk.END,f"\tnew hs: {self.best_score}\n")
			

			#	UPDATE VERBOSE 
			if verbose:
				print(f"[Episode {str(i).rjust(15)}/{int(iters)} -  {(100*i/iters):.2f}% complete\t{(time.time()-t0):.2f}s\te: {e:.2f}\thigh_score: {self.best_score}\t] lived_avg: {(sum(self.all_lived[-100:])/len(self.all_lived[-100:])):.2f} score_avg: {(sum(self.all_scores[-100:])/len(self.all_scores[-100:])):.2f}")
			

			#	UPDATE GUI
			if self.gui:
				self.parent_instance.var_step.set(f"{(sum(self.all_lived[-100:])/100):.2f}")
				self.parent_instance.var_score.set(f"{(sum(self.all_scores[-100:])/100):.2f}")
				self.parent_instance.var_error.set(f"{sum(self.errors)/len(self.errors):.2f}")
			
	
			# 	GET TRAINING SAMPLES
			#	AND TRAIN MODEL 
			if window_i > sample_size:
				
				#PICK RANDOMLY 
				if random_pick:
					training_set 	= random.sample(memory_pool,sample_size) 
				
				#PICK SELECTIVELY
				else:
					training_set 	= []
					training_ind	= []

					while len(training_set) < sample_size: 

						cur_i = random.randint(0,len(memory_pool)-1)						#Pick new random index 
						while cur_i in training_ind:
							cur_i = random.randint(0,len(memory_pool)-1)

						#Drop non-scoring experiences with odds: 'drop_rate'
						is_non_scoring 				= memory_pool[cur_i]['r'] == rewards['step']
						if is_non_scoring and random.random() < drop_rate:
							continue
								
						else:
							training_set.append(memory_pool[cur_i])
							training_ind.append(cur_i)

				qual 		= 100*sum([int(t['r'] == rewards['die'] or t['r'] == rewards['eat']) for t in training_set]) / len(training_set)
				bad_set 	= random.sample(memory_pool,sample_size)
				bad_qual 	= f"{100*sum([int(t['r'] == rewards['die'] or t['r'] == rewards['eat']) for t in training_set]) / len(memory_pool):.2f}"

				perc_str 	= f"{qual:.2f}%/{bad_qual}%".rjust(15)

				if verbose:# or True:
					print(f"[Quality\t{perc_str}  -  R_PICK: {'off' if not random_pick else 'on'}\t\t\t\t\t\t]\n")
				self.train_on_experiences(training_set,epochs=epochs,batch_size=batch_size,early_stopping=False,verbose=verbose)

				if self.gui and self.parent_instance.cancel_var:
					self.output.insert(tk.END,f"CANCELLING\n")
					return 
			
			#	UPDATE MODELS 
			if i/train_every % transfer_models_every == 0:
				self.transfer_models(verbose=verbose)
			
			i += train_every

			if self.gui:
				self.parent_instance.training_epoch_finished = True
		return self.cleanup()


	def cleanup(self):
		blocked_scores		= reduce_arr(self.all_scores,self.x_scale)
		blocked_lived 		= reduce_arr(self.all_lived,self.x_scale)
		graph_name = f"{self.name}_[{str(self.loss_fn).split('.')[-1][:-2]},{str(self.optimizer_fn).split('.')[-1][:-2]}]]]"

		if self.save_fig:
			plot_game(blocked_scores,blocked_lived,graph_name)

		if self.gui:
			self.output.insert(tk.END,f"Completed Training\n\tHighScore:{self.best_score}\n\tSteps:{sum(self.all_lived[-1000:])/1000}")
		return blocked_scores,blocked_lived,self.best_score,graph_name


	def train_on_experiences(self,big_set,epochs=1,batch_size=8,early_stopping=True,verbose=False):
		
		num_batches = int(len(big_set) / batch_size)

		#Telemetry 
		if verbose:
			print(f"TRAINING:")
			print(f"\tDataset:\n\t\t{'loss-fn'.ljust(12)}: {str(self.learning_model.loss).split('(')[0]}\n\t\t{'optimizer'.ljust(12)}: {str(self.learning_model.optimizer).split('(')[0]}\n\t\t{'size'.ljust(12)}: {len(big_set)}\n\t\t{'lr'.ljust(12)}: {self.learning_model.optimizer.param_groups[0]['lr']:.8f}\n\t\t{'batches'.ljust(12)}: {num_batches}")

		for epoch_i in range(epochs):
			
			if self.gui and self.parent_instance.cancel_var:
				return
			
			#	Telemetry Vars 
			t0 			= time.time()
			t_gpu 		= 0
			num_equals 	= 40 
			printed 	= 0
			total_loss	= 0


			#	Telemetry
			if verbose:
				print(f"\tEPOCH: {epoch_i}\tPROGRESS- [",end='')


			# Iterate through batches
			for batch_i in range(num_batches):

				i_start 					= batch_i * batch_size
				i_end   					= i_start + batch_size
				

				#	Telemetry
				percent = batch_i / num_batches
				if verbose:
					while (printed / num_equals) < percent:
						print("=",end='',flush=True)
						printed+=1
				

				#BELLMAN UPDATE 
				for param in self.learning_model.parameters():
					param.grad 	= None


				#Gather batch experiences
				batch_set 							= big_set[i_start:i_end]
				init_states 						= torch.stack([exp['s']  for exp in batch_set]).type(torch.float)
				action 								= [exp['a'] for exp in batch_set]
				next_states							= torch.stack([exp['s`'] for exp in batch_set]).type(torch.float)
				rewards 							= [exp['r']  for exp in batch_set]
				done								= [exp['done'] for exp in batch_set]
				
				#Calc final targets 
				initial_target_predictions 			= self.learning_model.forward(init_states)
				final_target_values 				= initial_target_predictions.clone().detach()
				
				#Get max from s`
				with torch.no_grad():
					stepped_target_predictions 		= self.target_model.forward(next_states)
					best_predictions 				= torch.max(stepped_target_predictions,dim=1)[0]
					#print(f"presented with\n{stepped_target_predictions}\n\npicked\n{best_predictions}")

				#Update init values 
				for i,val in enumerate(best_predictions):
					chosen_action						= action[i]
					final_target_values[i,chosen_action]= rewards[i] + (done[i] * self.gamma * val)
					if rewards[i] > .5 and random.random() < .01:
						print(f"maxs {best_predictions}")
						print(f"\nfor init val:{initial_target_predictions[i].cpu().detach().numpy()} + a:{chosen_action} - > update to {rewards[i]:.3f} + {self.gamma:.3f}*{val:.3f}*[done:{done[i]:.3f}] = {rewards[i] + (done[i] * self.gamma * val):.3f}")
						print(f"training with {final_target_values[i].cpu().detach().numpy()}\n\n")
				#	Calculate Loss
				t1 							= time.time()
				batch_loss 					= self.learning_model.loss(initial_target_predictions,final_target_values)
				total_loss 					+= batch_loss.mean().item()

				#Back Propogate
				batch_loss.backward()
				self.learning_model.optimizer.step()
				#print(f"optimizing with loss: {self.learning_model.loss} and optim {self.learning_model.optimizer}\n\n\n")
				t_gpu += time.time() - t1
			
			#	Telemetry
			if verbose :
				print(f"]\ttime: {(time.time()-t0):.2f}s\tt_gpu:{(t_gpu):.2f}\tloss: {(total_loss/num_batches):.6f}")
		if verbose:
			print("\n\n")
		self.error 		= total_loss/num_batches
		self.errors 	= self.errors[1:] + [self.error]


	def transfer_models(self,verbose=False,optimize=False):
		self.output.insert(tk.END,f"\tTransferring Model\n")
		if verbose:
			print("\ntransferring models\n\n")

		#Save the models
		if not os.path.isdir(self.PATH):
			os.mkdir(self.PATH)
		torch.save(self.learning_model.state_dict(),os.path.join(self.PATH,f"{self.fname}_lm_state_dict"))
		
		#Load the learning model as the target model
		self.target_model		= self.model_class(self.input_dim,self.optimizer_fn,self.loss_fn,self.activation)
		self.target_model.load_state_dict(torch.load(os.path.join(self.PATH,f"{self.fname}_lm_state_dict")))
		self.target_model.to(self.device)

		self.target_model.eval()


	def load_prev_models(self):
		
		self.target_model.load_state_dict(torch.load(os.path.join(self.PATH,f"{self.fname}_lm_state_dict")))
		self.learning_model.load_state_dict(torch.load(os.path.join(self.PATH,f"{self.fname}_lm_state_dict")))
		
		self.target_model.to(self.device)
		self.learning_model.to(self.device)

	@staticmethod
	def update_epsilon(percent):
		radical = -.4299573*100*percent -1.2116290 
		if percent > .50:
			return 0
		else:
			return pow(2.7182,radical)
	

if __name__ == "__main__":

	t = Trainer(10,10,True,False,"exps",history=2,gamma=.75,epsilon=.5)
	t.train_concurrent(1024*128,train_every=32,pool_size=1024*16,sample_size=2048,batch_size=32,epochs=1,transfer_models_every=4,verbose=True)