# ReinforcementLearning

This is my reinforcement learning personal project training to play Snake. Files are as follows:

SnakeConcurrent.py - defines the snake game and allows for multiple to be played at once for greater efficiency. One of the methods returns a tuple of game experiences and some game statistics for that run.

Trainer.py - incorporates SnakeConcurrent to play, store, and train the NN on the games it plays. Contains the full functionality for the self-training algorithm. 

GameViewer.py - a GUI I created to allow for testing of the various hyperparameters, models, rewards, etc... has some nice statistic displays and can watch the snake play games. 


Dependencies:
this is not an exhaustive list, just some that I remember...

- ttkthemes
- pygame 
- pytorch (Cuda not necessary, but speeds up alot. also some code may have hard coded in ".cuda()" (sorry!)
- matplotlib
- numpy
- pygame 


Algorithm at a 10_000ft view: 

- play snake games balancing exploration (random moves) and exploitation (current models best guess) 
- record all board states (a 3 channel image), move choices, outcomes (+reward if move led immediately to eating the food, -reward if death, either 0 or slight -reward for plain step) 
- after playing 'train_every' games, take the detached of experiences, add it to a pool, sample randomly from it, and training the NN to accurately predict the reward via the Bellman Eqn. 
- tweak and run at your pleasure!

enjoy!    