
"""

"""
import sys
import os
from os import path, makedirs

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .RL_trainer.env_ import KnapsackAssignmentEnv
from .RL_trainer.sac import SAC

def save_model(model, save_path):
    if not path.exists(save_path):
        makedirs(save_path)
    file_path = save_path + "/" + model.name + ".ckpt"
    torch.save({
        'model_state_dict': model.state_dict()}, 
        file_path)

def load_model(model, save_path):
	file_path = save_path + "/" + model.name + ".ckpt"
	
	if os.path.exists(file_path):
		print("Loading pre-trained model: ")
		checkpoint = torch.load(file_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		
	else:
		print("Creating new model: "+model.name)
	return model

def plot_learning_curve(x, scores, figure_file, title, label):#TODO delete method
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg, 'C0', linewidth = 1, alpha = 0.5, label=label)
    plt.plot(np.convolve(running_avg, np.ones((3000,))/3000, mode='valid'), 'C0')
    plt.title(title)
    plt.savefig(figure_file)
    plt.show()
    
def sac_train(model, statePrepares, save_path):
    env = KnapsackAssignmentEnv()
    trainer = SAC(model)
    
    #stateprepares = np.array(stateprepares)
    best_score = 0
    score_history = []
    n_steps = 0
    for i in tqdm(range(100000)):
        for statePrepare in statePrepares:
            env.setStatePrepare(statePrepare)
            
            observation, _ = env.reset()
            done = False
            while not done:
                #print(observation.size())
                observation.requires_grad = True#self.
                action = trainer.step_act(observation)
                observation_, reward, done, info = env.step(observation, action, trainer.actor_model)
                observation.requires_grad = False 

                trainer.save_step(observation, action, reward, observation_, done)
                trainer.train()
                observation = observation_
            score = env.score   
            score_history.append(score)
            avg_score = np.mean(score_history[-50:])
            
            
            if avg_score > best_score:
                best_score = avg_score
                save_model(trainer.actor_model, save_path)
                save_model(trainer.critic_model1, save_path)
                save_model(trainer.critic_model2, save_path)
                save_model(trainer.value_model1, save_path)
                save_model(trainer.value_model2, save_path)


            print('episode', i, 'score %.3f' % score, 'avg score %.2f' % avg_score,
                  'time_steps', n_steps)
    x = [i+1 for i in range(len(score_history))]
    figure_file = 'plots/fraction_sac_score_per_greedyScore.png'
    title = 'Running average of previous 50 scores'
    plot_learning_curve(x, score_history, figure_file, title)#TODO add visualization
    
    







