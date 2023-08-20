
"""

"""
import torch
import os

def save_model(model, save_path):
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

def sac_train(model):
    pass