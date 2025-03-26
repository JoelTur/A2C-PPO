import torch
from torch import distributions, nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import math
from collections import deque
import skimage
import ACTORCRITIC
import Transition
from config_loader import get_config
from wandb import wandb

# Load configuration
config = get_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def getFrame(x):
    x = x[config.environment.frame_crop['top']:config.environment.frame_crop['bottom'],
          config.environment.frame_crop['left']:config.environment.frame_crop['right']]
    state = skimage.color.rgb2gray(x)
    state = skimage.transform.resize(state, config.environment.input_size)
    state = skimage.exposure.rescale_intensity(state, out_range=(0,255))
    state = state.astype('uint8')
    return state

def makeState(state):
    return np.stack((state[0],state[1],state[2],state[3]), axis=0)

def saveModel(agent, filename):
    torch.save(agent.state_dict(), filename)
    print("Model saved!")

def loadModel(agent, filename):
    agent.load_state_dict(torch.load(filename))
    print("Model loaded!")

def predict(agent, state, action_space_size):
    with torch.no_grad():
        state = np.expand_dims(state, axis=0)
        logprob, values = agent(torch.from_numpy(state).float())
        values = torch.squeeze(values).float()
        prob = torch.exp(logprob)
        prob = prob.cpu().detach().numpy()
        prob = np.squeeze(prob)
        return np.argmax(prob), values, logprob

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test():
    env = gym.make(config.environment.name, render_mode="human")
    action_space_size = env.action_space.n
    state = deque(maxlen=4)
    
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb['project'],
            entity=config.logging.wandb['entity']
        )
    
    # Actors in the simulation
    actor_agent = ACTORCRITIC.NeuralNetwork(action_space_size).to(device)
    # Optimization stuff
    optimizer = optim.Adam(actor_agent.parameters(), lr=config.training.learning_rate)
    # Transition class
    transition = Transition.Transition(action_space_size)

    loadModel(actor_agent, config.paths.model_save.format(game=config.environment.name))
    
    total_time = 0
    batch_steps = 0

    # Updated for new Gymnasium API
    observation, info = env.reset()
    state.append(getFrame(observation))
    state.append(getFrame(observation))
    state.append(getFrame(observation))
    state.append(getFrame(observation))
    gamereward = 0
    games_played = 0
    update = 0
    
    while update < config.training.max_updates:
        action, reward_estimate, distribution = predict(actor_agent, makeState(state)/255, action_space_size)
        # Updated for new Gymnasium API
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        transition.addTransition(makeState(state), max(min(reward, 1), -1), action, reward_estimate, distribution)
        state.append(getFrame(observation))
        total_time += 1
        batch_steps += 1
        gamereward += reward
        env.render()
        
        if done:
            print("Running reward: ", gamereward)
            if config.logging.use_wandb:
                wandb.log({"RUNNING REWARD": gamereward})
            gamereward = 0
            # Updated for new Gymnasium API
            observation, info = env.reset()
            state.append(getFrame(observation))
            state.append(getFrame(observation))
            state.append(getFrame(observation))
            state.append(getFrame(observation))
            games_played += 1

if __name__ == "__main__":
    test()