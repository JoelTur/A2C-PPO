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

def train(states, actions, A, agent, optimizer, G, old_probs, old_valuess):
    print("LEARNING_RATE ", optimizer.param_groups[0]['lr'])
    indexs = np.arange(len(states))
    total_loss = 0
    total_entropy_loss = 0
    total_policy_loss = 0
    total_values_loss = 0
    update_steps = 0
    total_kl_approx_mean = 0
    
    for iter in range(config.training.max_iterations):
        lower_M = 0
        upper_M = 64
        np.random.shuffle(indexs)
        for m in range(32):
            index = indexs[lower_M:upper_M]
            state = states[index]
            G_ = G[index]
            A_ = A[index]
            actions_ = actions[index]
            pred, values = agent(state)
            new_dist = torch.distributions.Categorical(torch.exp(pred))
            entropies = new_dist.entropy()
            old_pred = old_probs[index]
            old_values = old_valuess[index]
            values = torch.squeeze(values)
            old_pred = torch.squeeze(old_pred)
            actions_ = actions_*A_.unsqueeze(1)
            
            pred_ratio = torch.exp(pred - old_pred)
            clip = torch.clamp(pred_ratio, 1-config.training.epsilon, 1+config.training.epsilon)
            policy_loss = -torch.mean(torch.min(pred_ratio*actions_, clip*actions_))

            clip = old_values + (values - old_values).clamp(-config.training.epsilon, config.training.epsilon)
            values_loss = (G_-values)**2
            clip_loss = (clip-values)**2
            values_loss = torch.max(values_loss, clip_loss)
            values_loss = torch.mean(values_loss)

            entropy_loss = torch.mean(entropies)

            loss = config.training.beta*values_loss + policy_loss - config.training.alpha*entropy_loss

            optimizer.zero_grad()
            loss.backward()
            for param in agent.parameters():
                param.grad.data.clamp_(-config.training.gradient_clip, config.training.gradient_clip)
            optimizer.step()

            lower_M += 64
            upper_M += 64
            total_loss += loss.item()
            total_entropy_loss += entropy_loss
            total_policy_loss += policy_loss
            total_values_loss += values_loss
            kl_approx = 0.5*(pred-old_pred)**2
            kl_approx = torch.mean(kl_approx)
            total_kl_approx_mean += kl_approx
            update_steps += 1
            if kl_approx > config.training.kl_limit:
                print("BREAKING AT STEP", update_steps)
                break
    return total_loss/(update_steps), total_entropy_loss/(update_steps), total_values_loss/(update_steps), total_policy_loss/(update_steps), total_kl_approx_mean / update_steps

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
        return np.random.choice(action_space_size, p=prob), values, logprob

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    env = gym.make(config.environment.name, render_mode=config.environment.render_mode)
    action_space_size = env.action_space.n
    state = deque(maxlen=4)
    
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb['project'],
            entity=config.logging.wandb['entity']
        )
    
    # Actors in the simulation
    actor_agent = ACTORCRITIC.NeuralNetwork(action_space_size).to(device)
    
    # Load pretrained model if configured
    if config.pretrained.use_pretrained:
        model_path = config.pretrained.model_path.format(game=config.environment.name)
        print(model_path)
        loadModel(actor_agent, model_path)
        
        # Freeze specified layers
        for name, param in actor_agent.named_parameters():
            if name in config.pretrained.freeze_layers:
                param.requires_grad = False
                
        # Adjust learning rate for fine-tuning
        learning_rate = config.training.learning_rate * config.pretrained.learning_rate_multiplier
    else:
        learning_rate = config.training.learning_rate
    
    # Optimization stuff
    optimizer = optim.Adam(actor_agent.parameters(), lr=learning_rate)
    # Transition class
    transition = Transition.Transition(action_space_size)

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
            
        if batch_steps % config.training.batch_steps == 0:
            print("POLICY/VALUES UPDATED", update, "Gamesplayed: ", games_played, " Steps: ", total_time)
            # Put data to a tensor form
            G = transition.discounted_reward(config.training.gamma)
            G = torch.from_numpy(G).to(device).float()
            states = [torch.from_numpy(np.array(state)/255) for state in transition.states]
            states = torch.stack(states)
            states = states.float()
            actions = [torch.from_numpy(np.array(action)) for action in transition.actions]
            actions = torch.stack(actions)
            actions = actions.float()
            
            # TRAIN
            V_ESTIMATES = torch.stack(transition.reward_estimate)
            V_ESTIMATES = V_ESTIMATES.float()
            old_probs = torch.stack(transition.old_probs).float()
            old_rewards = torch.stack(transition.reward_estimate).float()
            total_loss, entropy_loss, values_loss, policy_loss, kl_approx = train(
                states.to(device), 
                actions.to(device),  
                (G-V_ESTIMATES).to(device), 
                actor_agent, 
                optimizer, 
                G, 
                old_probs, 
                old_rewards
            )
            print(total_loss, values_loss, policy_loss, entropy_loss, kl_approx)
            
            if config.logging.use_wandb:
                wandb.log({
                    "TOTAL LOSS": total_loss,
                    "ENTROPY LOSS": entropy_loss,
                    "VALUES LOSS": values_loss,
                    "POLICY LOSS": policy_loss,
                    "KL_APPROX MEAN": kl_approx
                })
                
            batch_steps = 0
            transition.resetTransitions()
            update += 1
            update_linear_schedule(optimizer, update, config.training.max_updates, learning_rate)
            
            if update % config.training.save_interval == 0:
                saveModel(actor_agent, config.paths.model_save.format(game=config.environment.name))

if __name__ == "__main__":
    main()