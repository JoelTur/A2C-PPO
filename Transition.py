from typing import List, Union
import numpy as np
import numpy.typing as npt

class Transition:
    """A class to store and manage transitions in reinforcement learning.
    
    This class maintains lists of states, rewards, actions, and their associated
    probabilities and reward estimates. It also provides functionality to calculate
    discounted rewards.
    
    Attributes:
        actionSpaceSize (int): The size of the action space
        states (List[npt.ArrayLike]): List of state observations
        rewards (List[float]): List of rewards received
        actions (List[npt.ArrayLike]): List of one-hot encoded actions
        old_probs (List[npt.ArrayLike]): List of action probabilities
        reward_estimate (List[float]): List of estimated rewards
    """
    
    def __init__(self, actionspacesize: int) -> None:
        """Initialize the Transition object.
        
        Args:
            actionspacesize (int): The size of the action space
            
        Raises:
            ValueError: If actionspacesize is not positive
        """
        if actionspacesize <= 0:
            raise ValueError("actionspacesize must be positive")
        self.actionSpaceSize = actionspacesize
        self.states: List[npt.ArrayLike] = []
        self.rewards: List[float] = []
        self.actions: List[npt.ArrayLike] = []
        self.old_probs: List[npt.ArrayLike] = []
        self.reward_estimate: List[float] = []

    def addTransition(self, 
                     state: npt.ArrayLike, 
                     reward: float, 
                     action: int, 
                     reward_estimate: float, 
                     probs: npt.ArrayLike) -> None:
        """Add a new transition to the storage.
        
        Args:
            state: The state observation
            reward: The reward received
            action: The action taken (index)
            reward_estimate: The estimated reward
            probs: The action probabilities
            
        Raises:
            ValueError: If action is out of bounds
        """
        if not 0 <= action < self.actionSpaceSize:
            raise ValueError(f"Action {action} is out of bounds for action space size {self.actionSpaceSize}")
            
        self.states.append(state)
        self.rewards.append(reward)
        action_one_hot = np.zeros(self.actionSpaceSize)
        action_one_hot[action] = 1
        self.actions.append(action_one_hot)
        self.reward_estimate.append(reward_estimate)
        self.old_probs.append(probs)

    def resetTransitions(self) -> None:
        """Reset all transition storage to empty lists."""
        self.states = []
        self.rewards = []
        self.actions = []
        self.old_probs = []
        self.reward_estimate = []
        
    def discounted_reward(self, gamma: float) -> npt.ArrayLike:
        """Calculate discounted rewards with normalization.
        
        Args:
            gamma: The discount factor
            
        Returns:
            Array of normalized discounted rewards
            
        Raises:
            ValueError: If gamma is not between 0 and 1
        """
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1")
            
        if not self.rewards:
            return np.array([])
            
        G = np.zeros(len(self.rewards))
        running_sum = 0
        
        # Calculate discounted reward
        for t in reversed(range(len(self.rewards))):
            if self.rewards[t] != 0:
                running_sum = 0
            running_sum = running_sum * gamma + self.rewards[t]
            G[t] = running_sum
            
        # Normalize
        G = (G - np.mean(G)) / (np.std(G) + 1e-8)
        return G