import sys
import time
import numpy as np
#from matplotlib.pyplot import grid
from constants import *
from environment import *
from state import State
"""
solution.py

This file is a template you should use to implement your solution.

You should implement code for each of the TODO sections below.

COMP3702 2022 Assignment 3 Support Code

Last updated by njc 12/10/22
"""

## All of my code is very inspired both by the pseudocode given in the learning material and the Tutorial 9 solution. 

class RLAgent:

    #
    # TODO: (optional) Define any constants you require here.
    exploit_prob = 0.9

    def __init__(self, environment: Environment):
        #
        # TODO: (optional) Define any class instance variables you require (e.g. Q-value tables) here.
        self.environment = environment
        self.q_values = {}
        
        pass

    # === Q-learning ===================================================================================================

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        # TODO: Implement your Q-learning training loop here.
    
        episode_rewards = []
        R_100 = float(-100000000) #Just set this to a really low value
        timer = time.time()
        

        while (time.time()-timer) < self.environment.training_time_tgt and R_100 < self.environment.evaluation_reward_tgt:
                state = self.environment.get_init_state()
              
                rewardTot = 0
                iteration = 0
                while not self.environment.is_solved(state) and (iteration < 100):
                    action = self.choose_action()
                    reward, next_state = self.environment._Environment__apply_dynamics(state, action) #I tried using perform_action, but it did not work then. 
                    rewardTot += reward

                    old_q_value = self.q_values.get((state,action), 0)
                    best_next = self._get_best_action(next_state)

                    best_next_q = self.q_values.get((next_state, best_next), 0)
                    if self.environment.is_solved(next_state):
                        best_next_q = 0
                    target = reward + self.environment.gamma * best_next_q

                    new_q = old_q_value + self.environment.alpha * (target - old_q_value)
                    self.q_values[(state, action)] = new_q
                    state = next_state
                    iteration += 1
                episode_rewards.append(rewardTot)
                R_100 = np.mean(episode_rewards[-100:])
        f = open("policyQuality3.txt", "w")
        f.write(str(episode_rewards))   

        pass

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  Q-learning Q-values) here.

        best_action = self._get_best_action(state)
        return best_action

    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        #
        # TODO: Implement your SARSA training loop here.
        episode_rewards = []
        R_100 = float(-100000000) #Just set this to a really low value
        timer = time.time()


        while (time.time()-timer) < self.environment.training_time_tgt and R_100 < self.environment.evaluation_reward_tgt:
                state = self.environment.get_init_state()
              
                rewardTot = 0
                iteration = 0
                

                while not self.environment.is_solved(state) and (iteration < 100):
                    action = self._get_best_action(state)
                    reward, next_state = self.environment._Environment__apply_dynamics(state, action) 
                    rewardTot += reward

                    old_q_value = self.q_values.get((state,action), 0)
                    best_next = self._get_best_action(next_state)

                    best_next_q = self.q_values.get((next_state, best_next), 0)
                    if self.environment.is_solved(next_state):
                        best_next_q = 0
                    target = reward + self.environment.gamma * best_next_q

                    new_q = old_q_value + self.environment.alpha * (target - old_q_value)
                    self.q_values[(state, action)] = new_q
                    state = next_state
                    iteration += 1
                episode_rewards.append(rewardTot)
                R_100 = np.mean(episode_rewards[-100:])
                
        pass
    
    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  SARSA Q-values) here.

        return self._get_best_action(state)
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: (optional) Add any additional methods here.
    def _get_best_action(self, state: State):
        best_q = float('-inf')
        best_action = None
        for action in ROBOT_ACTIONS:
            q_value = self.q_values.get((state, action))
            if q_value is not None and q_value > best_q:
                best_q = q_value
                best_action = action
        return best_action
    
    def choose_action(self):
        current_state = self.environment.get_init_state()
        best_action = self._get_best_action(current_state)
        if best_action is None or random.random() < self.exploit_prob:
            return random.choice(ROBOT_ACTIONS)
        return best_action

