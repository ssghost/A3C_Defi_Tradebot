import numpy as np
import gym
import random
from gym import spaces
from scripts.operations import *
from gym_anytrading.datasets import STOCKS_GOOGL

class DefiEnv(gym.Env):
    def __init__(self):
        self.df = STOCKS_GOOGL
        self.reward_range = (-Web3.toWei(0.1, "ether"),
            Web3.toWei(0.1, "ether"))
        self.account = get_account()
        self.pool = get_pool()
        self.total = get_data()[0]
        self.graph_reward = []
        self.episode = 1
        self.max_steps = 1000

        self.action_space = spaces.Box(low=-1,
            high=1,
            shape=(1, ),
            dtype=np.float16)
        self.observation_space = spaces.Box(low=-1,
            high=1,
            shape=(3, ),
            dtype=np.float16)

    def reset(self):
        self.account = get_account()
        self.pool = get_pool()
        self.total = get_data()[0]
        self.graph_reward = []
        deposit()
        start = self.df.index[0]
        weights = [i for i in start]
        self.current_step = random.choices(start, weights)[0]
        self.start_step = self.current_step
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([self.df.loc[self.current_step—1:self.current_step, 'borrow'], 
                        self.df.loc[self.current_step—1:self.current_step, 'repay'], 
                        self.df.loc[self.current_step—1:self.current_step, 'hold']])
        return obs

    def _take_action(self, action):
        if action[0] > 0:
            borrow()
        elif action[0] < 0:
            repay()
        else:
            pass

    def step(self, action, end=True):
        self._take_action(action)
        self.current_step += 1
        reward = float((total_asset() - self.total)/self.total)
        self.total = total_asset()
        if self.current_step >= self.max_steps:
            end=True
        else:
            end=False
        done = (self.total<=0) or end
        if done and end:
            self.graph_reward.append(reward) 
            self.episode += 1
            obs = self._next_observation()
        return obs, reward, done, {}

    
        