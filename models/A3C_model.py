import os
from gym_env.defi_env import DefiEnv
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
from tf.compat.v1.keras.backend import set_session
from threading import Lock, Thread
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
K.set_session(sess)
graph = tf.get_default_graph()

class A3CAgent:
    def __init__(self):     
        self.env_name = "defi_env"
        self.action_size = self.env.action_space.n
        self.lock = Lock()
        self.threads = 5
        self.env_train = [DefiEnv()]*self.threads
        self.env_test = DefiEnv()
        self.lr = 0.000025
        self.episode = self.env.episode
        self.episode_test = 0
        self.Save_Path = '.'
        self.state_size = (3,)
        self.rewards = self.env.graph_reward
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A3C_{}'.format(self.env_name, str(self.lr).replace('.',''))
        self.Model_name = os.path.join(self.Save_Path, self.path)

        self.Actor, self.Critic = self.A3CModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr)

        self.Actor._make_predict_function()
        self.Critic._make_predict_function()

        global graph
        graph = tf.get_default_graph()

    def A3CModel(self, input_shape, action_space, lr):
        X_input = Input(input_shape)
        X = Flatten(input_shape=input_shape)(X_input)
        X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
        X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
        X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)
        X = Dense(16, activation="elu", kernel_initializer='he_uniform')(X) 

        action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
        value = Dense(1, kernel_initializer='he_uniform')(X)

        Actor = Model(inputs = X_input, outputs = action)
        Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

        Critic = Model(inputs = X_input, outputs = value)
        Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

        return Actor, Critic

    def reset(self, env):
        return env.reset()

    def act(self, state, action, env):
        prediction = self.Actor.predict(state)[0]
        if not action: 
            action = np.random.choice(self.action_size, p=prediction)
        env.step(action)
        return env.step(action), action

    def replay(self, states, actions):
        states = np.vstack(states)
        actions = np.vstack(actions)
        reward = self.rewards[-1]
        value = self.Critic.predict(states)[:, 0]
        advantages = reward - value

        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, reward, epochs=1, verbose=0)
 
    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        self.Critic.save(self.Model_name + '_Critic.h5')
    
    def train(self, env_i):
        cur_env = self.env_train[env_i]
        global graph
        with graph.as_default():
            e = 1
            while e <= self.episode:
                state = self.reset(cur_env)
                states, actions, rewards = [], [], []
                while not done:
                    next_state, reward, done, _, action = self.act(state, cur_env)
                    states.append(state)
                    action_onehot = np.zeros([self.action_size])
                    action_onehot[action] = 1
                    actions.append(action_onehot)
                    rewards.append(reward)
                    self.rewards.append(reward)
                    state = next_state
                    self.replay(states, actions, rewards)
                self.lock.acquire()
                self.replay(states, actions, rewards)
                self.rewards = rewards
                self.episode = cur_env.episode
                self.lock.release()
                with self.lock:
                    average = np.mean(self.rewards)
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                    if(e < self.episodes):
                        e += 1
            self.episode_test += cur_env.episode
            cur_env.close()
        print(f"Training thread {env_i} is done.") 
        
    def train_with_threads(self):
        threads = self.threads
        for env in self.env_train:
            env.close()
        Threads = [Thread(target=self.train,
                          daemon=True,
                          args=(self, i)) for i in range(threads)]
        for i, t in enumerate(Threads):
            time.sleep(2)
            t.start()
            print(f"Started training thread {i+1}.")
        
    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        assert self.episode_test != 0
        if self.episode_test > 1000:
            self.episode_test = int(self.episode_test/5)
        for e in range(self.episode_test):
            state = self.reset(self.env_test)
            done = False
            while not done:
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _, _ = self.act(state, action, self.env_test)
                assert self.rewards[-1] == reward
        print(f"Average Reward: {np.mean(self.rewards)}.")
        self.env_test.close()

