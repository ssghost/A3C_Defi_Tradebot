import os
from gym_env.defi_env import DefiEnv
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
from tf.compat.v1.keras.backend import set_session
from threading import Lock

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
K.set_session(sess)
graph = tf.get_default_graph()

class A3CAgent:
    def __init__(self):     
        self.env = DefiEnv()
        self.env_name = "defi_env"
        self.action_size = self.env.action_space.n
        self.lock = Lock()
        self.lr = 0.000025
        self.episode = self.env.episode
        self.Save_Path = 'Models'
        self.state_size = (3,)
        self.rewards = self.env.graph_reward
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A3C_{}'.format(self.env_name, self.lr)
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

    def reset(self):
        return self.env.reset()

    def act(self, state, action):
        prediction = self.Actor.predict(state)[0]
        if not action: 
            action = np.random.choice(self.action_size, p=prediction)
        self.env.step(action)
        return self.env.step(action), action

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
    
    def train(self):
        global graph
        with graph.as_default():
            e = 1
            while e < self.episode:
                state = self.reset()
                states, actions, rewards = [], [], []
                while not done:
                    next_state, reward, done, _, action = self.act(state)
                    states.append(state)
                    action_onehot = np.zeros([self.action_size])
                    action_onehot[action] = 1
                    actions.append(action_onehot)
                    rewards.append(reward)
                    state = next_state
                    self.replay(states, actions, rewards)
                self.lock.acquire()
                self.replay(states, actions, rewards)
                self.lock.release()
                with self.lock:
                    average = np.mean(self.rewards)
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                    if(e < self.episodes):
                        e += 1
            self.env.close()
        print("Training is done.")         

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(self.episode):
            state = self.reset()
            done = False
            while not done:
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _, _ = self.act(state, action)
                assert self.rewards[-1] == reward
        print(f"Average Reward: {np.mean(self.rewards)}.")
        self.env.close()

