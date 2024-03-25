import numpy as np
# import torch

from collections import deque
import os
import numpy as np
import time
import math

import importlib

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from random import shuffle
import tensorflow.keras.backend as K
import tensorflow as tf
from pickle import Pickler, Unpickler


from copy import deepcopy



class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]


training_para = AttrDict({
    'nEps': 10, 
    'nEpochs': 5,
    'nEvals': 6,
    'evaluate_every_n': 4, 
    'save_every_n': 5,        
    'batch_size': 64,
    'buf_size': 2**16,

    'nSims_train': 25,
    'nSims_eval': 25,

    'nIters': 100000, 
    'tempThreshold': 15,    
    'checkpoint_dir': './111022533/temp/', 
})


# training_para = AttrDict({
#     'nEps': 1, 
#     'nEpochs': 1,
#     'nEvals': 2,
#     'evaluate_every_n': 4, 
#     'save_every_n': 5,        
#     'batch_size': 64,
#     'buf_size': 2**16,

#     'nSims_train': 25,
#     'nSims_eval': 25,

#     'nIters': 100000, 
#     'tempThreshold': 15,    
#     'checkpoint_dir': './111022533/temp/', 
# })



    

   
class Game():

    actionSize = 64
    boardShape = (4,4,4)
    action_decode_map = np.arange(actionSize).reshape(shape)

    def __init__(self, board=np.zeros(Game.boardShape), player=1):
       
        self.t = 0
        self.board = np.copy(board)
        self.player = player

        self.validActions = np.zeros(Game.actionSize)
        for a in range(Game.actionSize):
            z,y,x = Game.decode_action(a)
            if(self.board[z][y][x] == 0):
                self.validActions[a] = 1

        if Game.has_winner(self.board):
            self.winner = -self.player
            self.done = True
        elif sum(self.validActions) < 0.5:
            self.winner = 0
            self.done = True
        else:
            self.winner = 0 
            self.done = False          
 
    def getTimeStep(self):
        return self.t

    def getWinner(self):
        return self.winner 

    def is_done(self):
        return self.done

    def getState(self): # player + board
        return np.hstack((np.array([self.getPlayer()]), self.getBoard().reshape(Game.actionSize)))

    def getPlayer(self):
        return  self.player

    def getBoard(self):
        return np.copy(self.board)
    
    def getCanonBoard(self):
        return np.copy(self.board) * self.player

    def getValidActions(self):
        return np.copy(self.validActions)

    def duplicate(self):
        return deepcopy(self)

    def step(self, a):
        assert not self.is_done()
        assert self.validActions[a]
        self.validActions[a] = 0

        z,y,x = Game.decode_action(a)    
        self.board[z][y][x] = self.player
                
        if Game.has_winner(self.board):
            self.winner = self.player
            self.done = True
        elif sum(self.validActions) < 0.5:
            self.done = True
        else:
            self.player = -self.player 
        
        self.t += 1

    @staticmethod
    def encode_action(a):
        return a[0] + 4*a[1] + 16*a[2]

    @staticmethod
    def decode_action(a):
        return np.argwhere(Game.action_decode_map==a)[0]   

    @staticmethod
    def encode_state(board):
        return board.tostring()

    @staticmethod
    def has_winner(board): 

        n = Game.boardShape[0]

        def reduce(v): return int(abs(sum(v)))
   
        for z in range(n):
            for y in range(n):                
                if(reduce(board[z,y,:]) == n): return True 
        for z in range(n):
            for x in range(n):                
                if(reduce(board[z,:,x]) == n): return True 
        for y in range(n):
            for x in range(n):                
                if(reduce(board[:,y,x]) == n): return True
 
        for z in range(n):
            if(reduce([board[z,i,i] for i in range(4)]) == n): return True
            if(reduce([board[z,i,n-i-1] for i in range(4)]) == n): return True            
        for y in range(n):
            if(reduce([board[i,y,i] for i in range(4)]) == n): return True
            if(reduce([board[i,y,n-i-1] for i in range(4)]) == n): return True            
        for x in range(n):
            if(reduce([board[i,i,x] for i in range(4)]) == n): return True
            if(reduce([board[n-i-1,i,x] for i in range(4)]) == n): return True

        if(reduce([board[i,i,i] for i in range(4)]) == n): return True
        if(reduce([board[n-i-1,i,i] for i in range(4)]) == n): return True
        if(reduce([board[n-i-1,n-i-1,i] for i in range(4)]) == n): return True
        if(reduce([board[i,n-i-1,i] for i in range(4)]) == n): return True
            
        return False

    @staticmethod
    def getSymmetries(board, pi):
        # mirror, rotational
        n = Game.boardShape[0]
 
        l = []
        newB = np.reshape(board, (n*n, n))
        newPi = np.reshape(pi, Game.shape)
        for i in range(1,5):

            for z in [True, False]:
                for j in [True, False]:
                    if j:
                        newB = np.fliplr(newB)
                        newPi = np.fliplr(newPi)
                    if z:
                        newB = np.flipud(newB)
                        newPi = np.flipud(newPi)
                    
                    newB = np.reshape(newB, Game.shape)
                    newPi = np.reshape(newPi, Game.shape)
                    l += [(newB, list(newPi.ravel()))]
        return l 
    
 


 
class Model(tf.keras.Model):

    def __init__(self):
  
        super(Model, self).__init__() 

        self.act = {}
        self.bn = {}
        self.conv = {}    
        padding = ['same', 'same', 'valid']
        n_filter = 128

        for i in range(3):
            self.act[i] = tf.keras.layers.Activation('relu')
            self.bn[i] = tf.keras.layers.BatchNormalization(axis=3)
            self.conv[i] = tf.keras.layers.Conv3D(n_filter, 3, padding=padding[i])

        self.flatten = tf.keras.layers.Flatten()
 
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.act1 = tf.keras.layers.Activation('relu')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)
        self.fc1 = tf.keras.layers.Dense(512)  

        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.act2 = tf.keras.layers.Activation('relu')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1)
        self.fc2 = tf.keras.layers.Dense(256)  

        self.fc_a = tf.keras.layers.Dense(Game.actionSize, activation='softmax')  
        self.fc_v = tf.keras.layers.Dense(1, activation='tanh')         
       
        self.optimizer = tf.keras.optimizers.Adam(0.001)
 

 
     
    @tf.function
    def call(self, x, training=None):
 
        x = tf.expand_dims(x, axis=4)
 
        for i in range(3):
            x = self.conv[i](x)
            x = self.bn[i](x)
            x = self.act[i](x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        a_prob = self.fc_a(x)
        v = self.fc_v(x)

        return a_prob, v
     

    @tf.function
    def _train_step(self, batch): 

        x, a_prob, v = batch 

        with tf.GradientTape() as tape:
            
            a_prob_pred, v_pred = self(x)

            v_pred = tf.squeeze(v_pred)

            a_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(a_prob, a_prob_pred))
            v_loss = tf.reduce_mean(tf.square(v - v_pred))
 
            total_loss = (a_loss + v_loss)

        gradients = tape.gradient(total_loss, self.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
        return a_loss, v_loss
    

    def train_step(self, batch):
        
        x, a_prob, v = list(zip(*batch))

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        a_prob = tf.convert_to_tensor(a_prob, dtype=tf.float32)
        v = tf.convert_to_tensor(v, dtype=tf.float32)
        a_loss, v_loss = self._train_step((x, a_prob, v))
        return a_loss.numpy(), v_loss.numpy()
        
  
    def _predict(self, game): 

        board = game.getCanonBoard()
        board = board[np.newaxis, :, :] 
        board = tf.convert_to_tensor(board, dtype=tf.float32) 
        pi, v = self(board) 
        
        pi = pi[0].numpy()
        v = v[0].numpy()[0]
 
        K.clear_session()
        tf.keras.backend.clear_session() 

        valids = game.getValidActions()
        pi = pi * valids  
        if np.sum(pi) > 0: pi = pi / np.sum(pi)  
        else: pi = valids / np.sum(valids)

        return pi, v 
 

    def save_checkpoint(self, path):  
        print(f'saved ckpt {path}') 
        self.save_weights(path)
        


    def load_checkpoint(self, path): 
        # need call once to enable load weights.
        self(tf.random.uniform(shape=[1,4,4,4]))
        self.load_weights(path)



class Agent(Model):
     
    def __init__(self, para):
        super().__init__()  

        self.para = para     
        self.reset()

    def set_n_sim(self, n):
        self.para.nSims = n

    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited 
        self.Ps = {}  # stores initial policy (returned by neural net) 
        
    def choose_action(self, state):

        player = state[0]
        board = np.reshape(state[1:], Game.boardShape)
        game = Game(board, player)

        a = np.argmax(self.predict(game))
        z,y,x = Game.decode_action(a) 
        assert board[z][y][x] == 0
        return [x, y, z]
     

    def predict(self, game, temp=1):
        """
        temp: smaller is sharper (more deterministic).
        """

        for i in range(self.para.nSims):
            self.search(game.duplicate())

        s = Game.hash(canonBoard)
        counts = [self.Nsa[s][a] if a in self.Nsa[s] else 0 for a in range(Game.actionSize)]
 
        if temp == 0: # deterministic
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
    
        return probs
 

    def search(self, game): 
        """
        select & expand-> simulate -> backprop
        """
        trajectory, game = self.select_expand(game)
        reward = self.simulate(game) 
        self.backprop(trajectory, reward)
 

    def select_expand(self, game):

        trajectory = {1: [], -1: []}
        while(True):  
            
            s = Game.encode_state(game.getCanonBoard())
            if s not in self.Ps: # terminal or first visit.
                return trajectory, game
            
            a = self.max_a(s) 
            trajectory[game.getPlayer()].append((s, a))
 
            game.step(a)

       
    def simulate(self, game):
        
        reward = {1: 0, -1: 0}

        if game.is_done(): 
            winner = game.getWinner() 
            if winner != 0:
                reward[winner] = 1
                reward[-winner] = -1  
            return reward          

        # first visit

        s = Game.encode_state(game.getCanonBoard())

        self.Ps[s], v = self._predict(game)          
        self.Qsa[s] = {}
        self.Nsa[s] = {}

        valids = game.getValidActions()        
        for a in range(Game.actionSize):
            if valids[a]:
                self.Qsa[s][a] = 0
                self.Nsa[s][a] = 0

        self.Ns[s] = 0

        p = game.getPlayer()
        reward[p] = v
        reward[-p] = -v 
                
        return reward


    def backprop(self, trajectory, reward): 
        for player in trajectory: 
            for s, a in reversed(trajectory[player]): 
                self.Qsa[s][a] = (self.Nsa[s][a] * self.Qsa[s][a] + reward[player]) / (self.Nsa[s][a] + 1)
                self.Nsa[s][a] += 1 
                self.Ns[s] += 1

 
    def max_a(self, s):
        """ 
            - pick the action with the highest upper confidence bound
            - 1st term "self.Qsa[s][a]": encourage exploitation.
            - 2nd term "(...) / (1 + self.Nsa[s][a])": encourage exploration.
        """
        bound_max = -float('inf')
        a_max = None        
        for a in self.Qsa[s].keys(): # all are valids actions.
            bound = self.Qsa[s][a] + self.para.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8) / (1 + self.Nsa[s][a])
            if bound > bound_max:
                bound_max = u
                a_max = a
        
        return a_max




class RandomPlayer():
    
    def choose_action(self, state):
        
        player = state[0]
        board = np.reshape(state[1:], Game.boardShape)
        game = Game(board, player)

        a = np.random.randint(Game.actionSize)
        valids = game.getValidActions()
        
        while valids[a]!=1:
            a = np.random.randint(Game.actionSize)

        z,y,x = Game.decode_action(a) 
        assert board[z][y][x] == 0
        return [x, y, z]

class ReplayBuffer():

    def __init__(self, maxLength):
        self.maxLength = maxLength
        self.buf = []
 

    def addExamples(self, examples):
         
        excess = len(self.buf) + len(examples) - self.maxLength

        if(excess > 0): 
            tmp = []
            tmp.extend(self.buf[excess:])
            self.buf = tmp   
        
        self.buf.extend(examples) 

    def sample(self):
        out = [i for i in self.buf]
        shuffle(out)
        return out

    def save(self, path): 
        with open(path, "wb+") as f:
            Pickler(f).dump(self.buf)

    def load(self, path):
        with open(path, "rb") as f:
            self.buf = Unpickler(f).load()

        print(f'loaded {len(self.buf)} examples')
 
  

 
class Trainer():

    def __init__(self, agent, replayBuf, para): 

        self.agent = agent
        self.para = para
        self.replayBuf = replayBuf

    def collectExamples(self): 

        examples = []

        self.agent.set_n_sim(self.para.nSims_train)

        for j in range(self.para.nEps): 
            
            self.agent.reset()
 
            game = Game()  
            epsExamples = []  

            while not game.is_done():
                 
                temp = int(game.getTimeStep() < self.para.tempThreshold)  
                pi = self.agent.predict(game, temp=temp)
 
                sym = Game.getSymmetries(game.getCanonBoard(), pi) # data augmentation
                for b, p in sym:
                    epsExamples.append([b, p, game.getPlayer()])

                action = np.random.choice(len(pi), p=pi)  
                game.step(action)
           

            winner = game.getWinner()  
            reward = {1: 0, -1: 0}
            if(winner != 0):
                reward[winner] = 1
                reward[-winner] = -1
            examples.extend([(x[0], x[1], reward[x[2]]) for x in epsExamples]) 
         

        return examples
        
 

    def train(self):

        for i in range(self.para.nIters):
            print('##############')
            print(f'iter: {i}')
 
            print('collecting examples...')
            self.replayBuf.addExamples(self.collectExamples())  
            examples = self.replayBuf.sample()
            
            self.agent.save_checkpoint(self.para.checkpoint_dir + f'temp.h5')  

            print('training...')            
            a_losses = []
            v_losses = []
            n = int(len(examples) / self.para.batch_size)  
            for epoch in range(self.para.nEpochs):     
                for j in range(n):
                    batch = examples[j*self.para.batch_size:(j+1)*self.para.batch_size] 
                    a_loss, v_loss = self.agent.train_step(batch)
                    # self.agent.train_step(batch)
                    a_losses.append(a_loss)
                    v_losses.append(v_loss)
            print(f'a_loss: {np.mean(a_losses)}, v_loss: {np.mean(v_losses)}')
 

            if i % self.para.save_every_n == 0:
                self.replayBuf.save(self.para.checkpoint_dir + f'checkpoint_{i}.pth.tar.examples')
                self.agent.save_checkpoint(self.para.checkpoint_dir + f'checkpoint_{i}.h5')        
 
            if i % self.para.evaluate_every_n == 0: 
                self.evaluate(self.para.nEvals)


               

    def evaluate(self, num=10):
        
        print('pitting old model...')

        mid = int(num / 2) 
        
        pnet = Agent(AttrDict({'nSims': self.para.nSims_eval, 'cpuct':1.0})) 
        pnet.load_checkpoint(self.para.checkpoint_dir + f'temp.h5')                        
  
        self.agent.set_n_sim(self.para.nSims_eval)

        for i in range(1, num+1) :

            self.agent.reset()
            pnet.reset()
            players = {1: self.agent, -1: pnet} 
            
            scores = {1: 0, -1: 0, 0: 0}

            if(i < mid): game = Game(player = 1) 
            else: game = Game(player = -1) 
              
            while not game.is_done():
                action = players[game.getPlayer()].choose_action(game.getState())   
                a = Game.encode_action(action)
                valids = game.getValidActions()
                assert valids[a] > 0
                game.step(a)

            winner = game.getWinner()
            scores[winner] += 1 

            print(f'[{i}/{num}] pwins: {scores[1]}, nwins: {scores[-1]}, draws: {scores[0]}') 


        if scores[1] + scores[-1] == 0 or float(scores[1]) / (scores[1] + scores[-1]) < 0.5:
            self.agent.load_checkpoint(self.para.checkpoint_dir + f'temp.h5')
        else:
            self.agent.save_checkpoint(self.para.checkpoint_dir + 'best.h5')


    # def evaluate(self, num=10):
        
    #     print('pitting random model...')

    #     mid = int(num / 2)
    #     oneWon = 0
    #     twoWon = 0
    #     draws = 0
        
    #     self.agent.set_n_sim(self.para.nSims_eval)

    #     for i in range(1, num+1) :

    #         self.agent.reset()
            
    #         players = {1: self.agent, -1: RandomPlayer()}


    #         if(i < mid): curPlayer = 1
    #         else: curPlayer = -1
            
    #         canonBoard = Game.getInitBoard()
        
    #         while Game.getGameEnded(canonBoard) == 0:
                
    #             board = canonBoard * curPlayer
    #             state = np.hstack((np.array([curPlayer]), board.reshape(Game.actionSize)))

    #             a = players[curPlayer].choose_action(state) 
    #             actionId = a[0] + 4*a[1] + 16*a[2]
    #             valids = Game.getValidMoves(canonBoard)
                
    #             assert valids[actionId] > 0
    
    #             canonBoard = Game.getNextState(canonBoard, actionId) 
    #             curPlayer = -curPlayer

    #         r = curPlayer * Game.getGameEnded(canonBoard)
                        
    #         if r == 1: oneWon += 1
    #         elif r == -1: twoWon += 1
    #         else: draws += 1       

    #         print(f'[{i}/{num}] pwins: {oneWon}, nwins: {twoWon}, draws: {draws}') 
 
 



def train(): 


    agent_para = AttrDict({   
        'nSims': 25,  
        'cpuct':1.0
    })
 
    agent = Agent(agent_para)
    # agent.load_checkpoint('./temp/checkpoint_18.h5')
    # agent.load_checkpoint('./111022533/temp/checkpoint_25.h5')
 
    # module = importlib.import_module('111022533_hw1_4_test')
    # agent = module.Agent() 
    # agent.load_policy()
 
    replayBuf = ReplayBuffer(training_para.buf_size)
    # replayBuf.load('./111022533/temp/checkpoint_25.pth.tar.examples')

    # agent.save_checkpoint(training_para.checkpoint_dir + f'temp.h5')  

    # trainer = Trainer(agent, replayBuf, training_para)
    trainer = Trainer(agent, replayBuf, training_para)
    trainer.train()



train()



