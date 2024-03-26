import numpy as np  
import time
import math

import importlib 
from random import shuffle
import tensorflow.keras.backend as K
import tensorflow as tf
from pickle import Pickler, Unpickler
 
from copy import deepcopy


module = importlib.import_module('111022533_hw1_4_test')

 
training_para = module.AttrDict({
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



class RandomAgent():
    
    def choose_action(self, state):
        
        player = state[0]
        board = np.reshape(state[1:], module.Game.boardShape)
        game = module.Game(board, player)

        a = np.random.randint(module.Game.actionSize)
        valids = game.getValidActions()
        
        while valids[a]!=1:
            a = np.random.randint(module.Game.actionSize)

        z,y,x = module.Game.decode_action(a) 
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
 
            game = module.Game()  
            epsExamples = []  

            while not game.is_done():
                 
                temp = int(game.getTimeStep() < self.para.tempThreshold)  
                pi = self.agent.predict(game, temp=temp)
 
                sym = module.Game.getSymmetries(game.getPrimeBoard(), pi) # data augmentation
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
        
        print('pitting random model...')

        mid = int(num / 2)  

        self.agent.set_n_sim(self.para.nSims_eval)

        scores = {1: 0, -1: 0, 0: 0}

        for i in range(1, num+1) :

            self.agent.reset() 
            players = {1: self.agent, -1: RandomAgent()}  

            if(i % 2 == 0): game = module.Game(player = 1) 
            else: game = module.Game(player = -1) 
              
            while not game.is_done():
                action = players[game.getPlayer()].choose_action(game.getState())   
                a = module.Game.encode_action(action)
                valids = game.getValidActions()
                assert valids[a] > 0
                game.step(a)

            winner = game.getWinner()
            scores[winner] += 1 

            print(f'[{i}/{num}] agent: {scores[1]}, random: {scores[-1]}, draw: {scores[0]}') 
 



def train(): 
 
 
    # agent = Agent()
    agent = module.Agent()  
    # agent.load_checkpoint('./111022533/temp/checkpoint_25.h5')
 
 
    replayBuf = ReplayBuffer(training_para.buf_size)
    # replayBuf.load('./111022533/temp/checkpoint_25.pth.tar.examples')

    trainer = Trainer(agent, replayBuf, training_para)
    trainer.train()
 
# train()


def evaluate(num=100):
    
    print('pitting random model...')

    mid = int(num / 2) 
  
    agent = module.Agent() 
    agent.load_policy()

    scores = {1: 0, -1: 0, 0: 0}

    for i in range(1, num+1) :

        agent.reset()
        players = {1: agent, -1: RandomAgent()}  

        if(i % 2 == 0): game = module.Game(player = 1) 
        else: game = module.Game(player = -1) 
            
        while not game.is_done():

            # if(game.getPlayer() == 1): start_time = time.time()
            action = players[game.getPlayer()].choose_action(game.getState())  
            # if(game.getPlayer() == 1): print(f"{time.time() - start_time} sec")

            a = module.Game.encode_action(action)
            valids = game.getValidActions()
            assert valids[a] > 0
            game.step(a)

        # print(f'board: {game.getBoard()}')
        winner = game.getWinner()
        scores[winner] += 1 

        print(f'[{i}/{num}] agent: {scores[1]}, random: {scores[-1]}, draw: {scores[0]}') 

 
# agent = module.Agent()  
# agent(tf.random.uniform(shape=[1,4,4,4]))
# agent.save_checkpoint('./111022533/111022533_hw1_4_data.h5')


evaluate()