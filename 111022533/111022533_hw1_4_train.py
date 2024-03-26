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



class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

# training_para = dotdict({
#     'nEps': 25, 
#     'nEpochs': 8,
#     'nEvals': 10,
#     'evaluate_every_n': 1,  
#     'save_every_n': 4,        
#     'batch_size': 64,
#     'buf_size': 2**16,

#     'nSims_train': 25,
#     'nSims_eval': 25,

#     'nIters': 100000, 
#     'tempThreshold': 10,   
#     'checkpoint_dir': './111022533/temp/', 
# })



training_para = dotdict({
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




 




class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n=3):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = np.zeros((n,n,n))

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        index1 = [None,None,None]
        for i in range(3):
            index1[i] = str(index[i])
        for i in range(len(index1)):
            x = index1[i]
            index1[i] = str(int(x) - 1)
        return self.pieces[list(map(int, index1))]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        moves = set()  # stores the legal moves.

        # Get all the empty squares (color==0)
        for z in range(self.n): 
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[z][x][y]==0:
                        newmove = (z,x,y)
                        moves.add(newmove)
        return list(moves)

    def has_legal_moves(self):
        for z in range(self.n):
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[z][x][y]==0:
                        return True
        return False
    
    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=white,-1=black)
        """
        win = self.n
        # check z-dimension
        count = 0
        for z in range(self.n):
            count = 0
            for y in range(self.n):
                count = 0
                for x in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True

        count = 0
        for z in range(self.n):
            count = 0
            for x in range(self.n):
                count = 0
                for y in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True
        
        # check x dimension
        count = 0
        for x in range(self.n):
            count = 0
            for z in range(self.n):
                count = 0
                for y in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True

        count = 0
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                count = 0
                for z in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True

        # check y dimension
        count = 0
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                count = 0
                for z in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True
        
        count = 0
        for y in range(self.n):
            count = 0
            for z in range(self.n):
                count = 0
                for x in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True
        
        # check flat diagonals
        # check z dimension
        count = 0
        for z in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[z,d,d]==color:
                    count += 1
            if count==win:
                return True
        
        count = 0
        for z in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[z,d,self.n-d-1]==color:
                    count += 1
            if count==win:
                return True

        # check x dimension
        count = 0
        for x in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[d,x,d]==color:
                    count += 1
            if count==win:
                return True

        count = 0
        for x in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[d,x,self.n-d-1]==color:
                    count += 1
            if count==win:
                return True

        # check y dimension
        count = 0
        for y in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[d,d,y]==color:
                    count += 1
            if count==win:
                return True

        count = 0
        for y in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[self.n-d-1,d,y]==color:
                    count += 1
            if count==win:
                return True
        
        # check 4 true diagonals
        count = 0
        if self.pieces[0,0,0] == color:
            count += 1
            if self.pieces[1,1,1] == color:
                count += 1
                if self.pieces[2,2,2] == color:
                    count += 1
                    if count == win:
                        return True
            
        count = 0
        if self.pieces[2,0,0] == color:
            count += 1
            if self.pieces[1,1,1] == color:
                count += 1
                if self.pieces[0,2,2] == color:
                    count += 1
                    if count == win:
                        return True
        
        count = 0
        if self.pieces[2,2,0] == color:
            count += 1
            if self.pieces[1,1,1] == color:
                count += 1
                if self.pieces[0,0,2] == color:
                    count += 1
                    if count == win:
                        return True
        
        count = 0
        if self.pieces[0,2,0] == color:
            count += 1
            if self.pieces[1,1,1] == color:
                count += 1
                if self.pieces[2,0,2] == color:
                    count += 1
                    if count == win:
                        return True

        # return false if no 3 is reached
        return False

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """

        (z,x,y) = move

        # Add the piece to the empty square.
        assert self.pieces[z][x][y] == 0
        self.pieces[z][x][y] = color

class Game():
    boardShape = (4,4,4)
    coord2actionId = np.arange(64).reshape(boardShape)

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n*self.n

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        boardvalues = np.arange(0,(self.n*self.n*self.n)).reshape(self.n,self.n,self.n)
        
        move = np.argwhere(boardvalues==action)[0]
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for z, x, y in legalMoves:
            boardvalues = np.arange(0,(self.n*self.n*self.n)).reshape(self.n,self.n,self.n)
            valids[boardvalues[z][x][y]] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

    def getCanonicalForm(self, board, player):

        # board: an np array of shape 4x4x4

        # return state if player==1, else return -state if player==-1
        return player*board


    def getSymmetries(self, board, pi):
        # mirror, rotational
        pi_board = np.reshape(pi, (self.n, self.n, self.n))
        l = []
        newB = np.reshape(board, (self.n*self.n, self.n))
        newPi = pi_board
        for i in range(1,5):

            for z in [True, False]:
                for j in [True, False]:
                    if j:
                        newB = np.fliplr(newB)
                        newPi = np.fliplr(newPi)
                    if z:
                        newB = np.flipud(newB)
                        newPi = np.flipud(newPi)
                    
                    newB = np.reshape(newB, (self.n,self.n,self.n))
                    newPi = np.reshape(newPi, (self.n,self.n,self.n))
                    l += [(newB, list(newPi.ravel()))]
        return l
    


    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()
 



 

class Agent(tf.keras.Model):

    def __init__(self):
  
        super(Agent, self).__init__() 

        self.act = {}
        self.bn = {}
        self.conv = {}    
        padding = ['same', 'same', 'valid']
        n_filter = 128

        self.game = Game(4)

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

        self.fc_a = tf.keras.layers.Dense(self.game.getActionSize(), activation='softmax')  
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
        # print(f'a_loss, v_loss: {a_loss}, {v_loss}')
        return a_loss.numpy(), v_loss.numpy()
        
  
    def _predict(self, canonboard): 

        board = canonboard
        board = board[np.newaxis, :, :] 
        board = tf.convert_to_tensor(board, dtype=tf.float32) 
        pi, v = self(board) 
        
        pi = pi[0].numpy()
        v = v[0].numpy()[0]
 
        K.clear_session()
        tf.keras.backend.clear_session() 

        mask = self.game.getValidMoves(canonboard, 1)
        pi = pi * mask  

        sum_Ps_s = np.sum(pi)
        if sum_Ps_s > 0:
            pi /= sum_Ps_s  
        else:
            pi = pi + valids
            pi /= np.sum(pi)

        return pi, v 
 
    def save_checkpoint(self, path):  
        print(f'saved ckpt {path}') 
        self.save_weights(path)
        


    def load_checkpoint(self, path): 
        # need call once to enable load weights.
        self(tf.random.uniform(shape=[1,4,4,4]))
        self.load_weights(path)






class MCTSAgent(Agent):
     
    def __init__(self, args):
        super().__init__()  
        self.game = game
        self.args = args    
        self.reset()


    def choose_action(self, state):

        player = state[0]
        board = state[1:] 
        canonBoard = board.reshape(Game.boardShape) * player
        
        a = np.argmax(self.predict(canonBoard))
 
        (z,y,x) = np.argwhere(Game.coord2actionId == a)[0] 
        assert canonBoard[z][y][x] == 0

        return [x, y, z]
    
 

    def set_n_sim(self, n):
        self.args.nSims = n
        
    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

  

    def predict(self, canonBoard, temp=1):
        """
            canonBoard: cannot be terminal state.
        """
        for i in range(self.args.nSims):
            self.search(canonBoard)

        s = self.game.stringRepresentation(canonBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        # counts = [self.Nsa[s][a] if a in self.Nsa[s] else 0 for a in range(game.getActionSize())]
 
        if temp == 0: # yes for pit, no for train
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
    
        return probs
 

    def search(self, canonicalBoard):

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

        if self.Es[s] != 0: # if ended.
            # terminal node
            return -self.Es[s] # end results of game: 1, -1, 1e-4.

        if s not in self.Ps:

            # leaf node
            # self.Ps[s]: 1d array, probability over all actions.
            self.Ps[s], v = self._predict(canonicalBoard)

            # a list of 0 and 1.
            valids = self.game.getValidMoves(canonicalBoard, 1)

            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    # 1st term "self.Qsa[(s, a)]": encourage exploitation.
                    # 2nd term "(...) / (1 + self.Nsa[(s, a)])": encourage exploration.
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act

        # always get next state as player 1 (canonical).
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)

        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            # avg of all self.Nsa[(s, a)] + 1 Qsa's.
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

 

game = Game(4)


class RandomPlayer():
    
    def choose_action(self, state):
        player = state[0]
        board = state[1:] 
        canonBoard = board.reshape(Game.boardShape) * player
        
        a = np.random.randint(game.getActionSize())
        valids = game.getValidMoves(canonBoard, 1) 
        while valids[a]!=1:
            a = np.random.randint(game.getActionSize())

        (z,y,x) = np.argwhere(Game.coord2actionId == a)[0] 
        assert canonBoard[z][y][x] == 0

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

    def __init__(self, mctsAgent, replayBuf, para): 

        self.mctsAgent = mctsAgent
        self.para = para
        self.replayBuf = replayBuf

    def collectExamples(self): 

        examples = []

        self.mctsAgent.set_n_sim(self.para.nSims_train)

        for j in range(self.para.nEps): 
            
            self.mctsAgent.reset()

            board = game.getInitBoard()  
            curPlayer = 1
            t = 1
            epsExamples = []  

            while True:
                 
                canonicalBoard = game.getCanonicalForm(board, curPlayer)

                temp = int(t < self.para.tempThreshold)             
                pi = self.mctsAgent.predict(canonicalBoard, temp=temp)
 
                sym = game.getSymmetries(canonicalBoard, pi) # data augmentation
                for b, p in sym:
                    epsExamples.append([b, p, curPlayer])

                action = np.random.choice(len(pi), p=pi) 
                # canonBoard = Game.getNextState(canonBoard, action) 
                # curPlayer = -curPlayer 

                board, curPlayer = game.getNextState(board, curPlayer, action)

                r = game.getGameEnded(board, curPlayer)
                t += 1
                 
                if r != 0: 
                    examples.extend([(x[0], x[1], r * ((-1) ** (x[2] != curPlayer))) for x in epsExamples]) 
                    break

        return examples
        
 

    def train(self):

        for i in range(self.para.nIters):
            print('##############')
            print(f'iter: {i}')
 
            print('collecting examples...')
            self.replayBuf.addExamples(self.collectExamples())  
            examples = self.replayBuf.sample()
            
            self.mctsAgent.save_checkpoint(self.para.checkpoint_dir + f'temp.h5')  

            print('training...')            
            a_losses = []
            v_losses = []
            n = int(len(examples) / self.para.batch_size)  
            for epoch in range(self.para.nEpochs):     
                for j in range(n):
                    batch = examples[j*self.para.batch_size:(j+1)*self.para.batch_size] 
                    a_loss, v_loss = self.mctsAgent.train_step(batch)
                    # self.mctsAgent.train_step(batch)
                    a_losses.append(a_loss)
                    v_losses.append(v_loss)
            print(f'a_loss: {np.mean(a_losses)}, v_loss: {np.mean(v_losses)}')
 

            if i % self.para.save_every_n == 0:
                self.replayBuf.save(self.para.checkpoint_dir + f'checkpoint_{i}.pth.tar.examples')
                self.mctsAgent.save_checkpoint(self.para.checkpoint_dir + f'checkpoint_{i}.h5')        
 
            if i % self.para.evaluate_every_n == 0: 
                self.evaluateRandom(self.para.nEvals)
                self.evaluate(self.para.nEvals)
                


               

    def evaluate(self, num=10):
        
        print('pitting old model...')

        mid = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
               
        pnet = MCTSAgent(dotdict({'nSims': self.para.nSims_eval, 'cpuct':1.0})) 
        pnet.load_checkpoint(self.para.checkpoint_dir + f'temp.h5')                        
  
        self.mctsAgent.set_n_sim(self.para.nSims_eval)

        for i in range(1, num+1) :

            self.mctsAgent.reset()
            pnet.reset()
            players = {1: self.mctsAgent, -1: pnet}


            if(i < mid): curPlayer = 1
            else: curPlayer = -1
            
            board = game.getInitBoard()
        
            while game.getGameEnded(board, curPlayer) == 0:
                
                
                canonBoard = board*curPlayer
                state = np.hstack((np.array([curPlayer]), board.reshape(game.getActionSize())))

                a = players[curPlayer].choose_action(state) 
                actionId = a[0] + 4*a[1] + 16*a[2]
                valids = game.getValidMoves(canonBoard, 1)
                
                assert valids[actionId] > 0
    
                board, curPlayer = game.getNextState(board, curPlayer, actionId) 
        

            r = curPlayer * game.getGameEnded(board, curPlayer)
                        
            if r == 1: oneWon += 1
            elif r == -1: twoWon += 1
            else: draws += 1       

            print(f'[{i}/{num}] pwins: {oneWon}, nwins: {twoWon}, draws: {draws}') 


        if twoWon + oneWon == 0 or float(oneWon) / (twoWon + oneWon) < 0.5:
            self.mctsAgent.load_checkpoint(self.para.checkpoint_dir + f'temp.h5')
        else:
            self.mctsAgent.save_checkpoint(self.para.checkpoint_dir + 'best.h5')



    def evaluateRandom(self, num=10):
        
        print('pitting random model...')

        mid = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
                     
  
        self.mctsAgent.set_n_sim(self.para.nSims_eval)

        for i in range(1, num+1) :

            self.mctsAgent.reset() 
            players = {1: self.mctsAgent, -1: RandomPlayer()}


            if(i < mid): curPlayer = 1
            else: curPlayer = -1
            
            board = game.getInitBoard()
        
            while game.getGameEnded(board, curPlayer) == 0:
                
                canonBoard = board*curPlayer
                state = np.hstack((np.array([curPlayer]), board.reshape(game.getActionSize())))

                a = players[curPlayer].choose_action(state) 
                actionId = a[0] + 4*a[1] + 16*a[2]
                valids = game.getValidMoves(canonBoard, 1)
                
                assert valids[actionId] > 0
    
                board, curPlayer = game.getNextState(board, curPlayer, actionId) 
        

            r = curPlayer * game.getGameEnded(board, curPlayer)
                        
            if r == 1: oneWon += 1
            elif r == -1: twoWon += 1
            else: draws += 1       

            print(f'[{i}/{num}] pwins: {oneWon}, nwins: {twoWon}, draws: {draws}') 

 

def train(): 


    agent_para = dotdict({   
        'nSims': 25,  
        'cpuct':1.0
    })
 
    mctsAgent = MCTSAgent(agent_para) 
 
    replayBuf = ReplayBuffer(training_para.buf_size) 
    trainer = Trainer(mctsAgent, replayBuf, training_para)
    trainer.train()

 
train()




