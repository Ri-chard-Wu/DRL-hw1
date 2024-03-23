import numpy as np
# import torch

from collections import deque
import os
import numpy as np
import time
import math
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from random import shuffle
import tensorflow.keras.backend as K
import tensorflow as tf

EPS = 1e-8

numEps = 20 #20
batch_size = 64
perBuf_size = 2**17 #2**17
epochs = 5 # 5

arenaCompare = 16 # 10
arenaCompareEveryN = 2

epsilon = 0.2
gamma = 0.99
# save_every_n_iter = 1

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

  

class Board():

    def __init__(self, n=3):
        self.n = n
        self.pieces = np.zeros((n,n,n))
 
    def __getitem__(self, index): 
        index1 = [None,None,None]
        for i in range(3):
            index1[i] = str(index[i])
        for i in range(len(index1)):
            x = index1[i]
            index1[i] = str(int(x) - 1)
        return self.pieces[list(map(int, index1))]

    def get_legal_moves(self, color):
        moves = set()  
 
        for z in range(self.n): 
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[z][y][x]==0:
                        newmove = (z,y,x)
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

        win = self.n
        # check z-dimension
        for z in range(self.n):
            for y in range(self.n):
                count = 0
                for x in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True

        for z in range(self.n):
            for x in range(self.n):
                count = 0
                for y in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True
        
        # check x dimension
        for x in range(self.n):
            for z in range(self.n):
                count = 0
                for y in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True

        for x in range(self.n):
            for y in range(self.n):
                count = 0
                for z in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True

        # check y dimension
        for y in range(self.n):
            for x in range(self.n):
                count = 0
                for z in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True
        
        for y in range(self.n):
            for z in range(self.n):
                count = 0
                for x in range(self.n):
                    if self.pieces[z,x,y]==color:
                        count += 1
                if count==win:
                    return True
        
        # check flat diagonals
        # check z dimension
        for z in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[z,d,d]==color:
                    count += 1
            if count==win:
                return True
        
      
        for z in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[z,d,self.n-d-1]==color:
                    count += 1
            if count==win:
                return True

        # check x dimension 
        for x in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[d,x,d]==color:
                    count += 1
            if count==win:
                return True
 
        for x in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[d,x,self.n-d-1]==color:
                    count += 1
            if count==win:
                return True

        # check y dimension 
        for y in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[d,d,y]==color:
                    count += 1
            if count==win:
                return True

       
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

        (z,y,x) = move
 
        assert self.pieces[z][y][x] == 0
        self.pieces[z][y][x] = color


class Game():
    ActionSize = 64

    def __init__(self, n):
        self.n = n

    def getInitBoard(self): 
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self): 
        return (self.n, self.n, self.n)

    def getActionSize(self): 
        return self.n*self.n*self.n

    def getNextState(self, board, player, action): 

        b = Board(self.n)
        b.pieces = np.copy(board)
        boardvalues = np.arange(0,(self.n*self.n*self.n)).reshape(self.n,self.n,self.n)
        
        # z, y, x
        move = np.argwhere(boardvalues==action)[0]
        
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player) 

        for z, y, x in legalMoves:
            boardvalues = np.arange(0,(self.n*self.n*self.n)).reshape(self.n,self.n,self.n)
            valids[boardvalues[z][y][x]] = 1
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
        return board.tostring()
 

    # @staticmethod
    # def getValidMoves(board, player):
    #     # return a fixed size binary vector
    #     valids = [0]*self.getActionSize()
    #     b = Board(self.n)
    #     b.pieces = np.copy(board)
    #     legalMoves =  b.get_legal_moves(player) 

    #     for z, y, x in legalMoves:
    #         boardvalues = np.arange(0,(self.n*self.n*self.n)).reshape(self.n,self.n,self.n)
    #         valids[boardvalues[z][y][x]] = 1
    #     return np.array(valids)
        

    @staticmethod
    def display(board):
        n = board.shape[0]
        for z in range(n):
            print("   ", end="")
            for y in range(n):
                print (y,"", end="")
            print("")
            print("  ", end="")
            for _ in range(n):
                print ("-", end="-")
            print("--")
            for y in range(n):
                print(y, "|",end="")    # print the row #
                for x in range(n):
                    piece = board[z][y][x]    # get the piece to print
                    if piece == -1: print("X ",end="")
                    elif piece == 1: print("O ",end="")
                    else:
                        if x==n:
                            print("-",end="")
                        else:
                            print("- ",end="")
                print("|")

            print("  ", end="")
            for _ in range(n):
                print ("-", end="-")
            print("--")

 
class Agent(tf.keras.Model):

    def __init__(self):
  
        super(Agent, self).__init__()

        self.game = Game(4)
        # self.action_size = game.getActionSize()

        self.args = dotdict({  
            'epochs': epochs,
            'batch_size': batch_size
        }) 

        self.init_layers()
        self.optimizer = tf.keras.optimizers.Adam(0.001)


    def init_layers(self):

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

        self.fc_a = tf.keras.layers.Dense(Game.ActionSize, activation='softmax')  
        self.fc_v = tf.keras.layers.Dense(1, activation='tanh')         
        
     
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
    def train_step(self, data): 

        # x: [bz, 4, 4, 4], a_prob: [bz, 65], v: [bz,]        
        x, a_prob, v = data 

        with tf.GradientTape() as tape:
            
            a_prob_pred, v_pred = self(x)
 
            a_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(a_prob, a_prob_pred))
            v_loss = tf.reduce_mean(tf.square(v - v_pred))
 
            total_loss = (a_loss + td_loss)

  
        gradients = tape.gradient(total_loss, self.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return total_loss
             
    # def choose_action(self, state):


    def predict(self, canonicalBoard): 

        board = canonicalBoard[np.newaxis, :, :] 
        board = tf.convert_to_tensor(board, dtype=tf.float32) 
        pi, v = self.model(board) 
        
        pi = pi[0].numpy()
        v = v[0].numpy()

        K.clear_session()
        tf.keras.backend.clear_session() 

        valids = self.game.getValidMoves(canonicalBoard, 1)

        pi = pi * valids  # masking invalid moves

        sum_Ps_s = np.sum(pi)
        if sum_Ps_s > 0:
            pi /= sum_Ps_s  # renormalize
        else:
            pi = pi + valids
            pi /= np.sum(pi)

        return pi, v 


    def train(self, examples):   
        
        batch_size = self.args.batch_size
 
        train_losses = []
         
        for epoch in range(self.args.epochs): 

            n = int(len(examples) / batch_size) 
            for i in range(n):
                _e = examples[i*batch_size:(i+1)*batch_size] 
                b, pis, vs = list(zip(*_e))

                b = tf.convert_to_tensor(b, dtype=tf.float32)
                pis = tf.convert_to_tensor(pis, dtype=tf.float32)
                vs = tf.convert_to_tensor(vs, dtype=tf.float32)
    
                total_loss = self.train_step((b, pis, vs))
                train_losses.append(total_loss.numpy())       

        return np.mean(train_losses)


    def save_checkpoint(self, path): 
        self.model.save_weights(path)
        print('saved ckpt') 


    def load_checkpoint(self, path): 
        # need call once to enable load weights.
        self.model(tf.random.uniform(shape=[1,4,4,4]))
        self.model.load_weights(path)




class MCTS():
     
    def __init__(self, game, agent, args):
        self.game = game
        self.agent = agent
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited 
        self.Ps = {}  # stores initial policy (returned by neural net) 
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Valids = {}  # stores game.getValidMoves for board s
 

    def getActionProb(self, canonicalBoard, temp=1):
      
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
 
        if temp == 0: # yes for pit, no for train
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        # return probs, v
        return probs

 

    def search(self, canonicalBoard): 

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

        if self.Es[s] != 0: # if ended.
            # terminal node
            return -self.Es[s] # end results of game: 1, -1, 1e-4.

        if s not in self.Ps:

            # # leaf node
            # # self.Ps[s]: 1d array, probability over all actions.
            self.Ps[s], v = self.agent.predict(canonicalBoard, to_prob=True)          
        

            # self.Ps[s], v = np.zeros(65), 0.5

            # a list of 0 and 1.
            valids = self.game.getValidMoves(canonicalBoard, 1)

            # self.Ps[s] = self.Ps[s] * valids  # masking invalid moves

            # sum_Ps_s = np.sum(self.Ps[s])
            # if sum_Ps_s > 0:
            #     self.Ps[s] /= sum_Ps_s  # renormalize
            # else:
            #     # if all valid moves were masked make all valid moves equally probable
            #     self.Ps[s] = self.Ps[s] + valids
            #     self.Ps[s] /= np.sum(self.Ps[s])

            self.Valids[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Valids[s]
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
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

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

 







class HumanTicTacToePlayer():
    def __init__(self, game, n):
        self.game = game
        self.n = n

    def play(self, board):
        boardvalues = np.arange(self.n*self.n*self.n).reshape(self.n,self.n,self.n)
        validvalue = np.arange(self.n*self.n*self.n)
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i] == 1:
                action = validvalue[i]
                # print(np.argwhere(boardvalues == action))

        while True:  
            a = input()  
            z,x,y = [int(x) for x in a.split(' ')]
            boardvalues = np.arange(self.n*self.n*self.n).reshape(self.n,self.n,self.n)
            a = boardvalues[z][x][y]
            if valid[a]:
                break
            else:
                print('Invalid')

        return a

class RandomPlayer():

    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a
 
class Arena(): 
    def __init__(self, player1, player2, game, display=None):
      
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display



    def playGame(self, verbose=False):
        
         
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
 
        while self.game.getGameEnded(board, curPlayer) == 0:

            it += 1

            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)

            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0: 
                assert valids[action] > 0

            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board) 
        return curPlayer * self.game.getGameEnded(board, curPlayer)


    def playGames(self, num, verbose=False): 

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num) :
            
            gameResult = self.playGame(verbose=verbose)
            print(f'playGame{_}, gameResult: {gameResult}')
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            
            gameResult = self.playGame(verbose=verbose)
            print(f'playGame{num+_}, gameResult: {gameResult}')
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws




# class Coach(): 

#     def __init__(self, game, agent, args):
#         self.game = game
#         self.agent = agent
#         self.args = args
#         self.mcts = MCTS(self.game, self.agent, self.args) 
#         # self.perBuf = perBuf 


#     def executeEpisode(self): 

#         trainExamples = []
#         board = self.game.getInitBoard() # an np array of shape 4x4x4
#         self.curPlayer = 1
#         episodeStep = 0

#         while True:
#             episodeStep += 1
            
#             canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)

#             temp = int(episodeStep < self.args.tempThreshold)

#             pi = self.mcts.getActionProb(canonicalBoard, temp=temp)

#             # data augmentation
#             sym = self.game.getSymmetries(canonicalBoard, pi)
#             for b, p in sym:
#                 trainExamples.append([b, self.curPlayer, p, None])

#             action = np.random.choice(len(pi), p=pi)
#             board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

#             r = self.game.getGameEnded(board, self.curPlayer)

#             if r != 0:
#                 # (canonicalBoard, pi, v)
#                 return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]



#     def learn(self): 

#         for i in range(1, self.args.numIters + 1):
#             # print(f'#### iter: {i} ####')

#             iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

#             for j in range(self.args.numEps):
#                 print(f'#### iter: {i}, eps: {j} ####')
#                 self.mcts = MCTS(self.game, self.agent, self.args)  # reset search tree
#                 iterationTrainExamples += self.executeEpisode()

#             self.trainExamplesHistory.append(iterationTrainExamples)


#             if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:                
#                 self.trainExamplesHistory.pop(0)
          
#             self.saveTrainExamples(i - 1)


#             trainExamples = []
#             for e in self.trainExamplesHistory:
#                 trainExamples.extend(e)


#             shuffle(trainExamples)
  
 
#             print('training...')
#             self.agent.train(trainExamples) 

#             if i % arenaCompareEveryN == 0:

#                 print('pitting random model...')

#                 nmcts = MCTS(self.game, self.agent, self.args)
#                 player1 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
#                 player2 = RandomPlayer(g).play
#                 arena = Arena(player1, player2, self.game)

#                 pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
    
#                 self.agent.save_checkpoint(folder=self.args.checkpoint,
#                                                     filename=f'temp.pth.tar')        
           

#     def saveTrainExamples(self, iteration):

#         folder = self.args.checkpoint

#         if not os.path.exists(folder):
#             os.makedirs(folder)
        
#         filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")

#         with open(filename, "wb+") as f:
#             Pickler(f).dump(self.trainExamplesHistory)

#         f.closed


#     def loadTrainExamples(self, path):

#         with open(path, "rb") as f:
#             self.trainExamplesHistory = Unpickler(f).load()
 
#     def getCheckpointFile(self, iteration):
#         return 'checkpoint_' + str(iteration) + '.pth.tar'


 




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


    def getBatch(self, i, bsz): 
        return self.buf[i*bsz:(i+1)*bsz]


    def sample(self):
        out = [i for i in self.buf]
        shuffle(out)
        return out

    def save(self, path): 
        with open(path, "wb+") as f:
            Pickler(f).dump(self.buf)

    def load(self, path)
        with open(path, "rb") as f:
            self.buf = Unpickler(f).load()
 



class Trainer():

    def __init__(self, agent, args):
        self.replayBuf = ReplayBuffer(2**16)
        self.game = Game(4)
        self.agent = agent
        self.args = args

    def collectExamples(self, nEps): 

        examples = []

        for j in range(nEps): 
            
            self.mcts = MCTS(self.game, self.agent, self.args) 
             
            board = self.game.getInitBoard()  
            self.curPlayer = 1
            t = 0
            tmpExamples = []

            while True:
                t += 1
                
                canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)

                temp = int(t < self.args.tempThreshold) 
                pi = self.mcts.getActionProb(canonicalBoard, temp=temp)

                # data augmentation
                sym = self.game.getSymmetries(canonicalBoard, pi)
                for b, p in sym:
                    tmpExamples.append([b, self.curPlayer, p, None])

                action = np.random.choice(len(pi), p=pi)
                board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

                r = self.game.getGameEnded(board, self.curPlayer)

                if r != 0:
                    # (canonicalBoard, pi, v)
                    examples.extend([(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in tmpExamples])

        return examples
        




    def train(self):

        for i in range():
 
            print('collecting examples...')
            self.replayBuf.addExamples(self.collectExamples()) 
            self.replayBuf.save(f'temp/iter-{i}.pth.tar.examples')
            examples = self.replayBuf.sample()
            

            print('training...')
            self.agent.train(examples) 



            if i % arenaCompareEveryN == 0:

                print('pitting random model...')

                nmcts = MCTS(self.game, self.agent, self.args)
                player1 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
                player2 = RandomPlayer(g).play
                arena = Arena(player1, player2, self.game)

                pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

          
                self.agent.save_checkpoint(folder=self.args.checkpoint,
                                                filename=f'temp.pth.tar')        



def train():



    agent = Agent() 
    # agent.load_checkpoint('./temp/checkpoint_17.h5')

    args = dotdict({
        'numIters': 100000,
        'numEps': numEps,       
        'tempThreshold': 10,  
        
        'updateThreshold': 0.5,
        'arenaCompare': arenaCompare,

        'numMCTSSims': 25,         
        'cpuct': 1,

        'checkpoint': './temp/',
        'load_model': False, 
        'numItersForTrainExamplesHistory': 5
    })
 

    trainer = Trainer(agent, args)





train()







class HumanTicTacToePlayer():
    def __init__(self, game, n):
        self.game = game
        self.n = n

    def play(self, board):
        boardvalues = np.arange(self.n*self.n*self.n).reshape(self.n,self.n,self.n)
        validvalue = np.arange(self.n*self.n*self.n)
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i] == 1:
                action = validvalue[i]
                # print(np.argwhere(boardvalues == action))

        while True:  
            a = input()  
            z,x,y = [int(x) for x in a.split(' ')]
            boardvalues = np.arange(self.n*self.n*self.n).reshape(self.n,self.n,self.n)
            a = boardvalues[z][x][y]
            if valid[a]:
                break
            else:
                print('Invalid')

        return a
 
def evaluate(): 

    g = Game(4)
 
    hp = HumanTicTacToePlayer(g, 4).play
    rp = RandomPlayer(g).play
 
    agent = Agent(g)
    agent.load_checkpoint('./temp', 'best.h5')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0}) 
    mcts1 = MCTS(g, agent, args1) 
    player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
 
    # player2 = hp
    player2 = rp
   
    arena = Arena(player1, player2, g, display=Game.display)

    # print(arena.playGames(2, verbose=True))
    pwins, nwins, draws = arena.playGames(10)
    print(f'pwins: {pwins} ,nwins: {nwins}, draws :{draws}')
   

# evaluate()



