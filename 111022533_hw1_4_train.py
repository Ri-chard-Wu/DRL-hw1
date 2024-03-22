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

numEps = 50 #50
batch_size = 64
perBuf_size = 2**17 #2**16
epochs = 5 # 5
arenaCompare = 16 # 16
epsilon = 0.4
gamma = 0.99
# save_every_n_iter = 1

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class PrioritizeExperienceReplayBuffer():

    def __init__(self):
        self.buf = []
        self.priorities = [] 
        self.maxLength = perBuf_size
        self.pendingUpdate = False
        self.beta = 1.0
    
    def addExamples(self, priorities, examples):
        
        if(self.pendingUpdate): return
 
        excess = len(self.buf) + len(examples) - self.maxLength

        if(excess > 0): 
            tmp = []
            tmp.extend(self.buf[excess:])
            self.buf = tmp  

            tmp = []
            tmp.extend(self.priorities[excess:])
            self.priorities = tmp
        
        self.buf.extend(examples)
        self.priorities.extend(priorities)
            
    def getExampleNum(self):
        return len(self.buf)

    def getTopPriorityExamples(self, n):
        
        self.pendingUpdate = True

        probs = list(np.array(self.priorities) / sum(self.priorities))
         
        # self.ids = sorted(range(len(probs)), key=lambda i: probs[i])[-n:]        
        self.ids = np.random.choice(len(probs), n, p=probs)
        
        probs_top = tf.clip_by_value([probs[i] for i in self.ids], 0.000001, 1.0)

        learning_weights = (1/(probs_top * len(probs)))**self.beta
        learning_weights = tf.cast(learning_weights, tf.float32)

        return [self.buf[i] for i in self.ids], learning_weights
 

    def updatePriorities(self, tdLosses):
        
        if(not self.pendingUpdate): return

        for i, tdLoss in enumerate(tdLosses):
            idx = self.ids[i]
            self.priorities[idx] = tdLoss
        
        self.pendingUpdate = False





class TicTacToeNNet(tf.keras.Model):

    def __init__(self, game, args):
  
        super(TicTacToeNNet, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(args.lr)

        self.action_size = game.getActionSize()
        self.args = args

        self.act = {}
        self.bn = {}
        self.conv = {}
        padding = ['same', 'same', 'same', 'valid']

        for i in range(4):
            self.act[i] = tf.keras.layers.Activation('relu')
            self.bn[i] = tf.keras.layers.BatchNormalization(axis=3)
            self.conv[i] = tf.keras.layers.Conv3D(args.num_channels, 3, padding=padding[i])

 
        self.flatten = tf.keras.layers.Flatten()
 
        self.dropout1 = tf.keras.layers.Dropout(args.dropout)
        self.act1 = tf.keras.layers.Activation('relu')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)
        self.fc1 = tf.keras.layers.Dense(1024)  

        self.dropout2 = tf.keras.layers.Dropout(args.dropout)
        self.act2 = tf.keras.layers.Activation('relu')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1)
        self.fc2 = tf.keras.layers.Dense(512)  

 
        # self.fc_a = tf.keras.layers.Dense(self.action_size, activation='softmax', name='pi') 

        self.fc_a = tf.keras.layers.Dense(self.action_size) 

        # self.fc_v = tf.keras.layers.Dense(1, activation='tanh', name='v') 
        
    
    # def max_Q(self, state):
    #     # Q(s,a,theta) for all a, shape (batch_size, num_action)
    #     output = self(state)

    #     # max(Q(s',a',theta')), shape (batch_size, 1)
    #     return tf.reduce_max(output, axis=1)

    @tf.function
    def call(self, x, training=None):
 
        x = tf.expand_dims(x, axis=4)
 
        for i in range(4):
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

        qa = self.fc_a(x)
        # v = self.fc_v(x)

        # return prob_a, v
        return qa
 
 

    @tf.function
    def train_step(self, data, learning_weights): 
      
        s, a, r, q_tar = data 

        with tf.GradientTape() as tape:
            
            qa = self(s)

            index = tf.stack([tf.range(tf.shape(a)[0]), a], axis=1)
            q = tf.gather_nd(qa, index)
             
            # td_loss = tf.keras.metrics.mean_squared_error(r + self.args.gamma * q_tar, q)

            td_losses = tf.square(r + self.args.gamma * q_tar - q)
            td_loss = tf.reduce_mean(td_losses * learning_weights)
            
        gradients = tape.gradient(td_loss, self.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return td_losses
        # return total_loss, td_loss
            




class NNetWrapper():
    def __init__(self, game):
 

        self.args = dotdict({
            'lr': 0.001,
            'gamma': gamma,
            'dropout': 0.3,
            'epochs': epochs,
            'batch_size': batch_size,
            'cuda': False,
            'num_channels': 256,
        }) 

        self.game = game

        self.model = TicTacToeNNet(game, self.args)

        self.board_z, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
 


    def train(self, perBuf):   
        
        batch_size = self.args.batch_size
 
        td_losses_all = []
        

        for epoch in range(self.args.epochs):  

            n = int(perBuf.getExampleNum() / batch_size) 
            for i in range(n):

                # print(f'epoch: {epoch}, n: {i} / {n}')

                examples, learning_weights = perBuf.getTopPriorityExamples(batch_size)
                
                s, a, r, s_next, valids = list(zip(*examples))

                q_tar = []   
                _s_next = tf.convert_to_tensor(s_next, dtype=tf.float32) 
                qa = self.model(_s_next).numpy()   
                 
                # print(f'max: {np.max(qa)}, min: {np.min(qa)}')

                for b, _s in enumerate(s_next): 
                    _valids = valids[b]
                    if(sum(_valids) > 0.5):
                            
                        tmp = np.array([qa[b][j] if _valids[j] else np.inf \
                                                        for j in range(len(_valids))])

                        _q_tar = -min(tmp)
                        assert(not np.isinf(_q_tar))
                        q_tar.append(_q_tar) 
                    else: # terminal state
                        q_tar.append(0.0)
                
                # print(f'q_tar: {q_tar}')

                s = tf.convert_to_tensor(s, dtype=tf.float32)
                a = tf.convert_to_tensor(a, dtype=tf.int32)
                r = tf.convert_to_tensor(r, dtype=tf.float32) 
                q_tar = tf.convert_to_tensor(q_tar, dtype=tf.float32) 

                td_losses = self.model.train_step((s, a, r, q_tar), learning_weights)
                td_losses_all.append(np.mean(td_losses.numpy()))             
                perBuf.updatePriorities(td_losses.numpy())
        
        K.clear_session()
        tf.keras.backend.clear_session() 

        return np.mean(td_losses_all)


    def predict(self, canonicalBoard): 

        board = canonicalBoard[np.newaxis, :, :] 
        board = tf.convert_to_tensor(board, dtype=tf.float32)
        
        qa = self.model(board) 
        
        K.clear_session()
        tf.keras.backend.clear_session() 

        qa = qa[0].numpy() 

        valids = self.game.getValidMoves(canonicalBoard, 1)  
        qa = np.array([qa[i] if valids[i] else None for i in range(len(valids))])
 
        return qa



    def toProb(self, qa): 
        return np.exp(qa) / np.sum(np.exp(qa), axis=0)


    def act(self, canonicalBoard):
        
        qa = self.predict(canonicalBoard)
        qa = np.array([-np.inf if qa[i] is None else qa[i] for i in range(len(qa))])                  
        pi = self.toProb(qa)

        action = np.random.choice(len(pi), p=pi)

        return action


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        
        print('saved ckpt')

        filename = filename.split(".")[0] + ".h5"
        
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        self.model.save_weights(filepath)


    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        
        filename = filename.split(".")[0] + ".h5"
        
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))

        # need call once to enable load weights.
        self.model(tf.random.uniform(shape=[1,4,4,4]))
        self.model.load_weights(filepath)





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

        (z,y,x) = move

        # Add the piece to the empty square.
        assert self.pieces[z][y][x] == 0
        self.pieces[z][y][x] = color




class TicTacToeGame():
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
        # if action == self.n*self.n*self.n:
        #     return (board, -player)

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

        # if len(legalMoves)==0:
        #     valids[-1]=1
        #     return np.array(valids)

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


    # def getSymmetries(self, board, pi):
    #     # mirror, rotational
    #     pi_board = np.reshape(pi, (self.n, self.n, self.n))
    #     l = []
    #     newB = np.reshape(board, (self.n*self.n, self.n))
    #     newPi = pi_board
    #     for i in range(1,5):

    #         for z in [True, False]:
    #             for j in [True, False]:
    #                 if j:
    #                     newB = np.fliplr(newB)
    #                     newPi = np.fliplr(newPi)
    #                 if z:
    #                     newB = np.flipud(newB)
    #                     newPi = np.flipud(newPi)
                    
    #                 newB = np.reshape(newB, (self.n,self.n,self.n))
    #                 newPi = np.reshape(newPi, (self.n,self.n,self.n))
    #                 l += [(newB, list(newPi.ravel()) + [pi[-1]])]
    #     return l
    

    def getSymmetries(self, board, a, board_next):
        # mirror, rotational
        pi = np.zeros(self.n*self.n*self.n)
        pi[a] = 1

        pi_board = np.reshape(pi, (self.n, self.n, self.n))
        l = []
        
        newB = np.reshape(board, (self.n*self.n, self.n))
        newB_next = np.reshape(board_next, (self.n*self.n, self.n))

        newPi = pi_board
        for i in range(1,5):

            for z in [True, False]:
                for j in [True, False]:
                    if j:
                        newB = np.fliplr(newB)
                        newB_next = np.fliplr(newB_next)
                        newPi = np.fliplr(newPi)
                    if z:
                        newB = np.flipud(newB)
                        newB_next = np.flipud(newB_next)
                        newPi = np.flipud(newPi)
                    
                    newB = np.reshape(newB, (self.n,self.n,self.n))
                    newB_next = np.reshape(newB_next, (self.n,self.n,self.n))
                    newPi = np.reshape(newPi, (self.n,self.n,self.n))
                     
                    l += [(newB, np.argmax(list(newPi.ravel())), newB_next)]
        return l




    

    def stringRepresentation(self, board): 
        return board.tostring()
 


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


# class MCTS():
     
#     def __init__(self, game, nnet, args):
#         self.game = game
#         self.nnet = nnet
#         self.args = args
#         self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
#         self.Nsa = {}  # stores #times edge s,a was visited
#         self.Ns = {}  # stores #times board s was visited
        
#         self.Ps = {}  # stores initial policy (returned by neural net)
        

#         self.Es = {}  # stores game.getGameEnded ended for board s
#         self.Valids = {}  # stores game.getValidMoves for board s
 

#     def getActionProb(self, canonicalBoard, temp=1):
      
#         for i in range(self.args.numMCTSSims):
#             self.search(canonicalBoard)

#         s = self.game.stringRepresentation(canonicalBoard)
#         counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        

#         if temp == 0: # yes for pit, no for train
#             bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
#             bestA = np.random.choice(bestAs)
#             probs = [0] * len(counts)
#             probs[bestA] = 1
#             return probs

#         counts = [x ** (1. / temp) for x in counts]
#         counts_sum = float(sum(counts))
#         probs = [x / counts_sum for x in counts]
#         # return probs, v
#         return probs

 

#     def search(self, canonicalBoard): 

#         s = self.game.stringRepresentation(canonicalBoard)

#         if s not in self.Es:
#             self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

#         if self.Es[s] != 0: # if ended.
#             # terminal node
#             return -self.Es[s] # end results of game: 1, -1, 1e-4.

#         if s not in self.Ps:

#             # # leaf node
#             # # self.Ps[s]: 1d array, probability over all actions.
#             self.Ps[s], v = self.nnet.predict(canonicalBoard)          
        

#             # self.Ps[s], v = np.zeros(65), 0.5

#             # a list of 0 and 1.
#             valids = self.game.getValidMoves(canonicalBoard, 1)

#             # self.Ps[s] = self.Ps[s] * valids  # masking invalid moves

#             # sum_Ps_s = np.sum(self.Ps[s])
#             # if sum_Ps_s > 0:
#             #     self.Ps[s] /= sum_Ps_s  # renormalize
#             # else:
#             #     # if all valid moves were masked make all valid moves equally probable
#             #     self.Ps[s] = self.Ps[s] + valids
#             #     self.Ps[s] /= np.sum(self.Ps[s])

#             self.Valids[s] = valids
#             self.Ns[s] = 0
#             return -v

#         valids = self.Valids[s]
#         cur_best = -float('inf')
#         best_act = -1

#         # pick the action with the highest upper confidence bound
#         for a in range(self.game.getActionSize()):
#             if valids[a]:
#                 if (s, a) in self.Qsa:
#                     # 1st term "self.Qsa[(s, a)]": encourage exploitation.
#                     # 2nd term "(...) / (1 + self.Nsa[(s, a)])": encourage exploration.
#                     u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
#                             1 + self.Nsa[(s, a)])
#                 else:
#                     u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

#                 if u > cur_best:
#                     cur_best = u
#                     best_act = a

#         a = best_act

#         # always get next state as player 1 (canonical).
#         next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)

#         next_s = self.game.getCanonicalForm(next_s, next_player)

#         v = self.search(next_s)

#         if (s, a) in self.Qsa:
#             # avg of all self.Nsa[(s, a)] + 1 Qsa's.
#             self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
#             self.Nsa[(s, a)] += 1
#         else:
#             self.Qsa[(s, a)] = v
#             self.Nsa[(s, a)] = 1

#         self.Ns[s] += 1
#         return -v

 



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
            print(f'playGame{_}')
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            print(f'playGame{num+_}')
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws





class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, perBuf, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        # self.mcts = MCTS(self.game, self.nnet, self.args)
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        self.perBuf = perBuf


    def executeEpisode(self): 

        trainExamples = {1: [], -1: []}
        board = self.game.getInitBoard() # an np array of shape 4x4x4
        self.curPlayer = 1
        episodeStep = 0 

        canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)

        while True: 

            episodeStep += 1 

            qa = self.nnet.predict(canonicalBoard) # all are valid actions
            qa = np.array([-np.inf if qa[i] is None else qa[i] for i in range(len(qa))])            
            pi = self.nnet.toProb(qa) # invalid actions have 0 prob.
            
            if np.random.rand() < epsilon:
                a = np.random.choice(len(pi), p=pi)   
            else: 
                a = np.argmax(pi) 
  
            nextBoard, nextPlayer = self.game.getNextState(board, self.curPlayer, a)
            r = self.game.getGameEnded(nextBoard, nextPlayer)

            nextCanonicalBoard = self.game.getCanonicalForm(nextBoard, nextPlayer)

            trainExamples[self.curPlayer].append([canonicalBoard, qa[a], a, nextCanonicalBoard, r])
            
            self.curPlayer = nextPlayer
            board = nextBoard
            canonicalBoard = nextCanonicalBoard


            if r != 0:  
                examples = []
                priorities = []
                for player in trainExamples:
                    
                    gamma = self.nnet.args.gamma
                    reward = r * ((-1) ** (player != self.curPlayer))

                    for x in reversed(trainExamples[player]):
                        
                        s = x[0]
                        q = x[1]
                        a = x[2]
                        s_next = x[3]
                        end = x[4]
                         

                        if(end != 0): # s_next is an end state.
                            q_tar = 0
                        else:
                            # opponent's lose is our gain.
                            qa_next = self.nnet.predict(s_next)
                            q_tar = -min(np.array([np.inf if q is None else q for q in qa_next])) 

                        
                        assert(not np.isinf(q_tar))

                        # data augmentation
                        sym = self.game.getSymmetries(s, a, s_next)
                        for s, a, s_next in sym: 
                            valids = self.game.getValidMoves(s_next, 1)  
                            examples.append((s, a, reward, s_next, valids))  
                            tdLoss = (reward + gamma * q_tar - q)**2  
                            priorities.append(tdLoss)  

                        reward = gamma * reward

                idx = [i for i in range(len(examples))]
                shuffle(idx)
                examples = [examples[i] for i in idx]
                priorities = [priorities[i] for i in idx]
                return examples, priorities
 

    def learn(self): 

        for i in range(1, self.args.numIters + 1): 
            print('####################')
            print(f'iter: {i}') 

            examples, priorities = [], []

            print('collecting data...')
            for j in range(self.args.numEps):
                # print(f'#### iter: {i}, eps: {j} ####') 
                # self.mcts = MCTS(self.game, self.nnet, self.args) 
                e, p = self.executeEpisode()
                examples += e
                priorities += p 

            self.perBuf.addExamples(priorities, examples)
          
            
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
           
            print('training...')
            loss = self.nnet.train(self.perBuf)
            print(f'loss: {loss}') 
          
            print('pitting with old model...') 
            arena = Arena(lambda x: self.nnet.act(x),
                          lambda x: self.pnet.act(x), self.game)
 

            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
 
            print(f'win rate: {float(nwins) / (pwins + nwins)}')

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')




    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'


 



def train():



    game = TicTacToeGame(4)
    perBuf = PrioritizeExperienceReplayBuffer()
    nnet = NNetWrapper(game)
    

    nnet.load_checkpoint('temp', 'checkpoint_11.h5')

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



    c = Coach(game, nnet, perBuf, args) 
    c.learn()





train()







# class HumanTicTacToePlayer():
#     def __init__(self, game, n):
#         self.game = game
#         self.n = n

#     def play(self, board):
#         boardvalues = np.arange(self.n*self.n*self.n).reshape(self.n,self.n,self.n)
#         validvalue = np.arange(self.n*self.n*self.n)
#         # display(board)
#         valid = self.game.getValidMoves(board, 1)
#         for i in range(len(valid)):
#             if valid[i] == 1:
#                 action = validvalue[i]
#                 # print(np.argwhere(boardvalues == action))

#         while True:  
#             a = input()  
#             z,x,y = [int(x) for x in a.split(' ')]
#             boardvalues = np.arange(self.n*self.n*self.n).reshape(self.n,self.n,self.n)
#             a = boardvalues[z][x][y]
#             if valid[a]:
#                 break
#             else:
#                 print('Invalid')

#         return a


# def evaluate():

#     human_vs_cpu = True

    
#     g = TicTacToeGame(4)
 
#     hp = HumanTicTacToePlayer(g, 4).play

#     # nnet players
#     n1 = NNetWrapper(g)
#     n1.load_checkpoint('./temp', 'iter5-710.h5')
#     args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
 
#     mcts1 = MCTS(g, n1, args1) 

#     n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
 
#     player2 = hp
   
#     arena = Arena(n1p, player2, g, display=TicTacToeGame.display)

#     print(arena.playGames(2, verbose=True))

# evaluate()




