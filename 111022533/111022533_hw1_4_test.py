
import numpy as np  
import math 

from random import shuffle
import tensorflow.keras.backend as K
import tensorflow as tf

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

   
class Game():
    actionSize = 64
    shape = (4,4,4)
    coord2actionId = np.arange(actionSize).reshape(shape)
  
    @staticmethod
    def getInitBoard(): 
        return np.zeros(Game.shape)
 
    @staticmethod
    def getNextState(canonBoard, action): 
 
        board = np.copy(canonBoard) 
        (z,y,x) = np.argwhere(Game.coord2actionId==action)[0]         
        assert board[z][y][x] == 0
        board[z][y][x] = 1
 
        return -1*board # flip to canon for next player


    @staticmethod
    def getValidMoves(canonBoard):
        
        valids = [0]*Game.actionSize   
        n = Game.shape[0] 

        moves = [] 
        for z in range(n): 
            for y in range(n):
                for x in range(n):
                    if canonBoard[z][y][x]==0: 
                        moves.append((z,y,x))
        
        for z, y, x in moves:
            valids[Game.coord2actionId[z][y][x]] = 1

        return np.array(valids)

    @staticmethod
    def getGameEnded(canonBoard):       

        if Game.is_win(canonBoard):
            return 1
        
        if Game.is_win(-1*canonBoard):
            return -1

        if Game.has_legal_moves(canonBoard):
            return 0

        # draw has a very little value 
        return 1e-4

    @staticmethod
    def has_legal_moves(board):
        n = Game.shape[0]
        for z in range(n):
            for y in range(n):
                for x in range(n):
                    if board[z][x][y]==0:
                        return True
        return False


    @staticmethod
    def is_win(canonBoard): 

        n = Game.shape[0]
        win = n

        # check z-dimension
        for z in range(n):
            for y in range(n):
                count = 0
                for x in range(n):
                    if canonBoard[z,x,y]==1:
                        count += 1
                if count==win:
                    return True

        for z in range(n):
            for x in range(n):
                count = 0
                for y in range(n):
                    if canonBoard[z,x,y]==1:
                        count += 1
                if count==win:
                    return True
        
        # check x dimension
        for x in range(n):
            for z in range(n):
                count = 0
                for y in range(n):
                    if canonBoard[z,x,y]==1:
                        count += 1
                if count==win:
                    return True

        for x in range(n):
            for y in range(n):
                count = 0
                for z in range(n):
                    if canonBoard[z,x,y]==1:
                        count += 1
                if count==win:
                    return True

        # check y dimension
        for y in range(n):
            for x in range(n):
                count = 0
                for z in range(n):
                    if canonBoard[z,x,y]==1:
                        count += 1
                if count==win:
                    return True
        
        for y in range(n):
            for z in range(n):
                count = 0
                for x in range(n):
                    if canonBoard[z,x,y]==1:
                        count += 1
                if count==win:
                    return True
        
        # check flat diagonals
        # check z dimension
        for z in range(n):
            count = 0
            for d in range(n):
                if canonBoard[z,d,d]==1:
                    count += 1
            if count==win:
                return True
        
      
        for z in range(n):
            count = 0
            for d in range(n):
                if canonBoard[z,d,n-d-1]==1:
                    count += 1
            if count==win:
                return True

        # check x dimension 
        for x in range(n):
            count = 0
            for d in range(n):
                if canonBoard[d,x,d]==1:
                    count += 1
            if count==win:
                return True
 
        for x in range(n):
            count = 0
            for d in range(n):
                if canonBoard[d,x,n-d-1]==1:
                    count += 1
            if count==win:
                return True

        # check y dimension 
        for y in range(n):
            count = 0
            for d in range(n):
                if canonBoard[d,d,y]==1:
                    count += 1
            if count==win:
                return True

       
        for y in range(n):
            count = 0
            for d in range(n):
                if canonBoard[n-d-1,d,y]==1:
                    count += 1
            if count==win:
                return True
        
        # check 4 true diagonals
        count = 0
        if canonBoard[0,0,0] == 1:
            count += 1
            if canonBoard[1,1,1] == 1:
                count += 1
                if canonBoard[2,2,2] == 1:
                    count += 1
                    if count == win:
                        return True
            
        count = 0
        if canonBoard[2,0,0] == 1:
            count += 1
            if canonBoard[1,1,1] == 1:
                count += 1
                if canonBoard[0,2,2] == 1:
                    count += 1
                    if count == win:
                        return True
        
        count = 0
        if canonBoard[2,2,0] == 1:
            count += 1
            if canonBoard[1,1,1] == 1:
                count += 1
                if canonBoard[0,0,2] == 1:
                    count += 1
                    if count == win:
                        return True
        
        count = 0
        if canonBoard[0,2,0] == 1:
            count += 1
            if canonBoard[1,1,1] == 1:
                count += 1
                if canonBoard[2,0,2] == 1:
                    count += 1
                    if count == win:
                        return True

        # return false if no 3 is reached
        return False


    @staticmethod
    def getSymmetries(board, pi):
        # mirror, rotational
        n = Game.shape[0]
 
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
    

    @staticmethod
    def hash(board): 
        return board.tostring()
 



class Model(tf.keras.Model):

    def __init__(self):
  
        super(Model, self).__init__() 

        self.act = {}
        self.bn = {}
        self.conv = {}    
        padding = ['same', 'same', 'valid']
        n_filter = 256

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
 
            a_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(a_prob, a_prob_pred))
            v_loss = tf.reduce_mean(tf.square(v - v_pred))
 
            total_loss = (a_loss + v_loss)

  
        gradients = tape.gradient(total_loss, self.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
        return total_loss
    

    def train_step(self, batch):
        
        x, a_prob, v = list(zip(*batch))

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        a_prob = tf.convert_to_tensor(a_prob, dtype=tf.float32)
        v = tf.convert_to_tensor(v, dtype=tf.float32)

        return self._train_step((x, a_prob, v)).numpy()
        
  
    def _predict(self, canonBoard): 

        board = canonBoard[np.newaxis, :, :] 
        board = tf.convert_to_tensor(board, dtype=tf.float32) 
        pi, v = self(board) 
        
        pi = pi[0].numpy()
        v = v[0].numpy()

        K.clear_session()
        tf.keras.backend.clear_session() 

        valids = Game.getValidMoves(canonBoard)

        pi = pi * valids  # masking invalid moves

        sum_Ps_s = np.sum(pi)
        if sum_Ps_s > 0:
            pi /= sum_Ps_s  # renormalize
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



agent_para = dotdict({   
    'nSims': 25,  
    'cpuct':1.0
})


class Agent(Model):
     
    def __init__(self):
        super().__init__()  

        self.para = agent_para     
        self.reset()

    def set_n_sim(self, n):
        self.para.nSims = n

    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited 
        self.Ps = {}  # stores initial policy (returned by neural net) 
        self.Es = {}  # stores game.getGameEnded ended for board s
       
    def choose_action(self, state):

        player = state[0]
        board = state[1:] 
        canonBoard = board.reshape(Game.shape) * player
        
        a = np.argmax(self.predict(canonBoard))
 
        (z,y,x) = np.argwhere(Game.coord2actionId == a)[0] 
        assert canonBoard[z][y][x] == 0

        return [x, y, z]
    
 
    def load_policy(self):    
        self(tf.random.uniform(shape=[1,4,4,4]))
        self.load_weights('./111022533/111022533_hw1_4_data')
        

    def predict(self, canonBoard, temp=1):
        """
            canonBoard: cannot be terminal state.
        """
        for i in range(self.para.nSims):
            self.search(canonBoard)

        s = Game.hash(canonBoard)
        # counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(Game.actionSize)]
        counts = [self.Nsa[s][a] if a in self.Nsa[s] else 0 for a in range(Game.actionSize)]
 
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
 

    def search(self, canonBoard): 
        """
        select & expand-> simulate -> backprop
        """
        trajectory, endCanonicalBoard = self.select_expand(canonBoard)
        r = self.simulate(endCanonicalBoard) 
        self.backprop(trajectory, r)


    def select_expand(self, canonBoard):

        trajectory = []
        while(True): 

            s = Game.hash(canonBoard)
            if s not in self.Ps: # terminal or first visit.
                return trajectory, canonBoard
            
            a = self.max_a(s) 
            trajectory.append((s, a))

            canonBoard = Game.getNextState(canonBoard, a) 
          
       
    def simulate(self, canonBoard):
        
        s = Game.hash(canonBoard)

        if s not in self.Es:  
            self.Es[s] = Game.getGameEnded(canonBoard)
        if self.Es[s] != 0: # terminal
            return self.Es[s]
 
        # first visit
        self.Ps[s], v = self._predict(canonBoard)          
        self.Qsa[s] = {}
        self.Nsa[s] = {}

        valids = Game.getValidMoves(canonBoard) 
        
        for a in range(Game.actionSize):
            if valids[a]:
                self.Qsa[s][a] = 0
                self.Nsa[s][a] = 0

        self.Ns[s] = 0
        return v


    def backprop(self, trajectory, r):
        for s, a in reversed(trajectory):
            r *= -1
            self.Qsa[s][a] = (self.Nsa[s][a] * self.Qsa[s][a] + r) / (self.Nsa[s][a] + 1)
            self.Nsa[s][a] += 1 
            self.Ns[s] += 1


    def max_a(self, s):
        """ 
            - pick the action with the highest upper confidence bound
            - 1st term "self.Qsa[s][a]": encourage exploitation.
            - 2nd term "(...) / (1 + self.Nsa[s][a])": encourage exploration.
        """
        cur_best = -float('inf')
        a_best = -1        
        for a in self.Qsa[s].keys(): # all are valids actions.
            u = self.Qsa[s][a] + self.para.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8) / (1 + self.Nsa[s][a])
            if u > cur_best:
                cur_best = u
                a_best = a
        
        return a_best