import numpy as np
# import torch

from collections import deque
 
import numpy as np

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *





class TicTacToeNNet():
    def __init__(self, game, args):
        # game params
        self.board_z, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_z, self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_z, self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        
        h_conv1 = Activation('relu')(
            BatchNormalization(axis=3)(
                Conv3D(args.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        
        h_conv2 = Activation('relu')(
            BatchNormalization(axis=3)(
                Conv3D(args.num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        
        h_conv3 = Activation('relu')(
            BatchNormalization(axis=3)(
                Conv3D(args.num_channels, 3, padding='same')(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        
        h_conv4 = Activation('relu')(
            BatchNormalization(axis=3)(
                Conv3D(args.num_channels, 3, padding='valid')(h_conv3)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        
        h_conv4_flat = Flatten()(h_conv4)       

        s_fc1 = Dropout(args.dropout)(
            Activation('relu')(
                BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024

        s_fc2 = Dropout(args.dropout)(
            Activation('relu')(
                BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024

        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])

        # loss is categorical_crossentropy + mean_squared_error
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], 
                                        optimizer=Adam(args.lr))

 


class NNetWrapper():
    def __init__(self, game):


        args = dotdict({
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 64,
            'cuda': False,
            'num_channels': 512,
        })

        self.nnet = TicTacToeNNet(game, args)
        self.board_z, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples): 
        """
        examples: list of examples, each example is of form (board, pi, v), # board is canonicalBoard?
        """
        
        input_boards, target_pis, target_vs = list(zip(*examples))

        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        self.nnet.model.fit(
            x = input_boards, y = [target_pis, target_vs],
            batch_size = args.batch_size,
            epochs = args.epochs)



    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board, verbose=False)


        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)








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




class TicTacToeGame(Game):
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
        return self.n*self.n*self.n + 1

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
        return player*board


    def getSymmetries(self, board, pi):
        # mirror, rotational
        pi_board = np.reshape(pi[:-1], (self.n, self.n, self.n))
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
                    l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l
    


    def stringRepresentation(self, board): 
        return board.tostring()
 



class MCTS():
     
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
 

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
            self.Ps[s], v = self.nnet.predict(canonicalBoard)

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

 


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()



    def executeEpisode(self): 

        trainExamples = []
        board = self.game.getInitBoard() # an np array of shape 4x4x4
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)

            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)

            # data augmentation
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                # (canonicalBoard, pi, v)
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]



    def learn(self): 

        for i in range(1, self.args.numIters + 1):
            print(f'iter: {i}')

            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            for i in range(self.args.numEps):
                self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                iterationTrainExamples += self.executeEpisode()

            # save the iteration examples to the history 
            self.trainExamplesHistory.append(iterationTrainExamples)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)


            shuffle(trainExamples)
 
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        
            self.nnet.train(trainExamples)
         




game = Game(4)
nnet = NNetWrapper(game)
args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False, 
    'numItersForTrainExamplesHistory': 20
})


c = Coach(game, nnet, args) 
c.learn()






















# class Node:
    
  
#     def __init__(self, game, reward, parent, player, canonicalBoard, a):
          
#         self.child = None
        
#         # self.T = 0
#         self.N = 0        
#         self.game = game
        
#         self.player = player
#         self.canonicalBoard = canonicalBoard
#         self.validActions = self.game.getValidMoves(self.canonicalBoard)

#         self.reward = reward
#         self.isTerminal = (reward != 0)
#         self.parent = parent
#         self.a = a

        
#         self.Qsa = 0  
#         self.Nsa = 0

#         self.Pa = []
#         self.T = 0
        

    

        
#     def getUCBscore(self, a):
        
#         # if self.N == 0:
#         #     return float('inf')
        
#         # top_node = self
#         # if top_node.parent:
#         #     top_node = top_node.parent
            
#         # return (self.T / self.N) + c * sqrt(log(top_node.N) / self.N) 
#         return self.Qsa[a] + self.args.cpuct * self.Pa[a] * math.sqrt(self.Ns[s]) / (
#                             1 + self.Nsa[a])


#     def create_child(self):
       
#         if self.isTerminal: return

#         child = {} 
#         for a in self.validActions:
             
#             # always take action for player 1 (canonical player).
#             next_board, reward = self.game.step(self.canonicalBoard, a)
            
#             # (game, reward, parent, player, observation, action_index)
#             child[a] = Node(self.game, reward, self, self.player*(-1), next_board*(-1), a)                        
            
#         self.child = child



#     def search(self):
        
#         if self.isTerminal: return

#         # s = self.game.stringRepresentation(self.canonicalBoard)
 
 
#         # cur_best = -float('inf')
#         # best_act = -1

#         # # pick the action with the highest upper confidence bound
#         # for a in range(self.game.getActionSize()):
#         #     if self.validActions[a]:
                 

#         #         if a in self.child:
#         #             u = self.Qsa[a] + self.args.cpuct * self.Pa[a] * math.sqrt(self.N) / (1 + self.Nsa)
#         #         else:
#         #             u = self.args.cpuct * self.Pa[a] * math.sqrt(self.N + EPS)   

#         #         if u > cur_best:
#         #             cur_best = u
#         #             best_act = a

 
#         # a = best_act

#         # # always take action for player 1 (canonical player).
#         # next_board, reward = self.game.step(self.canonicalBoard, a)
        
#         # # (game, reward, parent, player, observation, action_index)
#         # self.child[a] = Node(self.game, reward, self, self.player*(-1), next_board*(-1), a)  
                 

#         current = self
        
#         while current.child: # phase 1: select

#             child = current.child
            
#             max_U = max(c.getUCBscore() for c in child.values()) 
#             actions = [ a for a,c in child.items() if c.getUCBscore() == max_U ]

#             if len(actions) == 0:
#                 print("error zero length ", max_U)                      
#             action = random.choice(actions)
#             current = child[action]


#         # # phase 2&3: expand & simulate
#         # if current.N < 1: 
#         #     # current.T = current.T + current.rollout()
#         #     current.rollout()  
#         # else: 


#         # phase 2 & 3: expand & simulate
#         if current.N < 1:
#             # current.T = current.T + current.rollout() 
#             current.rollout() 
#         else:       
#             if (not this.isTerminal):   

#                 current.create_child()
#                 cur_best = -float('inf')
#                 best_act = -1
#                 for a in current.child:  
#                     u = self.args.cpuct * self.Pa[a] * math.sqrt(self.N + EPS)  # Q = 0 ?
#                     if u > cur_best:
#                         cur_best = u
#                         best_act = a

#                 a = best_act
#                 current = current.child[a]
#                 current.rollout()
            
#             else: # this node is terminal state

#         current.N += 1


#         # phase 4: backprop
            
#         # parent = current 
#         # while parent.parent: 
#         #     parent = parent.parent
#         #     parent.N += 1
#         #     parent.T = parent.T + current.T


#         parent = current
#         if(current.isTerminal):
#             v = current.reward
#         else:
#             v = current.T

#         while parent.parent:

#             v = -v
            
#             a = parent.a
#             parent = parent.parent

#             parent.Qsa[a] = (parent.Nsa[a] * parent.Qsa[a] + v) / (parent.Nsa[a] + 1)
#             parent.Nsa[a] += 1

            


#         # current = self
        
#         # while current.child:

#         #     child = current.child
#         #     max_U = max(c.getUCBscore() for c in child.values())
#         #     actions = [ a for a,c in child.items() if c.getUCBscore() == max_U ]
#         #     if len(actions) == 0:
#         #         print("error zero length ", max_U)                      
#         #     action = random.choice(actions)
#         #     current = child[action]
     
#         # if current.N < 1:
#         #     current.T = current.T + current.rollout()
#         # else:
#         #     current.create_child()
#         #     if current.child:
#         #         current = random.choice(current.child)
#         #     current.T = current.T + current.rollout()
            
#         # current.N += 1      
        
#         # parent = current
            
#         # while parent.parent:
            
#         #     parent = parent.parent
#         #     parent.N += 1
#         #     parent.T = parent.T + current.T


#     def rollout(self):
    
#         if self.isTerminal:
#             return 0        
          
#         Pa, T = nnet.predict(self.canonicalBoard)
#         self.Pa = Pa
#         self.T = T

        
#         self.Pa = self.Pa * self.validActions
#         sum_Pa = np.sum(self.Pa)
#         if sum_Pa > 0:
#             self.Pa /= sum_Pa  # re-normalize
#         else:
#             # if all valid moves were masked make all valid moves equally probable
#             self.Pa = self.Pa + self.validActions
#             self.Pa /= np.sum(self.Pa)

#         # return T

#         # v = 0

#         # isTerminal = False
#         # new_game = deepcopy(self.game)
#         # while not isTerminal:
#         #     action = new_game.action_space.sample()
#         #     observation, reward, isTerminal, _ = new_game.step(action)
#         #     v = v + reward
#         #     if isTerminal:
#         #         new_game.reset()
#         #         new_game.close()
#         #         break             
#         # return v


#     def next(self):
     
#         if self.isTerminal:
#             raise ValueError("game has ended")

#         if not self.child:
#             raise ValueError('no children found and game hasn\'t ended')
        
#         child = self.child
        
#         max_N = max(node.N for node in child.values())
       
#         max_children = [ c for a,c in child.items() if c.N == max_N ]
        
#         if len(max_children) == 0:
#             print("error zero length ", max_N) 
            
#         max_child = random.choice(max_children)
        
#         return max_child, max_child.action_index        




# def getActionProb(canonicalBoard):
 
    #            (game, reward, parent, player, observation, action_index):
    mctree = Node(game, 0, None, 1, canonicalBoard, None)
    
    for i in range(20):        
        mctree.search()
        



    # next_tree, next_action = mytree.next()
        
    # next_tree.detach_parent()
    
    # return next_tree, next_action        