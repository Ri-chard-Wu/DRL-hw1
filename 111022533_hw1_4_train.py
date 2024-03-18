import numpy as np
import torch


from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .TicTacToeLogic import Board
import numpy as np

"""
Game class implementation for the game of 3D TicTacToe or Qubic.

Author: Adam Lawson, github.com/goshawk22
Date: Feb 05, 2020

Based on the TicTacToeGame by Evgeny Tyurin.
"""
class TicTacToe3D():
    def __init__(self):
        self.n = 4

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

        # board: an np array of shape 4x4x4

        # return state if player==1, else return -state if player==-1
        return player * board


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
        # 8x8 numpy array (canonical board)
        return board.tostring()

    # @staticmethod
    # def display(board):
    #     n = board.shape[0]
    #     for z in range(n):
    #         print("   ", end="")
    #         for y in range(n):
    #             print (y,"", end="")
    #         print("")
    #         print("  ", end="")
    #         for _ in range(n):
    #             print ("-", end="-")
    #         print("--")
    #         for y in range(n):
    #             print(y, "|",end="")    # print the row #
    #             for x in range(n):
    #                 piece = board[z][y][x]    # get the piece to print
    #                 if piece == -1: print("X ",end="")
    #                 elif piece == 1: print("O ",end="")
    #                 else:
    #                     if x==n:
    #                         print("-",end="")
    #                     else:
    #                         print("- ",end="")
    #             print("|")

    #         print("  ", end="")
    #         for _ in range(n):
    #             print ("-", end="-")
    #         print("--")


class TicTacToe3DModel():
    
    def __init__(self):



class Agent:

    def __init__(self):
        self.model = TicTacToe3DModel()

    def mcts(self):

    def train(self):
        

    def load_policy(self):

    def choose_action(self):

game = TicTacToe3D()  

agent = Agent()

# def train():

#     env.init()










class Node:
    
  
    def __init__(self, game, reward, parent, player, canonicalBoard, a):
          
        self.child = None
        
        # self.T = 0
        self.N = 0        
        self.game = game
        
        self.player = player
        self.canonicalBoard = canonicalBoard
        self.validActions = self.game.getValidMoves(self.canonicalBoard)

        self.reward = reward
        self.isTerminal = (reward != 0)
        self.parent = parent
        self.a = a

        
        self.Qsa = 0  
        self.Nsa = 0

        self.Pa = []
        self.T = 0
        

    

        
    def getUCBscore(self):
        
        if self.N == 0:
            return float('inf')
        
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
            
        return (self.T / self.N) + c * sqrt(log(top_node.N) / self.N) 



    def create_child(self):
       
        if self.isTerminal: return

        child = {} 
        for a in self.validActions:
             
            # always take action for player 1 (canonical player).
            next_board, reward = self.game.step(self.canonicalBoard, a)
            
            # (game, reward, parent, player, observation, action_index)
            child[a] = Node(self.game, reward, self, self.player*(-1), next_board*(-1), a)                        
            
        self.child = child



    def search(self):
        
        if self.isTerminal: return

        # s = self.game.stringRepresentation(self.canonicalBoard)
 
 
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if self.validActions[a]:
                 

                if a in self.child:
                    u = self.Qsa[a] + self.args.cpuct * self.Pa[a] * math.sqrt(self.N) / (1 + self.Nsa)
                else:
                    u = self.args.cpuct * self.Pa[a] * math.sqrt(self.N + EPS)   

                if u > cur_best:
                    cur_best = u
                    best_act = a

 
        a = best_act

        # always take action for player 1 (canonical player).
        next_board, reward = self.game.step(self.canonicalBoard, a)
        
        # (game, reward, parent, player, observation, action_index)
        self.child[a] = Node(self.game, reward, self, self.player*(-1), next_board*(-1), a)  
                
        







        current = self
        
        while current.child: # phase 1: select

            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [ a for a,c in child.items() if c.getUCBscore() == max_U ]
            if len(actions) == 0:
                print("error zero length ", max_U)                      
            action = random.choice(actions)
            current = child[action]


        # # phase 2&3: expand & simulate
        # if current.N < 1: 
        #     # current.T = current.T + current.rollout()
        #     current.rollout()  
        # else: 


        # phase 2 & 3: expand & simulate
        if current.N < 1:
            # current.T = current.T + current.rollout() 
            current.rollout() 
        else:       
            if (not this.isTerminal):   

                current.create_child()
                cur_best = -float('inf')
                best_act = -1
                for a in current.child:  
                    u = self.args.cpuct * self.Pa[a] * math.sqrt(self.N + EPS)  # Q = 0 ?
                    if u > cur_best:
                        cur_best = u
                        best_act = a

                a = best_act
                current = current.child[a]
                current.rollout()
            
            else: # this node is terminal state

        current.N += 1


        # phase 4: backprop
            
        # parent = current 
        # while parent.parent: 
        #     parent = parent.parent
        #     parent.N += 1
        #     parent.T = parent.T + current.T


        parent = current
        if(current.isTerminal):
            v = current.reward
        else:
            v = current.T

        while parent.parent:

            v = -v
            
            a = parent.a
            parent = parent.parent

            parent.Qsa[a] = (parent.Nsa[a] * parent.Qsa[a] + v) / (parent.Nsa[a] + 1)
            parent.Nsa[a] += 1

            


        # current = self
        
        # while current.child:

        #     child = current.child
        #     max_U = max(c.getUCBscore() for c in child.values())
        #     actions = [ a for a,c in child.items() if c.getUCBscore() == max_U ]
        #     if len(actions) == 0:
        #         print("error zero length ", max_U)                      
        #     action = random.choice(actions)
        #     current = child[action]
     
        # if current.N < 1:
        #     current.T = current.T + current.rollout()
        # else:
        #     current.create_child()
        #     if current.child:
        #         current = random.choice(current.child)
        #     current.T = current.T + current.rollout()
            
        # current.N += 1      
        
        # parent = current
            
        # while parent.parent:
            
        #     parent = parent.parent
        #     parent.N += 1
        #     parent.T = parent.T + current.T


    def rollout(self):
    
        if self.isTerminal:
            return 0        
          
        Pa, T = nnet.predict(self.canonicalBoard)
        self.Pa = Pa
        self.T = T

        
        self.Pa = self.Pa * self.validActions
        sum_Pa = np.sum(self.Pa)
        if sum_Pa > 0:
            self.Pa /= sum_Pa  # re-normalize
        else:
            # if all valid moves were masked make all valid moves equally probable
            self.Pa = self.Pa + self.validActions
            self.Pa /= np.sum(self.Pa)

        # return T

        # v = 0

        # isTerminal = False
        # new_game = deepcopy(self.game)
        # while not isTerminal:
        #     action = new_game.action_space.sample()
        #     observation, reward, isTerminal, _ = new_game.step(action)
        #     v = v + reward
        #     if isTerminal:
        #         new_game.reset()
        #         new_game.close()
        #         break             
        # return v


    def next(self):
     
        if self.isTerminal:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')
        
        child = self.child
        
        max_N = max(node.N for node in child.values())
       
        max_children = [ c for a,c in child.items() if c.N == max_N ]
        
        if len(max_children) == 0:
            print("error zero length ", max_N) 
            
        max_child = random.choice(max_children)
        
        return max_child, max_child.action_index        




def getActionProb(canonicalBoard):
 
    #            (game, reward, parent, player, observation, action_index):
    mctree = Node(game, 0, None, 1, canonicalBoard, None)
    
    for i in range(20):        
        mctree.search()
        



    # next_tree, next_action = mytree.next()
        
    # next_tree.detach_parent()
    
    # return next_tree, next_action        