
import numpy as np
import torch
import importlib

MIN_EXPLORING_RATE = 0.01
MAX_EXPLORING_RATE = 0.5
MIN_LEARNING_RATE = 0.5
MAX_LEARNING_RATE = 0.5



nA = 9
nS = 2 * 3**9 # 2 is for 2 roles.

 
class TicTacToe:
    def __init__(self):
        self.reset()
        # self._playerSym = {-1: 'x', 1: 'o'}

    def reset(self):
        self.map = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]])
        self.player = np.random.choice([1, -1], 1, p=[0.5, 0.5])[0]
        observation = np.hstack(([self.player], self.map.flatten()))
        info = {'winner': 0}
        self.nSpace = 9
        return observation, info

    def isGameEnded(self):

        def reduce(v): return int(abs(sum(v)))

        for i in range(3):
            if(reduce(self.map[i, :]) == 3): return True
            if(reduce(self.map[:, i]) == 3): return True
  
        v = [self.map[i, i] for i in range(3)]
        if(reduce(v) == 3): return True

        v = [self.map[i, 2-i] for i in range(3)]
        if(reduce(v) == 3): return True

        return False


    def place(self, sym, pos):
        x = pos[0]
        y = pos[1]
        
        a = self.map[y, x]
        if(a != 0): 
            print("[place()] Warning: invalide action.")
            return False
        else:
            self.map[y, x] = sym
            return True
    
    
    def step(self, action):
        '''
            return: return observation, reward, terminated, False, None
        '''
        
        curPlayer = self.player 
        nextPlayer = -self.player
        # self.player = nextPlayer
 
        terminated = False
        reward = 0 
        info = {'winner': 0}

        isOk = self.place(self.player, action)

        observation = np.hstack(([nextPlayer], self.map.flatten()))

        if(not isOk):
            observation[0] = curPlayer
        else:
            self.nSpace -= 1

            if(self.isGameEnded()):                
                info['winner'] = curPlayer
                reward = 10
                terminated = True
            elif(self.nSpace == 0):
                info['winner'] = 0
                terminated = True

        self.player = observation[0]
        return observation, reward, terminated, False, info


    def render(self):
        symMap = {1: 'x', -1: 'o', 0: '-'}                    
        for r in self.map:  
            s = ''.join([symMap[sym] for sym in r])
            print(s)
        print()

# class Agent:

#     def __init__(self, nA, nS, 
#                  t=0,
#                  discount_factor=0.99):

#         self.update_parameters(t)  # init explore rate and learning rate
        
#         self.discount_factor = discount_factor
#         self.nA = nA
#         self.nS = nS
#         self.q_table = torch.zeros([self.nS, self.nA], dtype=torch.float32)

        

#         self._encodeAction = {
#             (0, 0): 0,
#             (1, 0): 1,
#             (2, 0): 2,
#             (0, 1): 3,
#             (1, 1): 4,
#             (2, 1): 5, 
#             (0, 2): 6,
#             (1, 2): 7,
#             (2, 2): 8                    
#         }

#         self._decodeAction = {
#             0: [0, 0],
#             1: [1, 0],
#             2: [2, 0],
#             3: [0, 1],
#             4: [1, 1],
#             5: [2, 1], 
#             6: [0, 2],
#             7: [1, 2],
#             8: [2, 2]                    
#         }

#     def encodeState(self, st):

#         num = 0
#         for i, e in enumerate(st[1:]): 
#             num += (e+1) * 3**i

#         num += int((st[0]+1)/2) * 3**9
#         return num

#     def encodeAction(self, a): 
#         return self._encodeAction[(a[0], a[1])]

#     def decodeAction(self, aId): 
#         return self._decodeAction[aId]

#     def _get_action_mask(self, state):
#         actionMask = []
#         for i, e in enumerate(state[1:]):
#             if(e == 0): actionMask.append(True)
#             else: actionMask.append(False)
#         return actionMask

 
#     def choose_action(self, state): 

#         stateId = self.encodeState(state)
#         actionMask = self._get_action_mask(state) 

#         if np.random.rand() < self.exploring_rate: 
#             action = np.random.choice(np.arange(self.nA)[actionMask])  
#         else:       
#             # print(f'a self.q_table.shape: {self.q_table.shape}')   
#             # print(f'actionMask: {actionMask}')       
#             # print(f'self.q_table[stateId]: {stateId}') 

#             idxMap = {}
#             idx = 0
#             for i, a in enumerate(actionMask):
#                 if(a): 
#                     idxMap[idx] = i
#                     idx += 1

#             action = torch.argmax(self.q_table[stateId, actionMask]).item()
#             action = idxMap[action]
        

#         return self.decodeAction(action)

     
#     def update_policy_MC(self, totalReturn, trajectory):

#         deltas = []
#         target = totalReturn
#         for s, a in reversed(trajectory):

#             sId = self.encodeState(s)
#             aId = self.encodeAction(a)
            
#             delta = abs(target - self.q_table[sId][aId])
#             deltas.append(delta)

#             self.q_table[sId][aId] += self.learning_rate * (target - self.q_table[sId][aId])

#             target = self.discount_factor * target
        
#         return np.mean(deltas)

#     def update_parameters(self, episode):
#         self.exploring_rate = \
#                 max(MIN_EXPLORING_RATE, min(MAX_EXPLORING_RATE, 0.99**((episode) / 30)))
#         self.learning_rate = \
#                 max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, 0.99 ** ((episode) / 30)))

#     def shutdown_explore(self): 
#         self.exploring_rate = 0

#     def load(self, path):
#         print(f'loaded agent: {path}')
#         self.q_table = torch.load(path)


#     def load_policy(self):

#         self.load('111022533_hw1_3_data')

env = TicTacToe()


def train(load_path=None, save_path=None):

   
    agent = Agent(nA, nS)
    # if(load_path is not None):
    #     agent.load(load_path)
    
    
    reward_per_epoch = []
    lifetime_per_epoch = []
    exploring_rates = []
    learning_rates = []
    deltas = []

    print_every_episode = 1
    save_every_episode = 500
    show_gif_every_episode = 5000
    NUM_EPISODE = 20000

    for episode in range(NUM_EPISODE):
    
        observation, info = env.reset() 
        
        trajectory = {1:[], -1:[]}
        
        while(1): 

            player = observation[0]

            action = agent.choose_action(observation) 
            
            observation_next, reward, terminated, _, info = env.step(action) 
            
            trajectory[player].append([observation, action])

            if terminated:  

                winner = info['winner']
                reward = {1:0, -1:0}   
                if(winner != 0):
                    reward[winner] = 10
                    reward[-winner] = -10


                delta1 = agent.update_policy_MC(reward[1], trajectory[1])
                delta2 = agent.update_policy_MC(reward[-1], trajectory[-1])
                 
                if episode % 100 == 0:
                    # print(f'eps: {episode}, delta1: {delta1}') 
                    deltas.append(delta1)

                break

            observation = observation_next
             
    
        agent.update_parameters(episode)

        if episode % save_every_episode == 0:
            if(save_path is not None):
                print(f'eps{episode} saved model')
                torch.save(agent.q_table, save_path)

    # if(save_path is not None):
    #     print('saved model')
    #     torch.save(agent.q_table, save_path)
      

    print(f'deltas: {deltas}')

# train(save_path='111022533_hw1_3_data')
# train()



# def evaluate(path=None):

#     agent = Agent(nA, nS)
#     # agent.load(path)
#     if(path is not None):
#         agent.load(path)

#     agent.shutdown_explore()

#     # agent.exploring_rate = 1
    
#     n = 10
#     for episode in range(n):
    
#         observation, info = env.reset()  
#         env.render()
#         while(1): 

#             player = observation[0]
#             if(player == 1): 
#                 action = input("enter action: ")                
#                 action = agent.decodeAction(int(action))
#             else:
#                 action = agent.choose_action(observation) 
            
#             observation_next, reward, terminated, _, info = env.step(action) 
#             env.render()

#             if terminated:   
#                 winner = info['winner']  
#                 print(f'winner: {winner}')
#                 break

#             observation = observation_next
     
     
# evaluate(path='111022533_hw1_3_data')



def evaluate():

    # agent = Agent(nA, nS)
    # agent.load(path)

    module = importlib.import_module('111022533_hw1_3_test')
    agent = module.Agent()
    
    agent.load_policy()

    n = 10
    for episode in range(n):
    
        observation, info = env.reset()  
        env.render()
        while(1): 

            player = observation[0]
            if(player == 1): 
                action = input("enter action: ")                
                action = agent.decodeAction(int(action))
            else:
                action = agent.choose_action(observation) 
            
            observation_next, reward, terminated, _, info = env.step(action) 
            env.render()

            if terminated:   
                winner = info['winner']  
                print(f'winner: {winner}')
                break

            observation = observation_next
     

     
evaluate()