import numpy as np

class GridworldEnv():
  

    def __init__(self):
        self.shape = (4, 4)
        self.nA = 4
        self.nS = 16
        self.P = {}
        self.terminal_st = [0, 15]

        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                
                s = self.pos2st((r, c))
                self.P[s] = {}

                for a in range(self.nA):
                    
                    p = None
                    
                    # up
                    if  (a == 0): p = (max(r-1, 0), c)
                    # right
                    elif(a == 1): p = (r, min(c+1, self.shape[1]-1))
                    # down
                    elif(a == 2): p = (min(r+1, self.shape[0]-1), c)
                    # left
                    elif(a == 3): p = (r, max(c-1, 0))
                    
                    st_nxt = self.pos2st(p)

                    done = False
                    # if st_nxt in self.terminal_st:
                    if(self.isTerminal(st_nxt)):
                        done = True

                    self.P[s][a] = [(1, st_nxt, -1, done)]

        for s in range(self.nS):
            for a in range(self.nA):
                print(f'{s}, {a}: {self.P[s][a]}')

    def pos2st(self, pos):
        return self.shape[0] * pos[0] + pos[1]

    def isTerminal(self, s):        
        return s in self.terminal_st

env = GridworldEnv()





def policy_eval(policy, env, discount_factor=0.9, theta=0.00001):
 
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            if(env.isTerminal(s)): continue
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for  prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        
        if delta < theta:
            break
            
    return np.array(V)
 


def policy_improvement(env):

    policy = np.ones([env.nS, env.nA]) / env.nA

    V = policy_eval(policy, env)

    return V

  


v = policy_improvement(env)
 
print(np.reshape(v, (4, 4)))

