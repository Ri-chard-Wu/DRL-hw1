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
                    if  (a == 0): p = (max(r-1, 0), c) # up 
                    elif(a == 1): p = (r, min(c+1, self.shape[1]-1)) # right 
                    elif(a == 2): p = (min(r+1, self.shape[0]-1), c) # down 
                    elif(a == 3): p = (r, max(c-1, 0)) # left
                    
                    st_nxt = self.pos2st(p) 
                    done = False 
                    if(self.isTerminal(st_nxt)): done = True 
                    self.P[s][a] = [(1, st_nxt, -1, done)]

        for s in range(self.nS):
            for a in range(self.nA):
                print(f'{s}, {a}: {self.P[s][a]}')

    def pos2st(self, pos):
        return self.shape[0] * pos[0] + pos[1]

    def isTerminal(self, s):        
        return s in self.terminal_st

 


def policy_eval(policy, env, gamma=0.9, theta=0.00001): 
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            if(env.isTerminal(s)): continue
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for  prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta: break 
    return np.array(V)
 

env = GridworldEnv()

policy = np.ones([env.nS, env.nA]) / env.nA

gamma = 0.1

v = policy_eval(policy, env, gamma)
 
print(np.reshape(v, (4, 4)))


s = ""
for st, _v in enumerate(v):
    if(env.isTerminal(st)): continue
    s += '{0:.2f} '.format(_v)
s = s[:-1]

f = open(f"111022533_hw1_1_data_gamma_{gamma}", "w")
f.write(s)
f.close()
