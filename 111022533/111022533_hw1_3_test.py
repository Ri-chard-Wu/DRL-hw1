

import numpy as np
import torch

MIN_EXPLORING_RATE = 0.01
MAX_EXPLORING_RATE = 0.5
MIN_LEARNING_RATE = 0.5
MAX_LEARNING_RATE = 0.5



nA = 9
nS = 2 * 3**9 # 2 is for 2 roles.

class Agent:

    def __init__(self):

        self.q_table = torch.zeros([nS, nA], dtype=torch.float32)

        self._encodeAction = {
            (0, 0): 0,
            (1, 0): 1,
            (2, 0): 2,
            (0, 1): 3,
            (1, 1): 4,
            (2, 1): 5, 
            (0, 2): 6,
            (1, 2): 7,
            (2, 2): 8                    
        }

        self._decodeAction = {
            0: [0, 0],
            1: [1, 0],
            2: [2, 0],
            3: [0, 1],
            4: [1, 1],
            5: [2, 1], 
            6: [0, 2],
            7: [1, 2],
            8: [2, 2]                    
        }

    def encodeState(self, st):

        num = 0
        for i, e in enumerate(st[1:]): 
            num += (e+1) * 3**i

        num += int((st[0]+1)/2) * 3**9
        return int(num)

    def encodeAction(self, a): 
        return self._encodeAction[(a[0], a[1])]

    def decodeAction(self, aId): 
        return self._decodeAction[aId]

    def _get_action_mask(self, state):
        actionMask = []
        for i, e in enumerate(state[1:]):
            if(e == 0): actionMask.append(True)
            else: actionMask.append(False)
        return actionMask

 
    def choose_action(self, state): 
        # print(f'state: {state}')
        stateId = self.encodeState(state)
        actionMask = self._get_action_mask(state) 

 
        idxMap = {}
        idx = 0
        for i, a in enumerate(actionMask):
            if(a): 
                idxMap[idx] = i
                idx += 1

        action = torch.argmax(self.q_table[stateId, actionMask]).item()
        action = idxMap[action]
        

        return self.decodeAction(action)
 

    def load_policy(self): 
        self.q_table = torch.load('./111022533/111022533_hw1_3_data')
