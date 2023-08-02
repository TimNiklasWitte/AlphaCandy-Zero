import numpy as np
from CandyCrushGym import *

from Config import *

from Utils import *

c_puct = 2

class Node:
    def __init__(self, parent, state, action, p):

        self.n = 0
        self.w = 0
        self.q = 0
        self.p = p

        self.parent = parent
        self.childrens = []

        self.state = state

        self.action = action

        self.done = False

    @property
    def UCT(self):
        
        if self.parent.n == 0:
            return np.inf

        # Upper Confidence Bound 1
        UCB1 = c_puct * np.sqrt( np.log(self.parent.n) / (1 + self.n) )
        
        # Upper Confidence Trees = MCTS + UCB1
        return self.q + UCB1 


class MCTS:

    def __init__(self, env: CandyCrushGym, request_state_img_shm, response_policy_shm, response_value_shm, wait_for_response_sema, stateToImageConverter):
        self.N = 0

        self.env = env
        self.root = Node(parent=None, state=env.state, action=-1, p=1)

        self.NUM_ROLLOUT_STEPS = 4
        self.gamma = 0.99
        self.discount_factors = np.array([self.gamma**i for i in range(self.NUM_ROLLOUT_STEPS)])


        self.reduced_action_space = get_reduced_action_space()
        self.num_actions = len(self.reduced_action_space)


        self.request_state_img_shm = request_state_img_shm
        self.response_policy_shm = response_policy_shm
        self.response_value_shm = response_value_shm
        self.wait_for_response_sema = wait_for_response_sema

        self.stateToImageConverter = stateToImageConverter

    def run(self, num_iterations: int):
        
        state_env = np.copy(self.env.state)

        for i in range(num_iterations):

            current = self.root

            #
            # Tree Traversal
            #
            while len(current.childrens) != 0:
                UCT_values = [node.UCT for node in current.childrens if not node.done]

                if len(UCT_values) == 0:
                
                    current.done = True 

                    # every possible action in root node leads to terminal state 
                    if current.parent == None:
                        return self.get_best_action(), self.get_policy(), self.get_value()
                        
                    current = current.parent
                    continue

                max_idx = np.argmax(UCT_values)
                current = current.childrens[max_idx]

            #
            # Expand and evaluate
            #


            # Evaluate
            state = current.state

            request_state_img = np.ndarray(shape=STATE_SHAPE, dtype=STATE_DTYPE, buffer=self.request_state_img_shm.buf)
            request_state_img[:] = state[:]
            
            self.wait_for_response_sema.acquire()

            response_policy = np.ndarray(shape=(self.num_actions,), dtype=POLICY_DTYPE, buffer=self.response_policy_shm.buf)
     
            response_value = np.ndarray(shape=(1,), dtype=VALUE_DTYPE, buffer=self.response_value_shm.buf)
            response_value = response_value[0]


            # Expand
            state_current = np.copy(current.state)
            can_not_expand = True
            for action_idx, action in enumerate(self.reduced_action_space):
             
                self.env.state = state_current

                next_state, reward, _, _ = self.env.step(action)
                  

                if reward != 0:
                    can_not_expand = False 
                    next_state = np.copy(next_state)
                    node = Node(parent=current, state=next_state, action=action, p=response_policy[action_idx])

                    current.childrens.append(node)

                    state_current = np.copy(current.state)

            if can_not_expand:

                current.done = True
                if current == self.root:
                    self.env.state = state_env
                    #return 1, 0
                    return self.get_best_action(), self.get_policy(), self.get_value()
           

        self.env.state = state_env

        return self.get_best_action(), self.get_policy(), self.get_value()
        
    def backup(self, node, v):

        current = node
        
        while current:
            
            current.n += 1
            current.w += v

            current.q = current.w / current.n

            current = current.parent
    

    def get_best_action(self):

        #
        # Determine best action
        #

        q_values = [node.v / node.n for node in self.root.childrens if node.n != 0]

        # no action can be selected 
        if len(q_values) == 0:
            return self.reduced_action_space[0]

        max_idx = np.argmax(q_values)
     
        best_action = self.root.childrens[max_idx].action

        return best_action

    def get_policy(self):

        n_values = np.array([node.n for node in self.root.childrens])
        n_total = np.sum(n_values)

        if n_total == 0:
            return np.full(shape=(self.num_actions,), fill_value=1/self.num_actions, dtype=POLICY_DTYPE)
        
        policy = n_values / n_total

        return policy


    def get_value(self):

        if self.root.n == 0:
            return 0
        
        q_root = self.root.w / self.root.n
     
        return q_root
        
   
    

    

           

            


    

           
