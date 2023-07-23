import numpy as np
from CandyCrushGym import *

from Config import *

from Utils import *

class Node:
    def __init__(self, parent, state, action, reward):

        self.q = 0
        self.reward = 0

        self.n = 0

        self.c = 2

        self.parent = parent
        self.childrens = []

        self.state = state

        self.action = action

        self.done = False

    @property
    def UCT(self):
        
        if self.n == 0:
            return np.inf 

        # Upper Confidence Bound 1
        UCB1 = self.c * np.sqrt( np.log(self.parent.n) / self.n )

        # Upper Confidence Trees = MCTS + UCB1
        return self.q + UCB1 


class MCTS:

    def __init__(self, env: CandyCrushGym, request_state_img_shm, response_policy_shm, wait_for_response_sema, stateToImageConverter):
        self.N = 0

        self.env = env
        self.root = Node(parent=None, state=env.state, action=-1, reward=-1)

        self.NUM_ROLLOUT_STEPS = 4
        self.gamma = 0.99
        self.discount_factors = np.array([self.gamma**i for i in range(self.NUM_ROLLOUT_STEPS)])


        valid_actions = [action for action in range(self.env.action_space.n) if self.env.isValidAction(action)]
        self.valid_actions = np.array(valid_actions)

        invalid_actions = [action for action in range(self.env.action_space.n) if not self.env.isValidAction(action)]
        self.invalid_actions = np.array(invalid_actions)

        self.request_state_img_shm = request_state_img_shm
        self.response_policy_shm = response_policy_shm
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
                
                    return self.get_best_action(), self.root.q

                max_idx = np.argmax(UCT_values)
                current = current.childrens[max_idx]

            #
            # Rollout?
            #   
            if current.n == 0:
                G = self.rollout(current)

                self.backpropagate(current, G)
              
            else:
                
                #
                # Node expansion
                #
                
                state_current = np.copy(current.state)
                can_not_expand = True
                for action in self.valid_actions:
                    
                    self.env.state = state_current

                    next_state, reward, _, _ = self.env.step(action)
                        
                    if reward != 0:
                        can_not_expand = False 
                        next_state = np.copy(next_state)
                        node = Node(parent=current, state=next_state, action=action, reward=reward)
                        current.childrens.append(node)

                        state_current = np.copy(current.state)
                
                if can_not_expand:
                    current.done = True

                    if current == self.root:
                        return 1, 0

        self.env.state = state_env

        return self.get_best_action(), self.root.q
        

    def get_best_action(self):

        #
        # Determine best action
        #

        q_values = [node.q for node in self.root.childrens]

        max_idx = np.argmax(q_values)

        best_action = self.root.childrens[max_idx].action

        return best_action

        
    def rollout(self, node):
        
        state_env = np.copy(self.env.state)

        self.env.state = np.copy(node.state)


        reward_list = []

        
        state = node.state

        for i in range(self.NUM_ROLLOUT_STEPS):
            
            request_state_img = np.ndarray(shape=STATE_IMG_SHAPE, dtype=STATE_IMG_DTYPE, buffer=self.request_state_img_shm.buf)
            state_img = self.stateToImageConverter(state)
            request_state_img[:] = state_img

            self.wait_for_response_sema.acquire()

            response_policy = np.ndarray(shape=(NUM_ACTIONS,), dtype=POLICY_DTYPE, buffer=self.response_policy_shm.buf)
                
            num_iterations = 0
            terminate_rollout = False

            # convert + normalize -> prevent error by np.random.choice 
            response_policy = response_policy.astype('float64')
            response_policy /= response_policy.sum()  
            while True:
                
                
                action = np.random.choice(np.arange(NUM_ACTIONS), p=response_policy)
               

                if isValidAction(action):
                    next_state, reward, _, _ = self.env.step(action)

                    if reward != 0:
                        reward_list.append(reward)
                        break 

                    if num_iterations > 1000:
                        terminate_rollout = True 
                        break 
                
                    num_iterations += 1

            if terminate_rollout:
                break 
                

    
        rewards = np.array(reward_list)
        
        # cumulative discounted reward
        G = np.dot(self.discount_factors[:len(rewards)], rewards)

        self.env.state = state_env

        return G
    

    def backpropagate(self, node, G):

        current = node

        while current:
            
            G = current.reward + self.gamma * G
            current.n += 1
            
            current.q = current.q + 1/current.n * (G - current.q)

            current = current.parent

           

            


    

           
