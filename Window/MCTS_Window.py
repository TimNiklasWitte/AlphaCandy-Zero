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


class MCTS_Window:

    def __init__(self, env: CandyCrushGym, policyValueNetwork, stateToImageConverter):
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

        self.policyValueNetwork = policyValueNetwork
        self.stateToImageConverter = stateToImageConverter


        self.current = self.root

        self.step(first_run=True)

    def step(self, first_run=False):
        state_env = np.copy(self.env.state)


        self.current = self.root
        
        #
        # Tree Traversal
        #
        while len(self.current.childrens) != 0:
            UCT_values = [node.UCT for node in self.current.childrens if not node.done]

            if len(UCT_values) == 0:
                return self.get_best_action(), self.get_policy(), self.root.q, False

            max_idx = np.argmax(UCT_values)
            self.current = self.current.childrens[max_idx]
        
        #
        # Rollout?
        #   
        if self.current.n == 0:
            G = self.rollout(self.current)
            self.backpropagate(self.current, G)

        else:

            #
            # Node expansion
            #
                
            state_current = np.copy(self.current.state)
            can_not_expand = True
            for action in self.valid_actions:
                    
                self.env.state = state_current

                next_state, reward, _, _ = self.env.step(action)
                        
                if reward != 0:
                    can_not_expand = False 
                    next_state = np.copy(next_state)
                    node = Node(parent=self.current, state=next_state, action=action, reward=reward)
                    self.current.childrens.append(node)

                    state_current = np.copy(self.current.state)
                
            if can_not_expand:
                self.current.done = True

                if self.current == self.root:
                    return None, None, None, True

        self.env.state = state_env
        
        if not first_run:
            return self.get_best_action(), self.get_policy(), self.root.q, False
    
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
                    print("here")
                    return self.get_best_action(), self.get_policy(), self.root.q, False

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
                        return None, None, None, True 

        self.env.state = state_env

        
        return self.get_best_action(), self.get_policy(), self.root.q, False

    def get_policy(self):
        return [(node.action, node.q) for node in self.root.childrens]         

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
            
            state_img = self.stateToImageConverter(state)
            state_img = np.expand_dims(state_img, axis=0)

            policy, _ = self.policyValueNetwork(state_img)
            policy = np.array(policy[0])

            num_iterations = 0
            terminate_rollout = False

            # convert + normalize -> prevent error by np.random.choice 
            policy = policy.astype('float64')
            policy /= policy.sum()  
            while True:
                
                
                action = np.random.choice(np.arange(NUM_ACTIONS), p=policy)
               

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

           

            


    

           
