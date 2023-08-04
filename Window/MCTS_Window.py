import numpy as np
from CandyCrushGym import *

from Config import *

from WindowUtils import *
from CandyCrushUtiles import *

c_puct = 2

class Node:
    def __init__(self, parent, state, action, action_idx, p):

        self.n = 0
        self.w = 0
        self.q = 0
        self.p = p

        self.parent = parent
        self.childrens = []

        self.state = state

        self.action = action
        self.action_idx = action_idx

        self.done = False

    @property
    def UCT(self):
        
        if self.parent.n == 0:
            return np.inf

        # Upper Confidence Bound 1
        UCB1 = c_puct * np.sqrt( np.log(self.parent.n) / (1 + self.n) )
        
        # Upper Confidence Trees = MCTS + UCB1
        return self.q + UCB1 


class MCTS_Window:

    def __init__(self, env: CandyCrushGym, policyValueNetwork, stateToImageConverter):
        self.N = 0

        self.env = env
        self.root = Node(parent=None, state=env.state, action=-1, action_idx=-1, p=1)

        self.policyValueNetwork = policyValueNetwork
        self.stateToImageConverter = stateToImageConverter


        self.reduced_action_space = get_reduced_action_space()
        self.num_actions = len(self.reduced_action_space)

        self.current = self.root

        self.step(first_run=True)

    def step(self, first_run=False, return_policy=False):
        state_env = np.copy(self.env.state)


        self.current = self.root
        
        #
        # Tree Traversal
        #
        while len(self.current.childrens) != 0:
            UCT_values = [node.UCT for node in self.current.childrens if not node.done]

            if len(UCT_values) == 0:
                
                self.current.done = True 

                # every possible action in root node leads to terminal state 
                if self.current.parent == None:
                    self.env.state = state_env
                    if return_policy:
                        return self.get_best_action(), self.get_policy(), self.get_value()
                    else:
                        return 
                        
                self.current = self.current.parent
                continue
            
            max_idx = np.argmax(UCT_values)
            self.current = self.current.childrens[max_idx]
        

        #
        # Expand and evaluate
        #

        # Evaluate
        state = self.current.state
        state_img = self.stateToImageConverter(state)
        
        state_img = np.expand_dims(state_img, axis=0)

        policy, value = self.policyValueNetwork(state_img)
        
        # remove batch dim
        policy = policy[0] 
        value = value[0][0] # and raw value (no array)


        # Expand
        state_current = np.copy(self.current.state)
        can_not_expand = True
        for action_idx, action in enumerate(self.reduced_action_space):
             
            self.env.state = state_current

            next_state, reward, _, _ = self.env.step(action)
                  

            if reward != 0:
                can_not_expand = False 
                next_state = np.copy(next_state)
                node = Node(parent=self.current, state=next_state, action=action, action_idx=action_idx, p=policy[action_idx])

                self.current.childrens.append(node)

                state_current = np.copy(self.current.state)


        if can_not_expand:

            self.current.done = True
                
            if self.current == self.root:
                self.env.state = state_env
                #return 1, 0

                if return_policy: 
                    return self.get_best_action(), self.get_policy(), self.get_value()
                else:
                    return 


        #
        # Backup
        #
        self.backup(self.current, value)


        self.env.state = state_env
        
        if not first_run:
            if return_policy:
                return self.get_best_action(), self.get_policy(), self.root.q
            else:
                return 

    
    def backup(self, node, v):

        current = node
        
        while current:
            
            current.n += 1
            current.w += v

            current.q = current.w / current.n

            current = current.parent

    def get_best_action(self):

        n_values = np.array([node.n for node in self.root.childrens])
        n_total = np.sum(n_values)

        # no action can be selected 
        if n_total == 0:
            return self.reduced_action_space[0]

        policy = n_values / n_total
        max_idx = np.argmax(policy)
     
        best_action = self.root.childrens[max_idx].action

        return best_action
    

    def get_policy(self):

        n_values = np.array([node.n for node in self.root.childrens])
        n_total = np.sum(n_values)

        if n_total == 0:
            return np.full(shape=(NUM_ACTIONS,), fill_value=1/NUM_ACTIONS, dtype=POLICY_DTYPE)
        
       
        policy = np.zeros(shape=(NUM_ACTIONS,) , dtype=POLICY_DTYPE)
        
        probs = n_values / n_total

        actions = [node.action for node in self.root.childrens]
        policy[actions] = probs

        
        reduced_action_space_alternative = get_reduced_action_space_alternative()
        action_idxs = [node.action_idx for node in self.root.childrens]
        equivalent_action_idxs = reduced_action_space_alternative[action_idxs]

        policy[equivalent_action_idxs] = probs

        policy /= policy.sum()

        return policy


    def get_value(self):
        
        q_root = self.root.w / self.root.n
     
        return q_root
           

            


    

           
