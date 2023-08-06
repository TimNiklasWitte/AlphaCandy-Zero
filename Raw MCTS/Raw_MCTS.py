import numpy as np

class Node:
    def __init__(self, parent, state, action, done):

        self.v = 0

        self.n = 0

        self.c = 2

        self.parent = parent
        self.childrens = []

        self.state = state

        self.action = action

        self.done = done

    @property
    def UCT(self):
        
        if self.n == 0:
            return np.inf 

        # Upper Confidence Bound 1
        UCB1 = self.c * np.sqrt( np.log(self.parent.n) / self.n )

        # Upper Confidence Trees = MCTS + UCB1
        q = self.v / self.n
        return q + UCB1 


class Raw_MCTS:

    def __init__(self, env):
        self.N = 0

        self.env = env
        self.root = Node(parent=None, state=env.state, action=-1, done=False)

        self.NUM_ROLLOUT_STEPS = 4
        self.gamma = 0.99
        self.discount_factors = np.array([self.gamma**i for i in range(self.NUM_ROLLOUT_STEPS)])


        # can represent every possible action only with top and right actions
        # also possible top-left, down-right, down-left
        reduced_action_space = []
        for action in range(env.action_space.n):
            if self.env.isValidAction(action):

                x_y_direction = self.env.get_x_y_direction(action)

                if "top" in x_y_direction:
                    reduced_action_space.append(action)
                    
                elif "right" in x_y_direction:
                    reduced_action_space.append(action)

        
        self.reduced_action_space = reduced_action_space
     
    def run(self, num_iterations: int ):
        
        state_env = np.copy(self.env.state)


        for i in range(num_iterations):

            current = self.root

            #
            # Tree Traversal
            #

      
            while len(current.childrens) != 0:
     

                childrens = [node for node in current.childrens if not node.done]

                if len(childrens) == 0:

                    current.done = True 

                    # every possible action in root node leads to terminal state 
                    if current.parent == None:
                        best_action = 0
                        self.env.state = state_env
                        #return best_action, self.root.v / self.root.n
                        return self.get_best_action(), self.get_policy(), self.get_value()
                        
                    current = current.parent
                    continue
          

                UCT_values = [node.UCT for node in childrens]

                max_idx = np.argmax(UCT_values)
                current = childrens[max_idx]

            #
            # Rollout?
            #   
            if current.n == 0:
                v, done = self.rollout(current)

                current.done = done 

                self.backpropagate(current, v)
              
            else:
                
                #
                # Node expansion
                #

                state_current = np.copy(current.state)
                can_not_expand = True

                for action in self.reduced_action_space:
                    

                    self.env.state = state_current

                    next_state, reward, _, _ = self.env.step(action)


                    if reward != 0:
                        can_not_expand = False 
                        next_state = np.copy(next_state)
                        node = Node(parent=current, state=next_state, action=action, done=False)
                        current.childrens.append(node)

                        state_current = np.copy(current.state)


                if can_not_expand:
                    current.done = True

                    if current == self.root:
                        self.env.state = state_env
                        #return 1, 0
                        return self.get_best_action(), self.get_policy(), self.get_value()

       
        self.env.state = state_env

        #
        # Determine best action
        #
     
        return self.get_best_action(), self.get_policy(), self.get_value()
        
    
    def rollout(self, node):
        
        state_env = np.copy(self.env.state)

        self.env.state = np.copy(node.state)


        reward_list = []
        

        for i in range(self.NUM_ROLLOUT_STEPS):

            
            np.random.shuffle(self.reduced_action_space)
            
            action_found = False
            for action in self.reduced_action_space:

                next_state, reward, done , _ = self.env.step(action)

                if reward != 0:
                    reward_list.append(reward)
                    action_found = True
                    break 
            
            if not action_found:
                self.env.state = state_env
                return 0, True

              

        rewards = np.array(reward_list)
        
        # cumulative discounted reward
        G = np.dot(self.discount_factors[:len(reward_list)], rewards)

        self.env.state = state_env

        return G, False
    

    def backpropagate(self, node, v):

        current = node

        while current:
            
            current.n += 1
            
            current.v += v

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


        self.root.childrens.sort(key=lambda node: node.action)

        n_values = np.array([node.n for node in self.root.childrens])
        n_total = np.sum(n_values)

        if n_total == 0:
            len_reduced_action_space = len(self.reduced_action_space)
            return np.full(shape=(len_reduced_action_space,), fill_value=1/len_reduced_action_space, dtype=np.float32)
        
        policy = n_values / n_total

        return policy

    def get_value(self):
        v = self.root.v / self.root.n
        return v

    

           