import copy
import numpy as np
from fimdp.core import ConsMDP, CounterStrategy
from fimdp.energy_solvers import GoalLeaningES, BasicES
from fimdp.objectives import AS_REACH, BUCHI, MIN_INIT_CONS, POS_REACH, SAFE
from matplotlib import pyplot as plt
from scipy.stats import vonmises
import matplotlib.animation as animation


class SynchronousMultiAgentEnv:

    def __init__(self, num_agents, grid_size, capacities, reloads, targets, init_states=None, enhanced_actionspace=0, weakaction_cost=1, strongaction_cost=2, velocity=5, heading_sd=0.524):
        """Class that models a Markov Decision Process where the agents and the environment model the dynamics of multiple UUV and 
        the ocean currents it is operating in. All transitions in the environment are synchronous. 
        """

        # environment attributes
        self.grid_size = grid_size
        self.capacities = capacities
        self.waction_cost = weakaction_cost
        self.saction_cost = strongaction_cost
        self.reloads = reloads
        self.targets = targets
        
        # agent attributes
        self.num_agents = num_agents
        self.agents = [agent for agent in range(self.num_agents)]
        self.targets_alloc = [[] for agent in self.agents]
        self.init_states = init_states
        self.positions = [None for agent in self.agents]
        self.strategies = [None for agent in self.agents]
        self.agent_done = [0 for agent in self.agents]
        self.agents_colors = None
        
        # logging attributes
        self.state_histories = [[] for agent in self.agents]
        self.action_histories = [[] for agent in self.agents]
        self.target_histories = [[] for agent in self.agents]
        self.num_timesteps = 0
        
        # consmdp related attributes
        self.consmdp = None
        self.num_states = grid_size[0]*grid_size[1]
        self.states = [s for s in range(self.num_states)]
        if enhanced_actionspace == 0:
            self.num_actions = 8
            self.weak_actions = [0,1,2,3]
            self.strong_actions = [4,5,6,7]
        elif enhanced_actionspace == 1:
            self.num_actions = 16
            self.weak_actions = [0,1,2,3,8,9,10,11]
            self.strong_actions = [4,5,6,7,12,13,14,15]
        self.actions = [a for a in range(self.num_actions)]
        self.is_reload = np.zeros(self.num_states)
        self.energies = copy.deepcopy(self.capacities)
        self.trans_prob = np.zeros([self.num_states, self.num_actions, self.num_states])
        self.action_to_label = {0:'Weak East', 1:'Weak North', 2:'Weak West', 
                               3:'Weak South', 4:'Strong East', 5:'Strong North',
                               6:'Strong West', 7:'Strong South', 8:'Weak North-East',
                               9: 'Weak North-West', 10:'Weak South-West', 
                               11:'Weak South-East', 12:'Strong North-East',
                               13:'Strong North-West', 14:'Strong South-West',
                               15:'Strong South-East'}        
        self.label_to_action = {v: k for k, v in self.action_to_label.items()}     

        # ocean current related attributes - demo config
        self.agent_v = velocity
        self.agent_headingsd = heading_sd
        self.flow_vx = 0.1*self.agent_v * np.ones([self.num_states])
        self.flow_vy = 0.2*self.agent_v * np.ones([self.num_states])
        self.flow_mag = np.sqrt([i**2 + j**2 for i,j in zip(self.flow_vx, self.flow_vy)])
        np.linalg.norm([self.flow_vx, self.flow_vy])
        self.flow_theta = np.arctan2(self.flow_vy, self.flow_vx)
        self.action_theta = {0:0, 1:np.pi/2, 2:np.pi, 3:-np.pi/2, 8:np.pi/4, 
                           9:3*np.pi/4, 10:-3*np.pi/4, 11:-np.pi/2}
        
        # initialize environment and create consmdp
        self.trans_prob = self._generate_dynamics()
        self._create_agents_colors()
        self.reset()
        self._add_reloads()
        self.create_consmdp()


    def _generate_dynamics(self):
        
        """
        Generate and return the state transition probability array for the given
        gridworld depending on the cardinality of the action space.
        """

        for s in self.states:
            for a in self.actions:
                if a in [0, 4]:
                    if (s % self.grid_size[1] == self.grid_size[1]-1):
                        self.trans_prob[s, a, s] = 1.0
                    else:
                        self.trans_prob[s, a, s+1] = 1.0
                elif a in [1, 5]:
                    if (s - self.grid_size[1] < 0):
                        self.trans_prob[s, a, s] = 1.0
                    else:
                        self.trans_prob[s, a, s-self.grid_size[1]] = 1.0
                elif a in [2, 6]:
                    if (s % self.grid_size[1] == 0):
                        self.trans_prob[s, a, s] = 1.0
                    else:
                        self.trans_prob[s, a, s-1] = 1.0
                elif a in [3, 7]:
                    if (s + self.grid_size[1] >= self.num_states):
                        self.trans_prob[s, a, s] = 1.0
                    else:
                        self.trans_prob[s, a, s+self.grid_size[1]] = 1.0
                    
                if self.num_actions == 16:
                    if a in [8, 12]:
                        if (s - self.grid_size[1] < 0) or (s % self.grid_size[1] == self.grid_size[1]-1):
                            self.trans_prob[s, a, s] = 1.0
                        else:
                            self.trans_prob[s, a, s-self.grid_size[1]+1] = 1.0
                    elif a in [9, 13]:
                        if (s - self.grid_size[1] < 0) or (s % self.grid_size[1] == 0):
                            self.trans_prob[s, a, s] = 1.0
                        else:
                            self.trans_prob[s, a, s-self.grid_size[1]-1] = 1.0
                    elif a in [10, 14]:
                        if (s + self.grid_size[1] >= self.num_states) or (s % self.grid_size[1] == 0):
                            self.trans_prob[s, a, s] = 1.0
                        else:
                            self.trans_prob[s, a, s+self.grid_size[1]-1] = 1.0
                    elif a in [11, 15]:
                        if (s + self.grid_size[1] >= self.num_states) or (s % self.grid_size[1] == self.grid_size[1]-1):
                            self.trans_prob[s, a, s] = 1.0
                        else:
                            self.trans_prob[s, a, s+self.grid_size[1]+1] = 1.0                    
                            
                # generate stochasic dynamics for weak actions at interior states
                if a in self.weak_actions:
                    if not ((s - self.grid_size[1] < 0) or
                        (s + self.grid_size[1] >= self.num_states) or
                        (s % self.grid_size[1] == self.grid_size[1]-1) or
                        (s % self.grid_size[1] == 0)):
                        self.trans_prob[s, a, :] = self._generate_stochastic_dynamics(s, a)

                # generate stochastic dynamics for weak actions at states on edges
                if a in self.weak_actions:
                    if ((s - self.grid_size[1] < 0) or
                        (s + self.grid_size[1] >= self.num_states) or
                        (s % self.grid_size[1] == self.grid_size[1]-1) or
                        (s % self.grid_size[1] == 0)):
                        self.trans_prob[s, a, :] = self._generate_edge_stochastic_dynamics(s, a)

        return self.trans_prob

    def _generate_edge_stochastic_dynamics(self, s, a):
        """
        Given state and action for edge states, generate the stochastic transition dynamics
        for different settings 
        """
        # combined actual heading
        actual_vx = self.agent_v*np.cos(self.action_theta[a]) + self.flow_mag[s]*np.cos(self.flow_theta[s])
        actual_vy = self.agent_v*np.sin(self.action_theta[a]) + self.flow_mag[s]*np.sin(self.flow_theta[s])
        actual_theta = np.arctan2(actual_vy, actual_vx)
        rv = vonmises(1/self.agent_headingsd, actual_theta)
        tp = np.zeros(self.num_states)

        if self.num_actions == 8:
            if (s - self.grid_size[1] < 0):
                if (s % self.grid_size[1] == self.grid_size[1]-1):
                    # North East
                    tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.75*np.pi) + rv.cdf(-0.75*np.pi) - rv.cdf(-np.pi),2)
                    tp[s+self.grid_size[1]] =  1.0 - np.sum(copy.deepcopy(tp))

                elif (s % self.grid_size[1] == 0):
                    # North West
                    tp[s+1] = round((rv.cdf(0.25*np.pi) - rv.cdf(-0.25*np.pi)),2)
                    tp[s+self.grid_size[1]] =  1.0 - np.sum(copy.deepcopy(tp))

                else:
                    # Just North
                    tp[s+1] = round((rv.cdf(0.25*np.pi) - rv.cdf(-0.25*np.pi)),2)
                    tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.75*np.pi) + rv.cdf(-0.75*np.pi) - rv.cdf(-np.pi),2)
                    tp[s+self.grid_size[1]] =  1.0 - np.sum(copy.deepcopy(tp))

            elif (s + self.grid_size[1] >= self.num_states):
                if (s % self.grid_size[1] == self.grid_size[1]-1):
                    # South East
                    tp[s-self.grid_size[1]] = round(rv.cdf(0.75*np.pi) - rv.cdf(0.25*np.pi),2)
                    tp[s-1] = 1.0 - np.sum(copy.deepcopy(tp))

                elif (s % self.grid_size[1] == 0):
                    # South West
                    tp[s+1] = round((rv.cdf(0.25*np.pi) - rv.cdf(-0.25*np.pi)),2)
                    tp[s-self.grid_size[1]] = 1.0 - np.sum(copy.deepcopy(tp))

                else:
                    # Just South
                    tp[s+1] = round((rv.cdf(0.25*np.pi) - rv.cdf(-0.25*np.pi)),2)
                    tp[s-self.grid_size[1]] = round(rv.cdf(0.75*np.pi) - rv.cdf(0.25*np.pi),2)
                    tp[s-1] = 1.0 - np.sum(copy.deepcopy(tp))

            elif (s % self.grid_size[1] == self.grid_size[1]-1):
                # Just East
                tp[s-self.grid_size[1]] = round(rv.cdf(0.75*np.pi) - rv.cdf(0.25*np.pi),2)
                tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.75*np.pi) + rv.cdf(-0.75*np.pi) - rv.cdf(-np.pi),2)
                tp[s+self.grid_size[1]] =  1.0 - np.sum(copy.deepcopy(tp))

            elif (s % self.grid_size[1] == 0):
                # Just West
                tp[s+1] = round((rv.cdf(0.25*np.pi) - rv.cdf(-0.25*np.pi)),2)
                tp[s-self.grid_size[1]] = round(rv.cdf(0.75*np.pi) - rv.cdf(0.25*np.pi),2)
                tp[s+self.grid_size[1]] =  1.0 - np.sum(copy.deepcopy(tp))

        elif self.num_actions == 16:
            if (s - self.grid_size[1] < 0):
                if (s % self.grid_size[1] == self.grid_size[1]-1):
                    # North East
                    tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.875*np.pi) + rv.cdf(-0.875*np.pi) - rv.cdf(-np.pi),2)
                    tp[s+self.grid_size[1]] = round(rv.cdf(-0.375*np.pi) - rv.cdf(-0.625*np.pi),2)
                    tp[s+self.grid_size[1]-1] =  1.0 - np.sum(copy.deepcopy(tp)) 

                elif (s % self.grid_size[1] == 0):
                    # North West
                    tp[s+1] = round(rv.cdf(0.125*np.pi) - rv.cdf(-0.125*np.pi),2)
                    tp[s+self.grid_size[1]] = round(rv.cdf(-0.375*np.pi) - rv.cdf(-0.625*np.pi),2)
                    tp[s+self.grid_size[1]+1] =  1.0 - np.sum(copy.deepcopy(tp))  

                else:
                    # Just North
                    tp[s+1] = round(rv.cdf(0.125*np.pi) - rv.cdf(-0.125*np.pi),2)
                    tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.875*np.pi) + rv.cdf(-0.875*np.pi) - rv.cdf(-np.pi),2)
                    tp[s+self.grid_size[1]] = round(rv.cdf(-0.375*np.pi) - rv.cdf(-0.625*np.pi),2)
                    tp[s+self.grid_size[1]-1] = round(rv.cdf(-0.625*np.pi) - rv.cdf(-0.875*np.pi),2)
                    tp[s+self.grid_size[1]+1] =  1.0 - np.sum(copy.deepcopy(tp))  

            elif (s + self.grid_size[1] >= self.num_states):
                if (s % self.grid_size[1] == self.grid_size[1]-1):
                    # South East
                    tp[s-self.grid_size[1]] = round(rv.cdf(0.625*np.pi) - rv.cdf(0.375*np.pi),2)
                    tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.875*np.pi) + rv.cdf(-0.875*np.pi) - rv.cdf(-np.pi),2)
                    tp[s-self.grid_size[1]-1] = 1.0 - np.sum(copy.deepcopy(tp))   

                elif (s % self.grid_size[1] == 0):
                    # South West
                    tp[s+1] = round(rv.cdf(0.125*np.pi) - rv.cdf(-0.125*np.pi),2)
                    tp[s-self.grid_size[1]] = round(rv.cdf(0.625*np.pi) - rv.cdf(0.375*np.pi),2)
                    tp[s-self.grid_size[1]+1] = 1.0 - np.sum(copy.deepcopy(tp))   

                else:
                    # Just South
                    tp[s+1] = round(rv.cdf(0.125*np.pi) - rv.cdf(-0.125*np.pi),2)
                    tp[s-self.grid_size[1]] = round(rv.cdf(0.625*np.pi) - rv.cdf(0.375*np.pi),2)
                    tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.875*np.pi) + rv.cdf(-0.875*np.pi) - rv.cdf(-np.pi),2)
                    tp[s-self.grid_size[1]+1] = round(rv.cdf(0.375*np.pi) - rv.cdf(0.125*np.pi),2)
                    tp[s-self.grid_size[1]-1] =  1.0 - np.sum(copy.deepcopy(tp))  

            elif (s % self.grid_size[1] == self.grid_size[1]-1):
                # Just East
                tp[s-self.grid_size[1]] = round(rv.cdf(0.625*np.pi) - rv.cdf(0.375*np.pi),2)
                tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.875*np.pi) + rv.cdf(-0.875*np.pi) - rv.cdf(-np.pi),2)
                tp[s+self.grid_size[1]] = round(rv.cdf(-0.375*np.pi) - rv.cdf(-0.625*np.pi),2)
                tp[s-self.grid_size[1]-1] = round(rv.cdf(0.875*np.pi) - rv.cdf(0.625*np.pi),2)
                tp[s+self.grid_size[1]-1] =  1.0 - np.sum(copy.deepcopy(tp))  

            elif (s % self.grid_size[1] == 0):
                # Just West
                tp[s+1] = round(rv.cdf(0.125*np.pi) - rv.cdf(-0.125*np.pi),2)
                tp[s-self.grid_size[1]] = round(rv.cdf(0.625*np.pi) - rv.cdf(0.375*np.pi),2)
                tp[s+self.grid_size[1]] = round(rv.cdf(-0.375*np.pi) - rv.cdf(-0.625*np.pi),2)
                tp[s-self.grid_size[1]+1] = round(rv.cdf(0.375*np.pi) - rv.cdf(0.125*np.pi),2)
                tp[s+self.grid_size[1]+1] =  1.0 - np.sum(copy.deepcopy(tp))  
        else:
            raise Exception('Infeasible actions in the action space. ')

        tp = tp.round(2)
        if (not np.all(tp>=0)) or (abs(np.sum(tp) - 1.0) >=  1e-8):
            print(tp)
            raise Exception('Invalid distribution for state {} and action {}'.format(s,a))
        else:
            return tp

    def _generate_stochastic_dynamics(self, s, a):
        """
        Given state and action, generate an array listing the probability of the 
        agent reaching all the states for the given weak action. 
        """
        # combined actual heading
        actual_vx = self.agent_v*np.cos(self.action_theta[a]) + self.flow_mag[s]*np.cos(self.flow_theta[s])
        actual_vy = self.agent_v*np.sin(self.action_theta[a]) + self.flow_mag[s]*np.sin(self.flow_theta[s])
        actual_theta = np.arctan2(actual_vy, actual_vx)
        rv = vonmises(1/self.agent_headingsd, actual_theta)
        tp = np.zeros(self.num_states)
        
        if self.num_actions == 8:
            tp[s+1] = round((rv.cdf(0.25*np.pi) - rv.cdf(-0.25*np.pi)),2)
            tp[s-self.grid_size[1]] = round(rv.cdf(0.75*np.pi) - rv.cdf(0.25*np.pi),2)
            tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.75*np.pi) + rv.cdf(-0.75*np.pi) - rv.cdf(-np.pi),2)
            tp[s+self.grid_size[1]] =  1.0 - np.sum(copy.deepcopy(tp))
            tp = tp.round(2)

        elif self.num_actions == 16:
            tp[s+1] = round(rv.cdf(0.125*np.pi) - rv.cdf(-0.125*np.pi),2)
            tp[s-self.grid_size[1]] = round(rv.cdf(0.625*np.pi) - rv.cdf(0.375*np.pi),2)
            tp[s-1] = round(rv.cdf(np.pi) - rv.cdf(0.875*np.pi) + rv.cdf(-0.875*np.pi) - rv.cdf(-np.pi),2)
            tp[s+self.grid_size[1]] = round(rv.cdf(-0.375*np.pi) - rv.cdf(-0.625*np.pi),2)
            tp[s-self.grid_size[1]+1] = round(rv.cdf(0.375*np.pi) - rv.cdf(0.125*np.pi),2)
            tp[s-self.grid_size[1]-1] = round(rv.cdf(0.875*np.pi) - rv.cdf(0.625*np.pi),2)
            tp[s+self.grid_size[1]-1] = round(rv.cdf(-0.625*np.pi) - rv.cdf(-0.875*np.pi),2)
            if np.sum(tp) < 1:
                tp[s+self.grid_size[1]+1] =  1.0 - np.sum(copy.deepcopy(tp)) 
            else: 
                tp[s+self.grid_size[1]+1] = 0.0
            tp = tp.round(2)
         
        if (not np.all(tp>=0)) or (abs(np.sum(tp) - 1.0) >=  1e-8):
            print(tp)
            raise Exception('Invalid distribution for state {} and action {}'.format(s,a))
        else:
            return tp
            
        
    def _add_reloads(self):    
        """
        Assign reload features to given list of states.
        """        
        for state in self.reloads:
            self.is_reload[state] = 1
                   
            
    def _get_dist(self, state, action):
        """
        Return a dictionary of states with nonzero probabilities for
        a given state and action pair.
        """
        dist = dict()
        agent_dist = self.trans_prob[state, action, :]
        agent_posstates = [] 
        for i in agent_dist.nonzero()[0]:
            agent_posstates.append(i)
        for i in list(agent_posstates):
            prob = agent_dist[i]
            dist[i] = round(prob,2)
        return dist
    

    def allocate_targets(self, allocation_list):
        """
        Assign targets to different agents based on user defined allocation 
        """
                
        for agent in self.agents:
            self.targets_alloc[agent] = allocation_list[agent]
        targets_all = []
        for alloc in self.targets_alloc:
            targets_all += alloc
        if set(targets_all) != set(self.targets):
            raise Exception('Given allocation does not cover all the targets.')

    
    def create_consmdp(self):
        """
        Export the UUV gridworld and target states into a pre-defined
        standard ConsMDP object which is stored in the consmdp attribute
        """
        mdp = ConsMDP()
    
        # Add states to the consMDP object
        for state in self.states:
            if self.is_reload[state]:
                mdp.new_state(True, str(state))  # (reload, label)
            else:
                mdp.new_state(False, str(state))

        # Extract and add actions to the consMDP object
        for state in self.states:
            for action in self.actions:
                dist = self._get_dist(state, action)
                if action in self.weak_actions:
                    mdp.add_action(mdp.state_with_name(str(state)), dist, self.action_to_label[action], self.waction_cost)
                elif action in self.strong_actions:
                    mdp.add_action(mdp.state_with_name(str(state)), dist, self.action_to_label[action], self.saction_cost)
                else:
                    raise Exception('lol')
        self.consmdp = mdp
        
        
    def get_consmdp(self):
        """
        Return the consMDP object and target set that already exists
        or generates the consMDP object if it does not exist and then returns it.
        """
        
        if self.consmdp is None:
            self.create_consmdp()
            return (self.consmdp, self.targets)
        else:
            return (self.consmdp, self.targets)        
        
        
    def update_strategy(self, strategy, agent):
        """
        Update the strategy attribute of a specified agent to a given counter strategy object. 
        """
        self.strategies[agent] = strategy


    def create_counterstrategies(self, solver=GoalLeaningES, objective=BUCHI, threshold=0.1):
        """
        Creates counter strategies for given parameters and the current consMDP object
        and stores them in the self.strategies attribute
        """
        
        if set([t for sublist in self.targets_alloc for t in sublist]) != set(self.targets):
            raise Exception('Target allocation is not complete and does not cover all targets.')
            
        for agent in self.agents:
            if solver == GoalLeaningES:
                slvr = GoalLeaningES(self.consmdp, self.capacities[agent], self.targets_alloc[agent], threshold=threshold)
            elif solver == BasicES:
                slvr = BasicES(self.consmdp, self.capacities[agent], self.targets_alloc[agent])
            selector = slvr.get_selector(objective)
            strategy = CounterStrategy(self.consmdp, selector, self.capacities[agent], self.energies[agent], init_state=self.init_states[agent])
            self.update_strategy(strategy, agent=agent)

        
    def reset(self, init_states=None, reset_energies=None):
        """
        Reset the position of the agents to init_states (if given) or to the 
        originally specified initial state. In the case no initial state was 
        specified while creating the instance, or when init_states="random", 
        reset position of the agent to randomly selected initial states. 
        Also resets the energy to the capacity  of the agents and clears any 
        stored data.
        """
        
        if (init_states is not None) and (init_states!="random"):
            if (len(init_states)!=self.num_agents) or not (all(s in self.states for s in init_states)):
                raise Exception('Invalid input argument init_states. Input should be None, "random", or a list of len num_agents.')
        if (reset_energies is not None):
            if (len(reset_energies)!=self.num_agents) or not (all(e > self.waction_cost for e in reset_energies)):
                raise Exception('reset energy levels for each agent must be greater than weak action cost')
            
        if init_states is not None:
            for idx, init_state in enumerate(init_states):
                if type(init_state) == list or type(init_state) == tuple:
                    init_states[idx] = self.get_state_id(init_state[0], init_state[1])
        else:
            for idx, init_state in enumerate(self.init_states):
                if type(init_state) == list or type(init_state) == tuple:
                    self.init_states[idx] = self.get_state_id(init_state[0], init_state[1])
            
        if init_states is None:
            if self.init_states is not None:
                self.positions = copy.deepcopy(self.init_states)
            else:
                self.positions = np.random.choice(self.states, self.num_agents)
                self.init_states = copy.deepcopy(self.positions)
        elif init_states == "random":
            self.positions = np.random.choice(self.states, self.num_agents)
            self.init_states = copy.deepcopy(self.positions)
        else:
            self.positions = copy.deepcopy(init_states)
            self.init_states = copy.deepcopy(self.positions)
            
        for idx, target in enumerate(self.targets):
            if type(target) == list or type(target) == tuple:
                self.targets[idx] = self.get_state_id(target[0], target[1])
        
        for idx, reload in enumerate(self.reloads):
            if type(reload) == list or type(reload) == tuple:
                self.reloads[idx] = self.get_state_id(reload[0], reload[1])
                   
        if reset_energies is None:
            self.energies = copy.deepcopy(self.capacities)
        else:
            self.energies = reset_energies
            
        for agent in self.agents:
            if self.strategies[agent] is not None:
                self.strategies[agent].reset(init_energy=self.energies[agent], init_state=self.positions[agent])
            else:
                pass
        self.state_histories = [[self.positions[agent]] for agent in self.agents]
        self.action_histories = [[] for agent in self.agents]
        self.target_histories = [[] for agent in self.agents]
        if self.positions[agent] in self.targets_alloc[agent]:
            if self.positions[agent] not in self.target_histories[agent]:
                self.target_histories[agent].append(self.positions[agent])
        self.agent_done = 0
        self.num_timesteps = 0   
                     

    def step(self, strategies=None, states=None, energies=None, do_render=0):
        """
        Select a feasible action to each agent based on its strategy and update
        the position and energy. Returns the resultant state, energy level, 
        and other info. Returns done to show the status of different agents.
        """
        
        if strategies is None:
            strategies = self.strategies
        if states is None:
            states = self.positions
        if energies is None:
            energies = self.energies
        
        # selection action and update states
        actions = [self.label_to_action[strategy.next_action().label] for strategy in self.strategies]
        done = [0 for agent in self.agents]
        
        for agent in self.agents:
            if actions[agent] == -1:
                done[agent] == 1
                continue
            else:
                self.positions[agent] = np.random.choice(self.num_states,
                    p=self.trans_prob[self.positions[agent], actions[agent], :])
                self.strategies[agent].update_state(self.positions[agent])
                self.energies[agent] = self.strategies[agent].energy
        
            # update and render
            self.state_histories[agent].append(self.positions[agent])
            self.action_histories[agent].append(actions[agent])
            if self.positions[agent] in self.targets_alloc[agent]:
                if self.positions[agent] not in self.target_histories[agent]:
                    self.target_histories[agent].append(self.positions[agent])
        self.num_timesteps += 1
        if do_render == 1:
            self.render_grid()
        info = (self.positions, actions, self.energies, done)
        return info
     
        
    def _create_agents_colors(self):
        """
        Create and store colors for visualizing agents
        """
        agentcolor_options = np.random.random(self.num_agents)
        agents_colors = {}
        for agent in self.agents:
            agents_colors[agent] = agentcolor_options[agent]*np.ones(3)
        agents_colors[0] = np.array([0.45,0.45,0.45])
        self.agents_colors = agents_colors      


    def get_state_id(self, x,y):
        """
        Returns the internal state ID for a cell with coordinates (x,y)
        """
        if not (0 <= x < self.grid_size[0]) or not (0 <= y < self.grid_size[1]):
            raise Exception("Input x and y must be valid coordinates in the current grid world")
        
        state_id = x*self.grid_size[1] + y
        return state_id
        
    
    def get_state_coord(self, state_id):
        """
        Returns the coordinates tuple (x,y) of a cell with the input state_id 
        as its state ID.
        """
        if state_id not in self.states:
            raise Exception("The input is not a valid state ID")
        
        x = state_id//self.grid_size[1]
        y = state_id%self.grid_size[1]
        return (x,y)


    def _states_to_colors(self):
        '''
        Assign colors to the cells based on their current identity for visualization
        of the environment and animation of a given policy
        '''
        
        # Define colors
        # 0: light blue; 1 : light gray; 2 : dark gray; 3 : brown; 4 : red; 5: dark blue
        
        COLORS_ENV = {0:np.array([0.678,0.847, 0.902]), 1:np.array([0.54,0.54,0.54]), \
                  2:np.array([0.42,0.42,0.42]), 3:np.array([0.0078,0.5450, 0.0666]), \
                      4:np.array([1.0,0.0,0.0]), 5:np.array([0.1529,0.5019,0.8705])}
        COLORS_AGENTS = self.agents_colors            
        
        data = np.zeros([self.grid_size[0],self.grid_size[1],3],dtype=np.float32)
        data[:] = COLORS_ENV[0] # baseline state color
        for agent in self.agents:
            for cell in self.state_histories[agent]:
                (x,y) = self.get_state_coord(cell)
                data[x, y] = COLORS_AGENTS[agent] # history
        for agent in self.agents:
            (x,y) = self.get_state_coord(self.positions[agent])
            data[x,y] = COLORS_AGENTS[agent] # current state
        for cell in self.targets:
            (x,y) = self.get_state_coord(cell)
            data[x,y] = COLORS_ENV[3] # targets
        for cell in self.reloads:
            (x,y) = self.get_state_coord(cell)
            data[x,y] = COLORS_ENV[4] # reloads

            if self.init_states is not None:
                (x,y) = self.get_state_coord(self.init_states[agent])
                data[x,y] = COLORS_ENV[5] # home/base        
        return data


    def render_grid(self):
        """
        Render the current state of the environment
        """
        
        img_data = self._states_to_colors()
        fig, ax = plt.subplots()
        ax.axis('off')
        energies_strlist = [str(energy) for energy in self.energies]
        energies_str = ", ".join(energies_strlist)
        plt.title("Agent Energy: {}, Time Steps: {}".format(energies_str, self.num_timesteps))
        plt.imshow(img_data) 
        plt.show()
        

    def animate_simulation(self, strategies=None, num_steps=100, interval=100):
        """
        Execute the strategies for num_steps number of time steps and animates the
        resultant trajectory.
        """
        
        if strategies is not None:
            self.strategies = strategies
        self.reset(self.init_states)
        fig = plt.figure()
        ax = fig.gca()
        ax.axis('off')
        im = plt.imshow(self._states_to_colors(), animated=True)
        plt.close()
        
        def updatefig(frame_count):
            if frame_count == 0:
                im.set_array(self._states_to_colors())
                energies_strlist = [str(energy) for energy in self.energies]
                energies_str = ", ".join(energies_strlist)
                ax.set_title("Agent Energy: {}, Time Steps: {}".format(energies_str, self.num_timesteps))
                return im
            self.step()
            im.set_array(self._states_to_colors())
            energies_strlist = [str(energy) for energy in self.energies]
            energies_str = ", ".join(energies_strlist)
            ax.set_title("Agent Energy: {}, Time Steps: {}".format(energies_str, self.num_timesteps))
            return im
        return animation.FuncAnimation(fig, updatefig, frames=num_steps, interval=interval)


    def _repr_png_(self):
        """
        Show graphical representation in notebooks.
        """
        return self.render_grid()
    
    


class SingleAgentEnv(SynchronousMultiAgentEnv):
    """Class that models a Markov decision process where the agent and the environment model the dynamics of a UUV and 
        the ocean currents it is operating in. This class is heavily inherits from the SynchronousMultiAgentEnv class. 
    """
    
    def __init__(self, grid_size, capacity, reloads, targets, init_state=None, enhanced_actionspace=0, weakaction_cost=1, strongaction_cost=2, velocity=5, heading_sd=0.524):
        if init_state is None:
            init_states = None
        else:
            init_states = [init_state]
        super().__init__(1, grid_size, [capacity], reloads, targets, init_states, enhanced_actionspace, weakaction_cost, strongaction_cost, velocity, heading_sd)
        self.allocate_targets([targets])
       
    def update_strategy(self, strategy, **kwargs):
        agent = kwargs.get('agent', None)
        if agent is None:
            super().update_strategy(strategy, 0)
        else:
            super().update_strategy(strategy, agent)
        
    def create_counterstrategy(self, solver=GoalLeaningES, objective=BUCHI, threshold=0.1):
        super().create_counterstrategies(solver, objective, threshold)
      
    def reset(self, init_state=None, reset_energy=None):
        if init_state is None:
            init_states = None
        elif init_state is self.init_states:
            init_states = self.init_states
        else:
            init_states = [init_state]
        if reset_energy is None:
            reset_energies = None
        else:
            reset_energies = [reset_energy]
        super(SingleAgentEnv, self).reset(init_states, reset_energies)
        self.allocate_targets([self.targets])
        
    def step(self, strategy=None, state=None, energy=None, do_render=0):
        if strategy is None:
            strategies = None
        else:
            strategies = [strategy]
        if state is None:
            states = None
        else:
            states = [state]
        if energy is None:
            energies = None
        else:
            energies = [energy]
        return super().step(strategies, states, energies)
    
    
    def animate_simulation(self, strategy=None, num_steps=100, interval=100):
        if strategy is None:
            return super().animate_simulation(strategy, num_steps, interval)
        else:
            return super().animate_simulation([strategy], num_steps, interval)
        
        
