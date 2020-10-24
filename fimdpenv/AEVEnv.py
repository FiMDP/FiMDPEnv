"""
Simulation environment modeling the stochastic energy consumption of an AEV
traversing the streets in a given region of interest. Requires energy
consumption and street network data as inputs
"""

import json
import time
import folium
import numpy as np
import networkx as nx
from folium import plugins
from fimdp.core import ConsMDP, CounterStrategy
from fimdp.energy_solvers import GoalLeaningES, BasicES
from fimdp.objectives import AS_REACH, BUCHI, MIN_INIT_CONS, POS_REACH, SAFE


class AEVEnv:
    
    def __init__(self, capacity, targets, reloads=None, init_state=None, datafile='NYC.json', mapfile='NYC.graphml'):
        """
        Class that models the stochastic energy consumption in routing problem
        as a consumption Markov decision process. 
        """
        
        # initialize environment attributes
        self.num_states = None
        self.num_actions = None
        self.states = []
        self.actions = {}
        self.reloads = reloads
        self.targets = targets
        self.consmdp = None
        self.datafile = datafile
        self.mapfile = mapfile
        self.state_to_name = None
        
        # agent attributes
        self.init_state = init_state
        self.capacity = capacity
        self.energy = None
        self.position = None
        self.strategy = None
        
        # logging attributes
        self.state_history = []
        self.action_history = []
        self.target_history = []
        self.energy_history = []
        self.num_timesteps = 0        
    
        # initialize environment
        self.create_consmdp()
        self.reset()

    
    def create_consmdp(self):
        """
        Create consumption MDP object from given transition and energy
        consumption data
        """
        cmdp = ConsMDP()
        states = []
        actions = dict()
        
        with open(self.datafile,'r') as f:
            g = json.load(f)
            
        for node in g["nodes"]:
            if node["action"]:
                actions[node["label"]] = dict()
            else:
                states.append(node)
        
        if self.reloads is None:
            reloads = []
            for s in states:
                if s["reload"]:
                    reloads.append(s["label"])
            self.reloads = reloads
            
        for s in states:
            if s["label"] in self.reloads:
                cmdp.new_state(True, s["label"])
            else:
                cmdp.new_state(False, s["label"])
                
        for edge in g["edges"]:
            fr = edge["tail"]
            to = edge["head"]
            if to in actions:
                actions[to]["from"] = fr
                actions[to]["cons"] = edge["consumption"]
            else:
                dist = actions[fr].get("dist")
                to = cmdp.state_with_name(to)
                if dist is None:
                    actions[fr]["dist"] = dict()
                    from decimal import Decimal
                actions[fr]["dist"][to] = Decimal(f'{edge["probability"]}')
                
        for label, a in actions.items():
            fr = cmdp.state_with_name(a["from"])
            cmdp.add_action(fr, a["dist"], label, a["cons"])
        
        self.states = states
        self.name_to_state = cmdp.names_dict
        self.state_to_name = {v: k for k, v in cmdp.names_dict.items()}
        self.actions = actions
        self.consmdp = cmdp
        
    
    def reset(self, init_state=None, init_energy=None):
        """
        Reset the position to init_state (if given) and the energy to init_energy
        (if given). Also resets the histories and the time step count.
        """
        
        if init_state is None:
            if self.init_state is not None:
                self.position = self.init_state
            else:
                self.position = np.random.choice(self.states)
        else:
            self.position = init_state
        
        if init_energy is None:
            self.energy = self.capacity
        else: self.energy = init_energy
            
        self.position = self.init_state
        self.energy = self.capacity
        if self.strategy is not None:
            self.strategy.reset(init_energy=self.energy, init_state=self.name_to_state[self.position])
        
        # reset histories
        self.state_history = [self.position]
        self.action_history = []
        self.target_history = []
        self.energy_history = [self.energy]
        self.num_timesteps = 0

        
    def get_consmdp(self):
        """
        Returns the consMDP object and target set that already exists
        or generates the consMDP object if it does not exist and then returns it.
        """
        
        if self.consmdp is None:
            self.create_consmdp()
            
        targets_consmdp = []
        for item in self.targets:
            targets_consmdp.append(self.consmdp.state_with_name(item))  
        return (self.consmdp, targets_consmdp)
    

    def update_strategy(self, strategy):
        '''
        Update the strategy attribute to the given strategy
        '''
        self.strategy = strategy
    
    
    def create_counterstrategy(self, solver=GoalLeaningES, objective=BUCHI, threshold=0.1):
        """
        Creates a counter stategy for the current ConsMDP using the given
        parameters and stores it as the strategy attribute.
        """
        
        consmdp, targets_consmdp = self.get_consmdp()
        
        if solver == GoalLeaningES:
            slvr = GoalLeaningES(consmdp, self.capacity, targets_consmdp, threshold=threshold)
        elif solver == BasicES:
            slvr = BasicES(consmdp, self.capacity, targets_consmdp)
        selector = slvr.get_selector(objective)
        strategy = CounterStrategy(consmdp, selector, self.capacity, self.energy, init_state=self.consmdp.state_with_name(self.init_state))
        self.update_strategy(strategy)
        
        
    def step(self, strategy=None, state=None, energy=None, do_render=0):
        """
        Select a feasible action based on its strategy and update
        the position and energy. Returns the resultant state, energy level, 
        and other info.
        """
        
        if strategy is None:
            strategy = self.strategy
        if state is None:
            state = self.position
        if energy is None:
            energy = self.energy
        
        # selection action and update states
        action = self.strategy.next_action()
        next_state = np.random.choice(list(action.distr.keys()), p=list(action.distr.values()))
        self.strategy.update_state(next_state)
        self.position = self.state_to_name[next_state]
        self.energy = self.strategy.energy
        
        # log data and render
        self.state_history.append(self.position)
        self.action_history.append(action.label)
        self.energy_history.append(self.energy)
        if self.position in self.targets:
            self.target_history.append(self.position)
        self.num_timesteps += 1
        if do_render == 1:
            self.render_map()
        info = (self.position, action.label, self.energy)
        return info
    
        
        
    def _repr_html_(self):
        """
        Show graphical representation in notebooks.
        """
        return self.render_map()
        
    
    def render_map(self):
        """
        Renders the environment map with locations of initial state, target, 
        and reload states. Also includes the trajectory of the agent. 
        """
        
        targets = self.targets
        init_state = self.init_state
        reloads = self.reloads
        def is_int(s):
            try: 
                int(s)
                return True
            except ValueError:
                return False
        trajectory = []
        for state in self.state_history:
            if is_int(state):
                trajectory.append(state)
            else:
                pass
        
        # Load NYC Geodata
        G = nx.MultiDiGraph(nx.read_graphml(self.mapfile))
        for _, data in G.nodes(data=True):
            data['lat'] = float(data['lat'])
            data['lon'] = float(data['lon'])
        for reload in reloads:
            if reload not in list(G.nodes):
                reloads.remove(reload)
        for target in targets:
            if target not in list(G.nodes):
                targets.remove(target)
        for point in trajectory:
            if point not in list(G.nodes):
                trajectory.remove(point)
        
        # create baseline map
        nodes_all = {}
        for node in G.nodes.data():
            name = str(node[0])
            point = [node[1]['lat'], node[1]['lon']]
            nodes_all[name] = point
        global_lat = []; global_lon = []
        for name, point in nodes_all.items():
            global_lat.append(point[0])
            global_lon.append(point[1])
        min_point = [min(global_lat), min(global_lon)]
        max_point =[max(global_lat), max(global_lon)]
        m = folium.Map(zoom_start=1, tiles='cartodbpositron')
        m.fit_bounds([min_point, max_point])
        
        # plot state history    
        if len(trajectory) < 2:       
            pass
        else:
            path = list(zip(trajectory[:-1], trajectory[1:]))
            points = []
            for pair in path:
                    points.append([G.nodes[pair[0]]['lat'], G.nodes[pair[0]]['lon']])
                    points.append([G.nodes[pair[1]]['lat'], G.nodes[pair[1]]['lon']])
            folium.PolyLine(locations=points, color="black", weight=2).add_to(m)            
    
        # add initial state, reload states and target states
        folium.CircleMarker(location=[G.nodes[init_state]['lat'], G.nodes[init_state]['lon']],
                        radius= 3,
                        popup = 'initial state: ' + init_state,
                        color='green',
                        fill_color = 'green',
                        fill_opacity=1,
                        fill=True).add_to(m)        
        for node in reloads:
            folium.CircleMarker(location=[G.nodes[node]['lat'], G.nodes[node]['lon']],
                        radius= 1,
                        popup = 'reload state: ' + node,
                        color="#0f89ca",
                        fill_color = "#0f89ca",
                        fill_opacity=1,
                        fill=True).add_to(m)
        for node in targets:
            folium.CircleMarker(location=[G.nodes[node]['lat'], G.nodes[node]['lon']],
                        radius= 3,
                        popup = 'target state: ' + node,
                        color="red",
                        fill_color = "red",
                        fill_opacity=1,
                        fill=True).add_to(m)
        return m
        

    def animate_simulation(self, strategy=None, num_steps=100, interval=100):
        """
        Obtain the animation of a simulation instance where the agent reaches
        the target state from the initial state using assigned counterstrategy
        """
        
        if strategy is not None:
            self.strategy = strategy
        if self.strategy is None:
            self.create_counterstrategy()
        self.reset(self.init_state)
        
        for i in range(num_steps):
            self.step()
            
        # extract data
        targets = self.targets
        init_state = self.init_state
        reloads = self.reloads
        def is_int(s):
            try: 
                int(s)
                return True
            except ValueError:
                return False
        trajectory = []
        for state in self.state_history:
            if is_int(state):
                trajectory.append(state)
            else:
                pass
        
        # Load NYC Geodata
        G = nx.MultiDiGraph(nx.read_graphml(self.mapfile))
        for _, _, data in G.edges(data=True, keys=False):
            data['time_mean'] = float(data['time_mean'])
        for _, data in G.nodes(data=True):
            data['lat'] = float(data['lat'])
            data['lon'] = float(data['lon'])
        for reload in reloads:
            if reload not in list(G.nodes):
                reloads.remove(reload)
        for target in targets:
            if target not in list(G.nodes):
                targets.remove(target)
        for point in trajectory:
            if point not in list(G.nodes):
                trajectory.remove(point)
        
        # create baseline map
        nodes_all = {}
        for node in G.nodes.data():
            name = str(node[0])
            point = [node[1]['lat'], node[1]['lon']]
            nodes_all[name] = point
        global_lat = []; global_lon = []
        for name, point in nodes_all.items():
            global_lat.append(point[0])
            global_lon.append(point[1])
        min_point = [min(global_lat), min(global_lon)]
        max_point =[max(global_lat), max(global_lon)]
        m = folium.Map(zoom_start=1, tiles='cartodbpositron')
        m.fit_bounds([min_point, max_point])        
            
        # add initial state, reload states and target states
        folium.CircleMarker(location=[G.nodes[init_state]['lat'], G.nodes[init_state]['lon']],
                        radius= 3,
                        popup = 'initial state',
                        color='green',
                        fill_color = 'green',
                        fill_opacity=1,
                        fill=True).add_to(m)        
        for node in reloads:
            folium.CircleMarker(location=[G.nodes[node]['lat'], G.nodes[node]['lon']],
                        radius= 1,
                        popup = 'reload state',
                        color="#0f89ca",
                        fill_color = "#0f89ca",
                        fill_opacity=1,
                        fill=True).add_to(m)
        for node in targets:
            folium.CircleMarker(location=[G.nodes[node]['lat'], G.nodes[node]['lon']],
                        radius= 3,
                        popup = 'target state',
                        color="red",
                        fill_color = "red",
                        fill_opacity=1,
                        fill=True).add_to(m)

        # Baseline time
        t = time.time()
        path = list(zip(trajectory[:-1], trajectory[1:]))
        lines = []
        current_positions = []
        for pair in path:
            
            t_edge = 1
            lines.append(dict({'coordinates':
                [[G.nodes[pair[0]]['lon'], G.nodes[pair[0]]['lat']],
                [G.nodes[pair[1]]['lon'], G.nodes[pair[1]]['lat']]],
                'dates': [time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t)),
                           time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t+t_edge))],
                           'color':'black'}))
            current_positions.append(dict({'coordinates':[G.nodes[pair[1]]['lon'], G.nodes[pair[1]]['lat']],
                        'dates': [time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t+t_edge))]}))
            t = t+t_edge

        features = [{'type': 'Feature',
                     'geometry': {
                                'type': 'LineString',
                                'coordinates': line['coordinates'],
                                 },
                     'properties': {'times': line['dates'],
                                    'style': {'color': line['color'],
                                              'weight': line['weight'] if 'weight' in line else 2
                                             }
                                    }
                     }
                for line in lines]
        
        positions = [{
            'type': 'Feature',
            'geometry': {
                        'type':'Point', 
                        'coordinates':position['coordinates']
                        },
            'properties': {
                'times': position['dates'],
                'style': {'color' : 'white'},
                'icon': 'circle',
                'iconstyle':{
                    'fillColor': 'white',
                    'fillOpacity': 1,
                    'stroke': 'true',
                    'radius': 2
                }
            }
        }
         for position in current_positions]
        data_lines = {'type': 'FeatureCollection', 'features': features}
        data_positions = {'type': 'FeatureCollection', 'features': positions}
        folium.plugins.TimestampedGeoJson(data_lines,  transition_time=interval,
                               period='PT1S', add_last_point=False, date_options='mm:ss', duration=None).add_to(m)      
        return m