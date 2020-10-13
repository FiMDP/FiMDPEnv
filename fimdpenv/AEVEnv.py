"""
Simulation environment modeling the stochastic energy consumption of an AEV
traversing the streets in a given region of interest. Requires energy
consumption and street network data as inputs
"""

# import required packages
import os
import ast
import json
import time
import folium
import timeit
import numpy as np
import networkx as nx
from decimal import Decimal
from folium import plugins
from fimdp import consMDP
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class AEVEnv:
    
    def __init__(self, capacity, target, init_state, datafile_name='NYCdata.json'):
        """
        class that models the stochastic energy consumption in routing problem
        as a Markov decision process. 
        """
        
        # define environment parameters
        self.filename = datafile_name
        self.capacity = capacity
        self.energy = capacity
        self.reload = None
        self.init_state = init_state
        self.states = []
        self.actions = {}
        self.target = target
        self.reload = []
        self.position = None
        self.strategy = None
        self.consmdp = None
        self.state_history = []
        self.action_history = []
        self.target_history = []
        self.consumption_history = []
        self.num_timesteps = None        
        

    def update_strategy(self, strategy):
        '''
        Update the strategy attribute to the given strategy
        '''
        self.strategy = strategy

    
    def create_consmdp(self):
        """
        """
        mdp = consMDP.ConsMDP()
        states = []
        actions = dict()
        with open(self.filename,'r') as f:
            g = json.load(f)
        
        
        for node in g["nodes"]:
            if node["action"]:
                actions[node["label"]] = dict()
            else:
                states.append(node)
            
        for s in states:
            mdp.new_state(s["reload"], s["label"])
                
        for edge in g["edges"]:
            fr = edge["tail"]
            to = edge["head"]
            if to in actions:
                actions[to]["from"] = fr
                actions[to]["cons"] = edge["consumption"]
            else:
                dist = actions[fr].get("dist")
                to = mdp.state_with_name(to)
                if dist is None:
                    actions[fr]["dist"] = dict()
                actions[fr]["dist"][to] = Decimal(f'{edge["probability"]}')
                
        for label, a in actions.items():
            fr = mdp.state_with_name(a["from"])
            mdp.add_action(fr, a["dist"], label, a["cons"])
        
        self.consmdp = mdp
        
        
    def get_consmdp(self):
        """
        Returns the consMDP object and target set that already exists
        or generates the consMDP object if it does not exist and then returns it.
        """
        
        if self.consmdp == None:
            self.create_consmdp()
            
        new_targets = []
        for item in self.target:
            new_targets.append(self.consmdp.state_with_name(item))     
            
        return (self.consmdp, new_targets)
   
        
    def update_environment(self, strategy):
        """
        Adds reload states, actual states, new strategy to the environment. 
        """

        # Initialize
        states_original = []
        states_nostrategy = [] 
        states_reload = [] 
        strategy_updated = {}
        map_statelabels = self.consmdp.names
    
        # Map states to original labels
        for index in range(self.consmdp.num_states):
            if not map_statelabels[index][:2] == 'ps':
                states_original.append(map_statelabels[index])
                if strategy[index]:
                    strategy_updated.update({map_statelabels[index]: strategy[index]})
                else:
                    strategy_updated.update({map_statelabels[index]: {}})
                    states_nostrategy.append(map_statelabels[index])

        # Map reload states to original labels and update instance attributes
        for index, item in enumerate(self.consmdp.reloads):
            if item:
                states_reload.append(map_statelabels[index])
        self.states = states_original
        self.reload = states_reload
                
        # Extract resultant states for actions
        dynamics = {}
        with open('NYCdata.json','r') as f:
            raw_data = json.load(f)
            for edge in raw_data["edges"]:
                tail = edge["tail"]
                head = edge["head"]
                if tail[:2] == 'pa':
                    action_label = tail[4:]
                    dynamics.update({action_label:head})
                
        # Map actions in strategy to resultant states
        for key, value in strategy_updated.items():
            if len(value) == 0:
                pass
            else:
                for energy, action_label in value.items():
                    value.update({energy: dynamics[action_label]})
    
        # update strategy
        self.strategy = strategy_updated
    
        
    def visualize_strategy(self):
        """
        Given initial state and target simulate and visualize strategy
        """
        
        def find_next_state(state):
            data_dict = self.strategy[state]
            dict_keys = list(data_dict.keys())
            if len(dict_keys) == 0:
                raise Exception('Strategy does not prescribe a safe action at this state. Increase the capacity of the agent.')
            feasible = []
            for value in dict_keys:
                if value <= self.energy:
                    feasible.append(value)
            if len(feasible) == 0:
                raise Exception('Strategy does not prescribe a safe action at this energy level. Increase the capacity of the agent.')
            next_state = data_dict[max(feasible)]
            return next_state
        
        
        # generate nodes in the path from initial state to target using strategy
        current_state = self.init_state
        while True:
            
            self.state_history.append(current_state)
            current_state = find_next_state(current_state)
            
            if current_state == self.target[0]:
                break
            
        
        # Load NYC Geodata
        path = os.path.abspath("nyc.graphml")
        G = nx.MultiDiGraph(nx.read_graphml(path))
        for _, _, data in G.edges(data=True, keys=False):
            data['speed_mean'] = float(data['speed_mean'])
            data['speed_sd'] = float(data['speed_sd'])
            data['time_mean'] = float(data['time_mean'])
            data['time_sd'] = float(data['time_sd'])
            data['energy_levels'] = ast.literal_eval(data['energy_levels'])
        for _, data in G.nodes(data=True):
            data['reload'] = ast.literal_eval(data['reload'])
            data['lat'] = float(data['lat'])
            data['lon'] = float(data['lon'])
        
        
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
        
        
        # add path to the map
#        path_pairwise = list(zip(self.state_history[:-1], self.state_history[1:]))
        points = []
        for item in self.state_history:
            
            for node in G.nodes.data():
                if node[0] == item:
                    name = str(node[0])
                    point = [node[1]['lat'], node[1]['lon']]
                    points.append(point)
            
        folium.PolyLine(locations=points, color="black", weight=2).add_to(m)

    
        
        # add reload state, initial state, target state
        nodes_reload = {}
        nodes_target = {}
        nodes_init = {}
        for node in G.nodes.data():
            if node[0] in self.reload:
                name = str(node[0])
                point = [node[1]['lat'], node[1]['lon']]
                nodes_reload[name] = point
            if node[0] in self.target:
                name = str(node[0])
                point = [node[1]['lat'], node[1]['lon']]
                nodes_target[name] = point
            if node[0] == self.init_state:
                name = str(node[0])
                point = [node[1]['lat'], node[1]['lon']]
                nodes_init[name] = point

        # Plot reload states
        for node_name, node_point in nodes_reload.items():
            folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 1,
                        popup = 'reload state',
                        color="#0f89ca",
                        fill_color = "#0f89ca",
                        fill_opacity=1,
                        fill=True).add_to(m)
        # Plot target nodes
        for node_name, node_point in nodes_target.items():
            folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'target state',
                        color="red",
                        fill_color = "red",
                        fill_opacity=1,
                        fill=True).add_to(m)
        # Plot initial nodes
        for node_name, node_point in nodes_init.items():
            folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'initial state',
                        color='green',
                        fill_color = 'green',
                        fill_opacity=1,
                        fill=True).add_to(m)
            
        return m
        
        
    def animate_strategy(self):
        """
        Given initial state and target simulate and visualize strategy
        """
        
        def find_next_state(state):
            data_dict = self.strategy[state]
            dict_keys = list(data_dict.keys())
            if len(dict_keys) == 0:
                raise Exception('Strategy does not prescribe a safe action at this state. Increase the capacity of the agent.')
            feasible = []
            for value in dict_keys:
                if value <= self.energy:
                    feasible.append(value)
            if len(feasible) == 0:
                raise Exception('Strategy does not prescribe a safe action at this energy level. Increase the capacity of the agent.')
            next_state = data_dict[max(feasible)]
            return next_state
        
        
        # generate nodes in the path from initial state to target using strategy
        current_state = self.init_state
        while True:
            
            self.state_history.append(current_state)
            current_state = find_next_state(current_state)
            
            if current_state == self.target[0]:
                break
            
        
        # Load NYC Geodata
        path = os.path.abspath("nyc.graphml")
        G = nx.MultiDiGraph(nx.read_graphml(path))
        for _, _, data in G.edges(data=True, keys=False):
            data['speed_mean'] = float(data['speed_mean'])
            data['speed_sd'] = float(data['speed_sd'])
            data['time_mean'] = float(data['time_mean'])
            data['time_sd'] = float(data['time_sd'])
            data['energy_levels'] = ast.literal_eval(data['energy_levels'])
        for _, data in G.nodes(data=True):
            data['reload'] = ast.literal_eval(data['reload'])
            data['lat'] = float(data['lat'])
            data['lon'] = float(data['lon'])
        
        
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
        
        # add reload state, initial state, target state
        nodes_reload = {}
        nodes_target = {}
        nodes_init = {}
        for node in G.nodes.data():
            if node[0] in self.reload:
                name = str(node[0])
                point = [node[1]['lat'], node[1]['lon']]
                nodes_reload[name] = point
            if node[0] in self.target:
                name = str(node[0])
                point = [node[1]['lat'], node[1]['lon']]
                nodes_target[name] = point
            if node[0] == self.init_state:
                name = str(node[0])
                point = [node[1]['lat'], node[1]['lon']]
                nodes_init[name] = point

        # Plot reload states
        for node_name, node_point in nodes_reload.items():
            folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 1,
                        popup = 'reload state',
                        color="#0f89ca",
                        fill_color = "#0f89ca",
                        fill_opacity=1,
                        fill=True).add_to(m)
        # Plot target nodes
        for node_name, node_point in nodes_target.items():
            folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'target state',
                        color="red",
                        fill_color = "red",
                        fill_opacity=1,
                        fill=True).add_to(m)
        # Plot initial nodes
        for node_name, node_point in nodes_init.items():
            folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'initial state',
                        color='green',
                        fill_color = 'green',
                        fill_opacity=1,
                        fill=True).add_to(m)
            
            
            
        # Baseline time
        t = time.time()
        path = list(zip(self.state_history[:-1], self.state_history[1:]))
        lines = []
        for pair in path:
            
            points_data = {}
            for node in G.nodes.data():
                if node[0] == pair[0]:
                    point = [node[1]['lat'], node[1]['lon']]
                    points_data['1'] = point
                if node[0] == pair[1]:
                    point = [node[1]['lat'], node[1]['lon']]
                    points_data['2'] = point
            
            
            

            # Time of the edge
            t_edge = 10 #G.edges[int(pair[0]), int(pair[1]),0]['time_mean']
            lines.append(dict({'coordinates':
                    [points_data['1'], points_data['2']],
                'dates': [time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t)),
                           time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t+t_edge))],
                           'color':'black'}))
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


        data = {'type': 'FeatureCollection', 'features': features}
        folium.plugins.TimestampedGeoJson(data,  transition_time=1,
                               period='PT1S', add_last_point=False, date_options='mm:ss').add_to(m)     
    
        
        return m

def create_geoplot(states_nostrategy, states_reload, states_target, strategy_updated):
    
    """
    Helper function to visualize the data from MDP and strategy on an OpenStreetMap.
    Uses geodata of the street network stored as a graph in 'nyc.graphml' file.
    """
    
    # Load NYC Geodata
    path = os.path.abspath("nyc.graphml")
    G = nx.MultiDiGraph(nx.read_graphml(path))
    for _, _, data in G.edges(data=True, keys=False):
        data['speed_mean'] = float(data['speed_mean'])
        data['speed_sd'] = float(data['speed_sd'])
        data['time_mean'] = float(data['time_mean'])
        data['time_sd'] = float(data['time_sd'])
        data['energy_levels'] = ast.literal_eval(data['energy_levels'])
    for _, data in G.nodes(data=True):
        data['reload'] = ast.literal_eval(data['reload'])
        data['lat'] = float(data['lat'])
        data['lon'] = float(data['lon'])

    # Create baseline map with edges 
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
    
    
    for edge in G.edges:
        points = [(G.nodes[edge[0]]['lat'], G.nodes[edge[0]]['lon']),
                  (G.nodes[edge[1]]['lat'], G.nodes[edge[1]]['lon'])]
        folium.PolyLine(locations=points,
                        color='gray',
                        weight=2,
                        opacity=0.8).add_to(m)
    
    for key, value in strategy_updated.items():
        color = '#2f2f2f'
        for energy, end_state in value.items():
            points = [(G.nodes[key]['lat'], G.nodes[key]['lon']),
                      (G.nodes[end_state]['lat'], G.nodes[end_state]['lon'])]
            line = folium.PolyLine(locations=points,
                                   color=color,
                                   tooltip=str(energy),
                                   weight=1.5).add_to(m)
            attr = {'fill': color, 'font-size': '12'}
            plugins.PolyLineTextPath(line,'\u25BA',
                                     repeat=False,
                                     center=True,
                                     offset=3.5,
                                     attributes=attr).add_to(m)
            folium.CircleMarker(location=[G.nodes[key]['lat'], G.nodes[key]['lon']],
                        radius= 2,
                        color=color,
                        fill=True).add_to(m)
    
    # Add reload states, target states, and states with no prescribed action
    nodes_reload = {}
    nodes_target = {}
    nodes_nostrategy = {}
    for node in G.nodes.data():
        if node[0] in states_reload:
            name = str(node[0])
            point = [node[1]['lat'], node[1]['lon']]
            nodes_reload[name] = point
        if node[0] in states_target:
            name = str(node[0])
            point = [node[1]['lat'], node[1]['lon']]
            nodes_target[name] = point
        if node[0] in states_nostrategy:
            name = str(node[0])
            point = [node[1]['lat'], node[1]['lon']]
            nodes_nostrategy[name] = point

    # Plot reload states
    for node_name, node_point in nodes_reload.items():
        folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'reload state',
                        color="#22af4b",
                        fill_color = "#22af4b",
                        fill_opacity=1,
                        fill=True).add_to(m)
    # Plot target nodes
    for node_name, node_point in nodes_target.items():
        folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'target state',
                        color="#0f89ca",
                        fill_color = "#0f89ca",
                        fill_opacity=1,
                        fill=True).add_to(m)
    # Plot no strategy nodes
    for node_name, node_point in nodes_nostrategy.items():
        folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'no guarantees state',
                        color='red',
                        fill_color = 'red',
                        fill_opacity=1,
                        fill=True).add_to(m)
    return m






        
        