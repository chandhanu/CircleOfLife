import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from CreateEnvironment import Graph_util, Node
import networkx as nx
import random
#random.seed(13) # Predator wins 
#random.seed(1) # Agent wins 
import numpy as np
from PreyPredator import Prey, Predator
import collections
import heapq as hq
import json
import sys 

DEBUG = False
SUCCESS = True 
FAILURE = False 

def print_my_dict(d, msg = "dict"):
    if DEBUG:
        print("--------"+msg+" ------------")
        print("Cost ", "Neighbor", "Path ")
        for k,v in d.items():
            for i in v:
                print(k , " ", i[0], "    ", i[1] )

def print_my_dict2(d, msg = "dict"):
    if DEBUG: 
        print("--------"+msg+" ------------")
        print("Neighbor", "Cost  ", "Path ")
        for k,v in d.items():
            print(k , "      ", v[0], "    ", v[1] )

def get_data(graph, prey, predator, agent):
    
    all_shortest_path_cost =  graph.all_shortest_paths() #floyd-warshall 
    all_pairs_shortest_path = dict(nx.all_pairs_shortest_path(graph.G))
    agent_to_prey = {}
    agent_to_predator = {}
    agent_neighbors = list(graph.G.neighbors(agent))
    agent_to_prey_by_cost = {}
    agent_to_predator_by_cost = {}
    for neighbor in agent_neighbors:
        agent_to_prey[neighbor] = (all_shortest_path_cost[neighbor][prey], all_pairs_shortest_path[neighbor][prey])
        agent_to_predator[neighbor] = (all_shortest_path_cost[neighbor][predator], list(reversed(all_pairs_shortest_path[neighbor][predator])))
        # New dict by cost from agent to prey by cost 
        if all_shortest_path_cost[neighbor][prey] in agent_to_prey_by_cost:
            agent_to_prey_by_cost[all_shortest_path_cost[neighbor][prey]].append((neighbor, all_pairs_shortest_path[neighbor][prey]))
        else:
            agent_to_prey_by_cost[all_shortest_path_cost[neighbor][prey]] = []
            agent_to_prey_by_cost[all_shortest_path_cost[neighbor][prey]].append((neighbor, all_pairs_shortest_path[neighbor][prey]))
        # New dict by cost from agent to predator by cost 
        if all_shortest_path_cost[neighbor][predator] in agent_to_predator_by_cost:
            agent_to_predator_by_cost[all_shortest_path_cost[neighbor][predator]].append((neighbor, list(reversed(all_pairs_shortest_path[neighbor][predator]))))
        else:
            agent_to_predator_by_cost[all_shortest_path_cost[neighbor][predator]] = []
            agent_to_predator_by_cost[all_shortest_path_cost[neighbor][predator]].append((neighbor, list(reversed(all_pairs_shortest_path[neighbor][predator]))))
    
    agent_to_prey_by_cost = {k: v for k, v in sorted(agent_to_prey_by_cost.items(), key=lambda item: item[0])}
    agent_to_predator_by_cost = {k: v for k, v in sorted(agent_to_predator_by_cost.items(), key=lambda item: item[0], reverse=True)}
    
    print_my_dict(agent_to_prey_by_cost, "agent_to_prey_by_cost")
    print_my_dict(agent_to_predator_by_cost, "agent_to_predator_by_cost")
    ''' DATA
    Prey     :  1
    Predator :  13
    Agent    :  18
    Neighbor(18): [17, 19, 23, 18]

    --------agent_to_prey_by_cost ------------
    Cost  Neighbor Path 
    4.0    23      [23, 22, 27, 0, 1]
    5.0    18      [18, 23, 22, 27, 0, 1]
    6.0    17      [17, 18, 23, 22, 27, 0, 1]
    6.0    19      [19, 18, 23, 22, 27, 0, 1]

    --------agent_to_predator_by_cost ------------
    Cost  Neighbor Path 
    4.0    19      [13, 12, 17, 18, 19]
    4.0    23      [13, 12, 17, 18, 23]
    3.0    18      [13, 12, 17, 18]
    2.0    17      [13, 12, 17]
    '''
    # Not needed now below
    agent_2_prey_predator = {}
    for neighbor in agent_neighbors: 
        agent_2_prey_predator[neighbor] = {
            "prey" : agent_to_prey[neighbor],
            "predator": agent_to_predator[neighbor]
        }

    agent_to_prey = {k: v for k, v in sorted(agent_to_prey.items(), key=lambda item: item[1][0])}
    print_my_dict2(agent_to_prey, "agent_to_prey")
    predator_to_agent = {k: v for k, v in sorted(agent_to_predator.items(), key=lambda item: item[1][0], reverse = True)}
    print_my_dict2(predator_to_agent, "predator_to_agent")

    return agent_to_prey, predator_to_agent,all_shortest_path_cost
            
def get_neighbor_data(graph, prey, predator, agent):
    return get_data(graph, prey, predator, agent)


class Agent1(Graph_util):
    '''
    Agent 1 will find the shortest distance between itself and the prey to catch the prey.
    Also it will avoid the predator. 
    '''
    def __init__(self, graph, node_count, prey, predator):
        self.graph = graph
        node_list = [i for i in range(node_count)]
        node_list.remove(graph.predator) # not produce an agent in the same position as predator
        self.value = random.choice([i for i in range(node_count)])
        self.node = graph.G.nodes[self.value]
        graph.agent = self.value
        self.path = []
        self.path.append(self.value)
    
    def move(self, prey_pos, predator_pos):
        agent_to_prey, predator_to_agent,all_shortest_path_cost = get_neighbor_data(self.graph, prey_pos, predator_pos, self.value)
        agent_pos = self.value
        agent_to_prey_cost = all_shortest_path_cost[agent_pos][prey_pos]
        agent_to_predator_cost = all_shortest_path_cost[agent_pos][predator_pos]
        x = agent_to_predator_cost
        y = agent_to_prey_cost
        agent_neighbors = list(self.graph.G.neighbors(agent_pos))
        next_node_list = []
        next_node_priority = { # This sets the priority by the 7 conditions for Agents moves
            1:[],
            2:[],
            3:[],
            4:[],
            5:[],
            6:[],
            7:[]
        }
        for neighbor in agent_neighbors:
            neighbor_cost_to_prey,  neighbor_path_to_prey = agent_to_prey[neighbor][0],agent_to_prey[neighbor][1]
            neighbor_cost_to_predator, neighbor_path_to_predator = predator_to_agent[neighbor][0],predator_to_agent[neighbor][1]
            # 1. Neighbors that are closer to the Prey and farther from the Predator.
            if neighbor_cost_to_prey<y and neighbor_cost_to_predator>x:
                next_node_priority[1].append(neighbor)
            # 2. Neighbors that are closer to the Prey and not closer to the Predator.
            elif neighbor_cost_to_prey<y and neighbor_cost_to_predator==x:
                next_node_priority[2].append(neighbor)
            # 3. Neighbors that are not farther from the Prey and farther from the Predator.
            elif neighbor_cost_to_prey==y and neighbor_cost_to_predator>x:
                next_node_priority[3].append(neighbor)
            # 4. Neighbors that are not farther from the Prey and not closer to the Predator.
            elif neighbor_cost_to_prey==y and neighbor_cost_to_predator==x:
                next_node_priority[4].append(neighbor)
            # 5. Neighbors that are farther from the Predator.
            elif  neighbor_cost_to_predator>x:
                next_node_priority[5].append(neighbor)
            # 6. Neighbors that are not closer to the Predator.
            elif neighbor_cost_to_predator==x:
                next_node_priority[6].append(neighbor)
            # 7. Sit still and pray.
            else:
                next_node_priority[7].append(neighbor)
        
        next_node_list = []
        for k,v in next_node_priority.items():
            if len(v):
                next_node_list = v # Only returns the list with highest priority of 7
                break 
        next_node = None
        if not len(next_node_list):
            next_node = agent_pos # Agent waits in the same position 
        else:
            next_node = random.choice(next_node_list) # Agent 2 update comes here 
            
        #update the graph and Agent pos
        self.value = next_node
        self.graph.agent = next_node
        self.path.append(next_node)
        
    def get_position(self):
        return self.value
    
    def run(self, prey, predator, threshold = 50):
        '''
        Main module to run the Agent, prey and predator and check status 
        Make the Agent - Prey - Predator model run, 
        after each move(Agent, Prey and Predator) the program checks for the goals 
        1. Check if predator catches the Agent - returns FAILURE
        2. Check if the Agent catches the prey - returns SUCCESS
        '''
        count = 0 
        #DEBUG = True # Uncomment this to know the location of all in all iterations 
        while ((self.get_position()!=prey.get_position()) and (predator.get_position()!=self.get_position())):
            # 1. agent moves 
            count +=1 
            if threshold and count == threshold :
                return FAILURE, "Threshold reached "+ str(threshold)
            self.move(prey.get_position(), predator.get_position())
            if DEBUG:
                print("----------Moved("+str(count)+")------prey-------")
                print("Prey     : ", prey.get_position())
                print("Predator : ", predator.get_position())
                print("Agent    : ", self.get_position())
            if self.get_position()==prey.get_position():
                return SUCCESS, "Agent caught prey at "+ str(prey.get_position())
            # 2. prey moves 
            prey.move()    
            if DEBUG:
                print("----------Moved("+str(count)+")------prey-------")
                print("Prey     : ", prey.get_position())
                print("Predator : ", predator.get_position())
                print("Agent    : ", self.get_position())
            #print(self.get_position==prey.get_position())
            if self.get_position()==prey.get_position():
                return SUCCESS, "Agent caught prey at "+ str(prey.get_position())
            # 3. predator moves
            predator.move(self.value)
            if DEBUG:
                print("----------Moved("+str(count)+")-------pred------")
                print("Prey     : ", prey.get_position())
                print("Predator : ", predator.get_position())
                print("Agent    : ", self.get_position())
            if predator.get_position()==self.get_position():
                return FAILURE, "Predator killed Agent at "+ str(predator.get_position())
            

        if predator.get_position()==self.get_position():
            return FAILURE, "Predator killed Agent at "+ str(predator.get_position())
        elif self.get_position()==prey.get_position():
            return SUCCESS, "Agent caught prey at "+ str(prey.get_position())
        else:
            print("agent prey predator ")
            print(self.get_position(), "  ", prey.get_position(), "  ", predator.get_position())
            return  False, "total count "+str(count)

if __name__ == "__main__":
    DEBUG = False
    #i = test()
    
    graph = Graph_util(50)
    prey = Prey(graph,graph.node_count)
    predator = Predator(graph,graph.node_count)
    #########################
    agent1 = Agent1(graph, graph.node_count, prey, predator)
    DEBUG = False
    if DEBUG: 
        print("----------CurrPos--------------")
        print("Prey     : ", prey.get_position())
        print("Predator : ", predator.get_position())
        print("Agent    : ", agent1.get_position())
    #########################
    DEBUG = False
    verdict, msg = agent1.run(prey, predator)
    print("Success ? :", verdict)
    print(msg)
    DEBUG = True
    if DEBUG: 
        print("MSG :", msg)
        print("agent1 path("+str(len(agent1.path))+") : ", agent1.path)
        print("prey path : ", prey.path)
        #print("predator path : ", predator.path)
        graph.display()
