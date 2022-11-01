import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from CreateEnvironment import Graph_util, Node
import networkx as nx
import random
#random.seed(13) # Predator wins 
random.seed(1) # Agent wins 
import numpy as np
from PreyPredator import Prey, Predator
import collections
import heapq as hq
import json
import sys 

DEBUG = False
SUCCESS = True 
FAILURE = False 

def get_dict(source, target):
    neighbors = list(graph.G.neighbors(source))
    possible_nodes = neighbors[:]
    possible_nodes.append(source)
    all_pair_shortest_paths = dict(nx.all_pairs_shortest_path(graph.G))
    d = {}
    min = np.inf
    for i in possible_nodes:
        i_path = all_pair_shortest_paths[i][target]
        l=len(i_path)
        if l<min:
            min = l 
        if l in d:
            d[l].append(i_path)
        else:
            d[l] = []
            d[l].append(i_path)
    od = collections.OrderedDict(sorted(d.items()))
    
    return d, min 

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
    #print("--------new---------------------")
    agent_to_prey = {}
    agent_to_predator = {}
    agent_neighbors = list(graph.G.neighbors(agent))
    #agent_neighbors.append(agent) # curr pos of agent 
    #print("Neighbor("+str(agent)+"):", agent_neighbors)
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
    '''for neighbor in agent_neighbors:
        print("Neighbor : ", neighbor)
        #print(agent_to_prey[neighbor][0], agent_to_prey[neighbor][1])
        #print(agent_to_predator[neighbor][0], agent_to_predator[neighbor][1])
        print("prey:", agent_2_prey_predator[neighbor]["prey"][0],agent_2_prey_predator[neighbor]["prey"][1])
        print("predator:", agent_2_prey_predator[neighbor]["predator"][0],agent_2_prey_predator[neighbor]["predator"][1])
    '''
    return agent_to_prey_by_cost, agent_to_prey,  agent_to_predator_by_cost, predator_to_agent,all_shortest_path_cost,all_pairs_shortest_path
            
def get_neighbor_data(graph, prey, predator, agent):
    return get_data(graph, prey, predator, agent)


class Agent1(Graph_util):
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
        agent_to_prey_by_cost, agent_to_prey,  agent_to_predator_by_cost, predator_to_agent,all_shortest_path_cost,all_pairs_shortest_path = get_neighbor_data(self.graph, prey_pos, predator_pos, self.value)
        min_cost_to_prey = next(iter(agent_to_prey_by_cost.keys()))
        max_cost_to_predator = next(iter(agent_to_predator_by_cost.keys()))
        agent_pos = self.value
        agent_to_prey_cost = all_shortest_path_cost[agent_pos][prey_pos]
        agent_to_predator_cost = all_shortest_path_cost[agent_pos][predator_pos]
        x = agent_to_predator_cost
        y = agent_to_prey_cost
        agent_neighbors = list(self.graph.G.neighbors(agent_pos))
        next_node_list = []
        #print(">>>>>>>>>>>>>>>>>>> x:",x, "y:",y)
        next_node_priority = {
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
            #print("p:",k, "   n:",v)
            if len(v):
                next_node_list = v
                break 
        next_node = None
        if not len(next_node_list):
            next_node = agent_pos # Agent waits in the same position 
        else:
            next_node = next_node_list[0] # CHAN - Update required if more than 1 node existing with same priority
            
        #print(next_node_list, "chosen : ", next_node)
        #update the graph and Agent pos
        self.value = next_node
        self.graph.agent = next_node
        self.path.append(next_node)
        
    def get_position(self):
        return self.value
    
    def run(self, prey, predator, threshold = 50):
        #print("Agent", "Prey", "Predator" )
        count = 0 
        #DEBUG = True
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

def test():
    i = 0
    while (True):
        random.seed(i)
        print(i)
        graph = Graph_util(50)
        prey = Prey(graph,graph.node_count)
        predator = Predator(graph,graph.node_count)
        agent1 = Agent1(graph, graph.node_count, prey, predator)
        verdict, msg = agent1.run(prey, predator)
        print(msg, "pred_pos", predator.get_position())
        if verdict == False : 
            print(i)
            sys.exit(0)
        i+=1

if __name__ == "__main__":
    DEBUG = False
    #i = test()
    
    graph = Graph_util(50)
    prey = Prey(graph,graph.node_count)
    predator = Predator(graph,graph.node_count)
    #########################
    agent1 = Agent1(graph, graph.node_count, prey, predator)
    DEBUG = True
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
