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
import copy

DEBUG = False
SUCCESS = True 
FAILURE = False 

def max_d(d):
    key_max = max(d, key=d.get)
    max_val = d[key_max]
    l = []
    for k,v in d.items():
        if d[k] == max_val:
            l.append(k)
    return l 

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

def print_b(d, msg = "belief"):
    DEBUG = True
    if DEBUG:
        print("--------"+msg+" ------------")
        print("node ", "belief  ")
        sum = 0
        for k,v in d.items():
            print(k , "      ", v)
            sum += v
        print("-"*20)
        print("---Sum :",sum,"---missin: ",1-sum)
        print("-"*20)
    return sum
    
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
    max_belief(pred):  1
    Predator        :  13
    Agent           :  18
    Neighbor(18): [17, 19, 23, 18]

    --------agent_to_max_belief(pred)_by_cost ------------
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
    #return get_believes(graph, prey, predator, agent)
    return get_data(graph, prey, predator, agent) # CHAN : for agent 1 and 2 

class Agent5(Graph_util):
    def __init__(self, graph, node_count, prey, predator):
        self.graph = graph
        node_list = [i for i in range(node_count)]
        node_list.remove(graph.predator) # not produce an agent in the same position as predator
        self.value = random.choice([i for i in range(node_count)])
        self.node = graph.G.nodes[self.value]
        graph.agent = self.value
        self.path = []
        self.path.append(self.value)
        self.believes = {}
        self.node_count = graph.node_count
        self.belief_found_char_count = 0
    
    def move(self, prey_pos, believes,  FIRST_TIME = False):
        DEBUG = False
        
        agent_pos = self.value
        if not FIRST_TIME: # Agent does not know where predator is 
            max_belief_predator_list = max_d(believes)
            max_belief_predator = self.break_ties(max_belief_predator_list)
            agent_to_prey, predator_to_agent, all_shortest_path_cost = get_neighbor_data(self.graph, prey_pos, max_belief_predator, self.value) # Instead of Prey's position, the max(P(prey could be)) is given 
            agent_to_predator_cost = all_shortest_path_cost[agent_pos][max_belief_predator] # Instead of Prey's position, the max(P(prey could be)) is given
            
        else: # Agent knows where predator is at the first time 
            predator_pos = believes #believes actually contains predator pos in the first time 
            agent_to_prey, predator_to_agent, all_shortest_path_cost = get_neighbor_data(self.graph, prey_pos, predator_pos, self.value) # Instead of Prey's position, the max(P(prey could be)) is given 
            agent_to_predator_cost = all_shortest_path_cost[agent_pos][predator_pos] # Agent knows predator pos at the first 
            for k,v in self.believes.items(): # Beliefs we know where the predator at start
                if k == predator_pos: 
                    self.believes[k]=1
                else:
                    self.believes[k]=0

        agent_to_prey_cost = all_shortest_path_cost[agent_pos][prey_pos] 
        x = agent_to_predator_cost
        y = agent_to_prey_cost 
        agent_neighbors = list(self.graph.G.neighbors(agent_pos))
        next_node_list = []
        next_node_priority = { # Set priority to Agent's movements based on 7 conditions 
            1:[],
            2:[],
            3:[],
            4:[],
            5:[],
            6:[],
            7:[]
        }
        for neighbor in agent_neighbors:
            neighbor_cost_to_prey = agent_to_prey[neighbor][0]
            neighbor_cost_to_predator = predator_to_agent[neighbor][0]
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
                next_node_list = v # stores the highest priority list of moves 
                break 
        next_node = None
        if not len(next_node_list):
            next_node = agent_pos # Agent waits in the same position 
        else:
            max_v = -9999
            pos = random.choice(next_node_list)
            for i in next_node_list:
                z = all_shortest_path_cost[agent_pos][i]
                if z>max_v:
                    max_v = z
                    pos = i
            next_node = pos #returns the position that is longest from the max_belief_pred
            #next_node = random.choice(next_node_list) # CHAN - Update required if more than 1 node existing with same priority
        
        self.value = next_node
        self.graph.agent = next_node
        self.path.append(next_node)
        
    def get_position(self):
        return self.value
    
    def run(self, prey, predator, threshold = 200):
        '''
        Main module to run the Agent, prey and predator and check status 
        Make the Agent - Prey - Predator model run, 
        after each move(Agent, Prey and Predator) the program checks for the goals 
        1. Check if predator catches the Agent - returns FAILURE
        2. Check if the Agent catches the prey - returns SUCCESS
        '''
        count = 0 
        #DEBUG = True
        while ((self.get_position()!=prey.get_position()) and (predator.get_position()!=self.get_position())):
            if count == threshold:
                return  FAILURE, "total count "+str(count)
            self.status( prey, predator)
            if count == 0 : # First time the Agent knows the location of the predator
                # 1. Survey and 2. update beliefs
                self.update_believes(predator, survey=True, FIRST_TIME = True) # Survey + update believes 
                # 3. Move Agent - check if it catches the prey
                self.move(prey.get_position(), predator.get_position(), FIRST_TIME  = True) # At first Agent knows where the predator is 
            else:
                # 1. Survey and 2. update beliefs
                self.update_believes(predator, survey=True, FIRST_TIME = False) # Survey + update believes 
                # 3. Move Agent - check if it catches the prey
                self.move(prey.get_position(), self.believes)
            self.status( prey, predator)
            # 4. Update belief for prey/predator based on observation
            self.update_believes(predator, survey=False) # Only updates believes 
            # 5. Move Prey
            prey.move()    
            self.status( prey, predator) 
            # 6. Move Predator - check if it catches the agent
            predator.move_or_distract(self.value) #Predator can be distracted with the probablity of distribution of 40%  | with 60% it will pursue the target/agent
            self.status( prey, predator)
            # 7. Update belief for prey/predator based on their transition model
            self.update_believes(predator, survey=False) # Only updates believes
            # 8. Go to Step 1
            count+=1
            
        if predator.get_position()==self.get_position():
            return FAILURE, "Predator killed Agent at "+ str(predator.get_position())
        elif self.get_position()==prey.get_position():
            return SUCCESS, "Agent caught prey at "+ str(prey.get_position())
        else:
            print("agent prey predator ")
            print(self.get_position(), "  ", prey.get_position(), "  ", predator.get_position())
            return  FAILURE, "total count "+str(count)

    def inititate_believes(self, graph, predator):
        believes = {}
        # initialize
        for i in range(graph.node_count):
            if i == predator.get_position():
                believes[i] = 1 # Initially Agent knows wwhere the predator is 
            else:
                believes[i] = 0 
        self.believes = believes
        
    def get_believes(self, graph, prey, predator):
        return self.update_believes(self, graph, prey, predator)
    
    def break_ties(self, key_list):
        '''
        Breaking ties when in close proximity with Agent then at random
        '''
        node = None 
        all_shortest_path_cost =  self.graph.all_shortest_paths() #floyd-warshall 
        min_v = 99999
        for k in key_list:
            x = all_shortest_path_cost[self.get_position()][k]
            if x < min_v:
                min_v = x
        k_list = []
        for i in key_list:
            if all_shortest_path_cost[self.get_position()][i] == min_v:
                k_list.append(i)
        node = random.choice(k_list)       
        return node

    def survey(self, predator):
        believes = self.believes
        max_belief_predator_list = max_d(believes)
        max_belief_predator = self.break_ties(max_belief_predator_list)
        surveyed_node = max_belief_predator
        
        if surveyed_node == predator.get_position():
            self.belief_found_char_count+=1
            return SUCCESS, surveyed_node
        return FAILURE, surveyed_node
    
    def update_believes(self,predator, survey=False, FIRST_TIME = False):
        believes = self.believes
        if survey:
            is_predator_found,surveyed_node = self.survey(predator)

            if FIRST_TIME:
                surveyed_node = predator.get_position()
                is_predator_found = True
            if is_predator_found:
                for k,v in believes.items():
                    if k == surveyed_node:
                        believes[k] = 1
                    else:
                        believes[k] = 0
                p_predator_in_node_now = copy.deepcopy(believes)
                
            else:
                p_not_finding_predator_at_node = 1 - believes[surveyed_node]
                # Updating belief to find P(predator in ith node/ survyed and failed to find predator in random_node)
                for k,v in believes.items():
                    if k == surveyed_node:
                        believes[k] = 0
                    else:
                        believes[k] = (believes[k]/p_not_finding_predator_at_node)
                p_predator_in_node_now = copy.deepcopy(believes)
                # Updating belief to find P(Prey in ith node next)
                
        else:
            #Update belief
            k = self.get_position()
            k_belief = believes[k]
            p_predator_not_found_in_agent_node = 1-k_belief
            believes[self.get_position()] = 0
            for i,v in believes.items():
                if not (i == k):
                    believes[i] = believes[i]/p_predator_not_found_in_agent_node
            
            p_predator_in_node_now = copy.deepcopy(believes)
            max_belief_predator_list = max_d(believes)
            max_belief_predator = self.break_ties(max_belief_predator_list)
            max_belief = believes[max_belief_predator]
            believes[max_belief_predator] = 0 

            all_shortest_path_cost =  self.graph.all_shortest_paths() 
            all_pairs_shortest_path = dict(nx.all_pairs_shortest_path(self.graph.G))
            all_pairs_shortest_path[self.get_position()][max_belief_predator].remove(max_belief_predator)
            ratio_40 = 0.4*max_belief # Factoring in the distraction
            ratio_60 = 0.6*max_belief # Factoring in the target pursuation 
            
            shortest_path_from_agent_to_pred = all_pairs_shortest_path[self.get_position()][max_belief_predator]
            believes[max_belief_predator] = 0
            s= []
            ratio_60_count = 0
            ratio_40_count = 0
            for i in list(self.graph.G.neighbors(max_belief_predator)):
                x = all_shortest_path_cost[self.get_position()][i]               
                if len(shortest_path_from_agent_to_pred) == x:
                    ratio_60_count +=1
                    s.append(x) #Count of eligible node agent can move to pursuing the target - 60%
                else:
                    ratio_40_count+=1 # Count of nodes agent can move to distracted
                #print(i)
            # Calculation P(moving m to n) - prey and predator(0.6+0.4)
            
            p_predator_in_node_next = 0
            for i in list(self.graph.G.neighbors(max_belief_predator)):
                if i == max_belief_predator:
                    continue
                elif i in s: 
                    p_predator_in_node_next += p_predator_in_node_now[k]+ratio_60/len(s)
                else:
                    p_predator_in_node_next += p_predator_in_node_now[k]+ratio_40/ratio_40_count
                believes[i] = p_predator_in_node_next 

            # Distracted Predator acts like a prey (hence updating the beliefs)
        # Updating belief to find P(Prey in ith node next)
        
        for i,v in believes.items():
            p_predator_in_node_next = 0
            neighbors = list(self.graph.G.neighbors(i))
            neighbors.append(i)
            for k in neighbors:    
                p_predator_in_node_next += p_predator_in_node_now[k]*(1/len(neighbors))
            believes[i] = p_predator_in_node_next 

    def status(self, prey, predator):
        '''
        Module to check 
        1. Predator catched Agent - returns Failure 
        2. Agent Catches prey - returns Success 
        '''
        if self.get_position()==predator.get_position():
            #print("Predator caught agent at "+ str(predator.get_position()))
            #sys.exit(-1)
            return FAILURE, "Predator caught agent at "+ str(predator.get_position())
        if self.get_position()==prey.get_position():
            #print( "Agent caught prey at "+ str(prey.get_position()))
            #sys.exit(1)
            return SUCCESS, "Agent caught prey at "+ str(prey.get_position())


if __name__ == "__main__":
    DEBUG = False
    #graph = Graph_util(10)
    graph = Graph_util(50)
    prey = Prey(graph,graph.node_count)
    predator = Predator(graph,graph.node_count)
    #########################
    agent = Agent5(graph, graph.node_count, prey, predator)
    agent.inititate_believes(graph, predator)
    DEBUG = False
    if DEBUG: 
        print("----------CurrPos--------------")
        print("Prey     : ", prey.get_position())
        print("Predator : ", predator.get_position())
        print("Agent    : ", agent.get_position())
    #########################
    DEBUG = False
    verdict, msg = agent.run(prey, predator)
    print("Success ? :", verdict)
    print(msg)
    #print(agent.belief_found_char_count) # 1/5 times it find the predator

