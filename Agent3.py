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
    max_belief(prey):  1
    Predator        :  13
    Agent           :  18
    Neighbor(18): [17, 19, 23, 18]

    --------agent_to_max_belief(prey)_by_cost ------------
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


class Agent3(Graph_util):
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
    
    def move(self, prey_pos, predator_pos, believes):
        DEBUG = False
        max_belief = max(believes, key=believes.get)
        if DEBUG: print(">>>>>>>max belief : ", max_belief)
        agent_to_prey, predator_to_agent, all_shortest_path_cost = get_neighbor_data(self.graph, max_belief, predator_pos, self.value) # Instead of Prey's position, the max(P(prey could be)) is given 
        agent_pos = self.value
        #agent_to_prey_cost = all_shortest_path_cost[agent_pos][prey_pos]
        agent_to_prey_cost = all_shortest_path_cost[agent_pos][max_belief] # Instead of Prey's position, the max(P(prey could be)) is given 
        agent_to_predator_cost = all_shortest_path_cost[agent_pos][predator_pos]
        x = agent_to_predator_cost
        y = agent_to_prey_cost # Changes come here, coz we dont the position of prey yet 
        agent_neighbors = list(self.graph.G.neighbors(agent_pos))
        next_node_list = []
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
            #print("p:",k, "   n:",v)
            if len(v):
                next_node_list = v
                break 
        next_node = None
        if not len(next_node_list):
            next_node = agent_pos # Agent waits in the same position 
        else:
            next_node = next_node_list[0] # CHAN - Update required if more than 1 node existing with same priority
        if DEBUG: print(">>>>>>>max belief : ", max_belief, "next_node : ", next_node)
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
            if count == 100:
                return  FAILURE, "total count "+str(count)
            DEBUG = False
            if DEBUG: 
                print("----------CurrPos--------------")
                print("Prey - ", prey.get_position(), end="   ")
                print("Predator - ", predator.get_position(), end="  ")
                print("Agent - ", self.get_position())
                print("---------------------------------")
                #print("----------Believes-------------")
                #print_b(agent.believes)
            self.status( prey, predator)
            # 1. Survey
            self.survey(prey, predator)
            #print_b(agent.believes, "before agent move")
            # 2. Update belief for prey/predator based on observation
            # 3. Move Agent - check if it catches the prey
            self.move(predator.get_position(), self.believes)
            self.status( prey, predator)
            if DEBUG: 
                print("----------CurrPos--------------")
                print("Prey - ", prey.get_position(), end="   ")
                print("Predator - ", predator.get_position(), end="  ")
                print("Agent - ", self.get_position())
                print("---------------------------------")
            # 4. Update belief for prey/predator based on observation
            self.survey(prey, predator)
            #print_b(agent.believes, "after agent move")
            # 5. Move Prey
            prey.move()    
            self.status( prey, predator)
            if DEBUG: 
                print("----------CurrPos--------------")
                print("Prey - ", prey.get_position(), end="   ")
                print("Predator - ", predator.get_position(), end="  ")
                print("Agent - ", self.get_position())
                print("---------------------------------")
            # 6. Move Predator - check if it catches the agent
            predator.move(self.value)
            self.status( prey, predator)
            if DEBUG: 
                print("----------CurrPos--------------")
                print("Prey - ", prey.get_position(), end="   ")
                print("Predator - ", predator.get_position(), end="  ")
                print("Agent - ", self.get_position())
                print("---------------------------------")
            # 7. Update belief for prey/predator based on their transition model
            #agent.survey(prey, predator)
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
            if i == self.get_position():
                believes[i] = 0
            else:
                believes[i] = 1/(graph.node_count-1) # Total node - 2 (agent pos + predator + pos)
        self.believes = believes
        #print_b(believes)
    
    def update_believes(self, graph, prey, predator):
        return

    def get_believes(self, graph, prey, predator):
        return self.update_believes(self, graph, prey, predator)

    def survey(self, prey, predator):
        DEBUG = False
        believes = self.believes
        random_survey_node = random.choice([i for i in range(self.node_count)])
        if DEBUG: print("Random : ", random_survey_node, "'s belief : ", believes[random_survey_node])
        if random_survey_node == prey.get_position():
            if DEBUG: print("=========Prey found==========")
            # Prey found 
            for k,v in believes.items():
                if k == random_survey_node:
                    believes[k] = 1
                else:
                    believes[k] = 0
            p_prey_in_node_now = copy.deepcopy(believes)
            if DEBUG: print_b(believes, "p_prey_in_node_now")
            for i,v in believes.items():
                #believes[i] = believes[i]/p_not_finding_prey_at_node
                #print("+++for - ",i)
                p_prey_in_node_next = 0
                #if i != self.get_position():
                neighbors = list(self.graph.G.neighbors(i))
                neighbors.append(i)
                for k in neighbors:
                    #p_prey_in_node_next = 0
                    #if k != random_survey_node:
                    p_prey_in_node_next += p_prey_in_node_now[k]*(1/len(neighbors))
                    
                    #print(k,":",p_prey_in_node_now[k], "*",1/len(neighbors) ,"=", p_prey_in_node_next) 
                believes[i] = p_prey_in_node_next 
            if DEBUG: print_b(believes, "p_prey_in_node_next")
        else:
            if DEBUG: print("========Prey Not found========")
            p_not_finding_prey_at_node = 1 - believes[random_survey_node]
            #P(prey in A/ failed to find prey in B)

            for k,v in believes.items():
                if k == random_survey_node:
                    believes[k] = 0
                    
                    '''elif k == self.get_position():
                        believes[k] = 0'''
                else:
                    believes[k] = (believes[k]/p_not_finding_prey_at_node)#+(believes[k]/p_not_finding_prey_at_node)/(self.node_count-2)
            '''sum = print_b(believes,"p_prey_in_node_now")
            miss = 1- sum

            if miss > 0 :
                for k,v in believes.items():
                    if k == random_survey_node:
                        believes[k] = 0
                    elif k == self.get_position():
                        believes[k] = 0
                    else:
                        believes[k]+=miss/(self.node_count-2) # Surveyed node + agent pos 
            '''
            if DEBUG : input()
            p_prey_in_node_now = copy.deepcopy(believes)
            #P(prey in A next)
            for i,v in believes.items():
                #believes[i] = believes[i]/p_not_finding_prey_at_node
                #print("+++for - ",i)
                p_prey_in_node_next = 0
                #if i != self.get_position():
                neighbors = list(self.graph.G.neighbors(i))
                neighbors.append(i)
                for k in neighbors:
                    #p_prey_in_node_next = 0
                    #if k != random_survey_node:
                    p_prey_in_node_next += p_prey_in_node_now[k]*(0.25)
                    
                    if DEBUG: print(k,":",p_prey_in_node_now[k], "*",0.25 ,"=", p_prey_in_node_next) 
                believes[i] = p_prey_in_node_next
                if DEBUG: print("+++P(",i,") = ",believes[i])
            if DEBUG: print_b(believes, "p_prey_in_node_next")
            #self.graph.display()
            if DEBUG : input()

            #Prey not found 

        return 

    def status(self, prey, predator):
        if self.get_position()==predator.get_position():
            #print("Predator caught agent at "+ str(predator.get_position()))
            #sys.exit(-1)
            return FAILURE, "Predator caught agent at "+ str(predator.get_position())
        if self.get_position()==prey.get_position():
            #print( "Agent caught prey at "+ str(prey.get_position()))
            #sys.exit(1)
            return SUCCESS, "Agent caught prey at "+ str(prey.get_position())
def test(): # update run() later
    i = 0
    
    graph = Graph_util(10)
    prey = Prey(graph,graph.node_count)
    predator = Predator(graph,graph.node_count)
    agent = Agent3(graph, graph.node_count, prey, predator)
    agent.inititate_believes(graph, predator)
    #agent.survey(prey, predator)
    #prey.value = agent.value # CHAN: For DEBUGGING
    #verdict, msg = agent3.run(prey, predator)
    while(True):
        DEBUG = True
        if i == 100:
            return FAILURE, "Agent failed after threshold count:"+str(i)
        if DEBUG: 
            print("----------CurrPos--------------")
            print("Prey - ", prey.get_position(), end="   ")
            print("Predator - ", predator.get_position(), end="  ")
            print("Agent - ", agent.get_position())
            print("---------------------------------")
            #print("----------Believes-------------")
            #print_b(agent.believes)
        agent.status( prey, predator)
        # 1. Survey
        agent.survey(prey, predator)
        # 2. Update belief for prey/predator based on observation
        # 3. Move Agent - check if it catches the prey
        agent.move(predator.get_position(), agent.believes)
        agent.status( prey, predator)
        if DEBUG: 
            print("----------CurrPos--------------")
            print("Prey - ", prey.get_position(), end="   ")
            print("Predator - ", predator.get_position(), end="  ")
            print("Agent - ", agent.get_position())
            
        # 4. Update belief for prey/predator based on observation
        agent.survey(prey, predator)
        # 5. Move Prey
        prey.move()    
        agent.status( prey, predator)
        if DEBUG: 
            print("----------CurrPos--------------")
            print("Prey - ", prey.get_position(), end="   ")
            print("Predator - ", predator.get_position(), end="  ")
            print("Agent - ", agent.get_position())
            
        # 6. Move Predator - check if it catches the agent
        predator.move(agent.value)
        agent.status( prey, predator)
        if DEBUG: 
            print("----------CurrPos--------------")
            print("Prey - ", prey.get_position(), end="   ")
            print("Predator - ", predator.get_position(), end="  ")
            print("Agent - ", agent.get_position())
            
        # 7. Update belief for prey/predator based on their transition model
        #agent.survey(prey, predator)
        # 8. Go to Step 1
        i+=1
        input()


    DEBUG = False
    if DEBUG: 
        print("----------FinalPos--------------")
        print("Prey     : ", prey.get_position())
        print("Predator : ", predator.get_position())
        print("Agent    : ", agent.get_position())


if __name__ == "__main__":
    DEBUG = False
    #verdict, msg = test()
    #print(msg)
    #exit(-1)
    #graph = Graph_util(10)
    graph = Graph_util(10)
    prey = Prey(graph,graph.node_count)
    predator = Predator(graph,graph.node_count)
    #########################
    agent = Agent3(graph, graph.node_count, prey, predator)
    agent.inititate_believes(graph, predator)
    DEBUG = True
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
    DEBUG = True
    if DEBUG: 
        print("MSG :", msg)
        print("agent3 path("+str(len(agent.path))+") : ", agent.path)
        print("prey path : ", prey.path)
        #graph.display()
        #A = nx.adjacency_matrix(graph.G)
        #print(A)

