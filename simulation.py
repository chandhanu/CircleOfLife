import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from CreateEnvironment import Graph_util, Node
import networkx as nx
import random
import numpy as np
from PreyPredator import Prey, Predator
import collections
import heapq as hq
from Agent1 import Agent1
from TestAgent import TestAgent
from Agent2 import Agent2
from Agent3 import Agent3
from Agent4 import Agent4
from Agent5 import Agent5
from Agent6 import Agent6
from Agent7 import Agent7
from Agent7_noisy import Agent7_noisy
from Agent8 import Agent8
from Agent8_noisy import Agent8_noisy
import json
import sys 

def run_simulation(runs = 100, trials = 30, file = "agent1",variation= ""):
    d = {}
    for i in range(1, runs+1):
        d[i] = {}
        graph = Graph_util(50)
        failure_count = success_count = 0
        for j in range (1, trials+1):
            prey = Prey(graph,graph.node_count)
            predator = Predator(graph,graph.node_count)
            if file == "agent1":
                agent = Agent1(graph, graph.node_count, prey, predator)
            if file == "testagent":
                agent = TestAgent(graph, graph.node_count, prey, predator)
            elif file == "agent2":
                agent = Agent2(graph, graph.node_count, prey, predator)
            elif file == "agent3":
                agent = Agent3(graph, graph.node_count, prey, predator)
                agent.inititate_believes(graph, predator)
            elif file == "agent4":
                agent = Agent3(graph, graph.node_count, prey, predator)
                agent.inititate_believes(graph, predator)
            else:
                if file == "agent5":
                    agent = Agent5(graph, graph.node_count, prey, predator)
                    agent.inititate_believes(graph,predator)
                if file == "agent6":
                    agent = Agent6(graph, graph.node_count, prey, predator)
                    agent.inititate_believes(graph,predator)
                if file == "agent7":
                    agent = Agent7(graph, graph.node_count, prey, predator)
                    agent.inititate_believes(graph, prey,predator)
                if file == "agent7noisy":
                    agent = Agent7_noisy(graph, graph.node_count, prey, predator)
                    agent.inititate_believes(graph, prey,predator)
                if file == "agent8":
                    agent = Agent8(graph, graph.node_count, prey, predator)
                    agent.inititate_believes(graph, prey,predator)
                if file == "agent8noisy":
                    agent = Agent8_noisy(graph, graph.node_count, prey, predator)
                    agent.inititate_believes(graph, prey,predator)
                # Markup 
                agent.value = (prey.get_position()+2)%graph.node_count
                diff = predator.get_position() - agent.get_position()
                if diff < 0 :
                    diff = - diff 
                if diff < 5: #randomly place the characters 
                    r = random.choice ([20,21,23,25])
                    predator.value = (agent.get_position()+r)%graph.node_count
                    #prey.value = (agent.get_position())%graph.node_count
                # Markup ends '''
            verdict, msg = agent.run(prey, predator, threshold=100)
            print(msg)
            if verdict == False : 
                #print(msg)
                failure_count+=1
            else:
                success_count+=1
        success_rate = success_count/trials
        failure_rate = failure_count/trials

        d[i] = { j : (success_rate, failure_rate) }
        print("run: "+str(i)+", trials: "+str(j)+", success_rate : "+str(success_rate))
        with open(file+variation+".log", "a") as myfile:
                myfile.write("\n")
                myfile.write("run: "+str(i)+", trials: "+str(j)+", success_rate : "+str(success_rate))
    with open(file+variation+".json", "w") as outfile:
        json.dump(d, outfile)
        
if __name__ == "__main__":
    #run_simulation(file = "agent1")
    #run_simulation(file = "testagent")
    #run_simulation(file = "agent2")
    #run_simulation(file = "agent3", variation = "")
    #run_simulation(file = "agent4", variation = "")
    #run_simulation(file = "agent5", variation = "")
    #run_simulation(file = "agent6", variation = "")
    #run_simulation(file = "agent7", variation = "")
    run_simulation(file = "agent8", variation = "new")
    #run_simulation(file = "agent7noisy", variation = "")
    #run_simulation(file = "agent8noisy", variation = "")