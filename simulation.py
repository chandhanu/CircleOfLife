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
from Agent1 import FAILURE, Agent1
from Agent2 import Agent2
import json
import sys 

def run_simulation(runs = 100, trials = 30, file = "agent1"):
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
            elif file == "agent2":
                agent = Agent2(graph, graph.node_count, prey, predator)
            verdict, msg = agent.run(prey, predator)
            if verdict == False : 
                #print(msg)
                failure_count+=1
            else:
                success_count+=1
        success_rate = success_count/trials
        failure_rate = failure_count/trials
        d[i] = { j : (success_rate, failure_rate) }
        print("run: "+str(i)+", trials: "+str(j)+", success_rate : "+str(success_rate))
        with open(file+".log", "a") as myfile:
                myfile.write("\n")
                myfile.write("run: "+str(i)+", trials: "+str(j)+", success_rate : "+str(success_rate))
    with open(file+".json", "w") as outfile:
        json.dump(d, outfile)
        
if __name__ == "__main__":
    #run_simulation(file = "agent1")
    run_simulation(file = "agent2")