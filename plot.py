import json 
import random
import random
from functools import reduce
import sys 

import numpy as np
import matplotlib.pyplot as plt
data = {}
file = "agent5new.json"
file = "results/agent7noisy.json"
file = "results/agent5.json"
file = "agent6.json"


#file = sys.argv[1]
files = ["agent1.json", 
        "agent2.json",
        "agent3.json",
        "agent4.json",
        "agent5.json",
        "agent6.json",
        "agent7.json",
        "agent7noisy.json",
        "agent8.json",
        "agent8noisy.json",
        #"testagent.json"
]
d = {}

for file in files :

    with open("results/"+file) as json_file:
        data = json.load(json_file)
        
    add = 0 
    trials = 100
    print("FILE :", "results/"+file )
    try: 
        for i in range(1, 100+1):
            add = data[str(i)]["30"][0] + add 
            #print(data[str(i)]["30"][0])
        mean = (add/100)*100
        sd = []
        for i in range(1, 100+1):
                x = data[str(i)]["30"][0]*100
                sd.append((x-mean)**2)
                #print(mean, "-", x,"=", (x-mean)**2 )

        variations=sum(sd)/trials

        print("variations", variations)
    except:
        for i in range(1, 30+1):
            #print(data[str(i)]["100"][0])
            add = data[str(i)]["100"][0] + add 
        #print(data[str(i)]["30"][0])
        trials = 30
        mean = (add/30)*100
        sd = []
        for i in range(1, 30+1):
                x = data[str(i)]["100"][0]*100
                sd.append((x-mean)**2)
                #print(mean, "-", x,"=", (x-mean)**2 )

        variations=sum(sd)/trials

        print("variations", variations)
    
    d[file.replace(".json", "")] = (mean, variations, variations**(0.5))

    '''with open("results/"+"data"+".log", "a") as myfile:
        myfile.write("\n")
        myfile.write(file +" :\n")
        myfile.write("mean: "+str(mean)+", variance: "+str(variations)+", sd : "+str(variations**(0.5)))
with open(f"results/"+"data"+".json", "w") as outfile:
    json.dump(d, outfile)'''

with open("results/"+"data.json") as json_file:
        data = json.load(json_file)


x = []
y = []
success = False
if success:
    for k,v in data.items():
        '''ignore_list = ["testagent", "agent7noisy", "agent8noisy"]
        if k in ignore_list:
            continue'''
        if k in ["agent7", "agent8","agent8noisy", "agent7noisy"]:
            x.append(k)
            y.append(v[0]) # success rate
    
    fig = plt.figure(figsize = (8, 5))
    
    # creating the bar plot
    plt.bar(x, y, color ='green',
            width = 0.4)
    
    plt.ylabel("Success Rate")
    plt.xlabel("Agents")
    plt.title("Agents comparison")
    plt.show()
failure = True 
if failure: 
    for k,v in data.items():
        '''ignore_list = ["testagent", "agent7noisy", "agent8noisy"]
        if k in ignore_list:
            continue'''
        if k in ["agent7", "agent8","agent8noisy", "agent7noisy"]:
            x.append(k)
            s = 100-v[0]
            y.append(s) # Failure rate
    
    fig = plt.figure(figsize = (8, 5))
    
    # creating the bar plot
    plt.bar(x, y, color ='red',
            width = 0.4)
    
    plt.ylabel("Failure Rate")
    plt.xlabel("Agents")
    plt.title("Agents comparison")
    plt.show()
step = False 
if step: 
    for k,v in data.items():
        #ignore_list = ["testagent", "agent7noisy", "agent8noisy"]
        #if k in ignore_list:
        #    continue
        if k in ["agent7", "agent8","agent8noisy", "agent7noisy"]:
            x.append(k)
            s = v[3]
            y.append(s) # Failure rate
    
    fig = plt.figure(figsize = (8, 5))
    
    # creating the bar plot
    plt.bar(x, y, color ='grey',
            width = 0.4)
    
    plt.ylabel("Step size")
    plt.xlabel("Agents")
    plt.title("Agent comparison")
    plt.show()