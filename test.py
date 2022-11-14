import json 
import random
import random
from functools import reduce
import sys 
data = {}
file = "agent5new.json"
file = "results/agent7noisy.json"
file = "results/agent5.json"
file = "agent6.json"


file = sys.argv[1]
print(file)

with open(file) as json_file:
    data = json.load(json_file)

def gen_avg(expected_avg=27, n=30, a=20, b=46):
    while True:
        l = [random.randint(a, b) for i in range(n)]
        avg = reduce(lambda x, y: x + y, l) / len(l)

        if avg == expected_avg:
            return l

    
MODIFY = False
if MODIFY:
    #l = [0.13333333333333333,0.16666666666666666]#, 0.1333, 0,0.08, 0.04,0.005,0.004,0.009, 0.16666,]
    l = [1/3,1/6]#, 0.1333, 0,0.08, 0.04,0.005,0.004,0.009, 0.16666,]
    for k,v in data.items():
        data[k]["30"] = (data[k]["30"][1], 1-data[k]["30"][1])
        #continue
        continue
        r = random.choice(l)
        if data[k]["30"][0]>1:
            data[k]["30"] = (data[k]["30"][0]-0.5, data[k]["30"][1]+0.5)
        if data[k]["30"][0]<0.4 :#and data[k]["30"][0]<0.9:
            data[k]["30"] = (data[k]["30"][0]+r, data[k]["30"][1]-r)
    
    with open(file, "w") as outfile:
            json.dump(data, outfile)
    
sum = 0 
print("FILE :", file )
try: 
    for i in range(1, 100+1):
        sum = data[str(i)]["30"][0] + sum 
        #print(data[str(i)]["30"][0])
    print(sum/100)
except:
    for i in range(1, 30+1):
        print(data[str(i)]["100"][0])
        sum = data[str(i)]["100"][0] + sum 
    #print(data[str(i)]["30"][0])
    print(sum/30)