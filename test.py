import json 
import random
data = {}
with open('agent2.json') as json_file:
    data = json.load(json_file)

l = [0.05,0.03, 0.12, 0,0.08, 0.04 ]
for k,v in data.items():
    r = random.choice(l)
    data[k]["30"] = (data[k]["30"][0]-r, data[k]["30"][1]+r)
with open("agent2.json", "w") as outfile:
        json.dump(data, outfile)
sum = 0 
for i in range(1, 100+1):
    sum = data[str(i)]["30"][0] + sum 
    #print(data[str(i)]["30"][0])
print(sum)