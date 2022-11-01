import json 
data = {}
with open('agent2.json') as json_file:
    data = json.load(json_file)

sum = 0 
for i in range(1, 100+1):
    sum = data[str(i)]["30"][0] + sum 
    #print(data[str(i)]["30"][0])
print(sum)