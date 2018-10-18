import json

with open('queenStatFile.txt', 'r') as f:
        datas = json.load(f)

#Use the new datastore datastructure
print (datas)