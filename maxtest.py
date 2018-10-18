import numpy as np


a = np.array([1,2,1, 3,3,1])  # Can be of any shape

print(a)
print (a.max(0)) #axis argument
# print (a.max(1))

indices = np.where(a == a.max())[0]
print(indices)

indices = np.where(a >= 1.5)[0]
print(indices)

indices = np.where(a == a.min())[0]

print(indices)