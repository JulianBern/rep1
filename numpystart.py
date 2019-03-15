import numpy as np

#convert a list into a numpy array

mylist = [1, 2, 3, 4]
a_np = np.array(mylist)
print(a_np)

# 2 lines

b_np = np.array([mylist, mylist])
print(b_np)

#arange (number of elements), (number of lines, element per line)

c_np = np.arange(15).reshape(5,3)
print(c_np)

#define the name and read it
file = x.read()

