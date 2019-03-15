#author: jb
import math
import numpy as np

datafile1 = 'C:\Users\cyber\Desktop\University\precip_data2.txt'                    # parameters for the program

def summarize(datafile):                                                            #function: name of function(input parameters)
    data = np.genfromtxt(datafile, skiprows=1)                                          #read inputfile (skip header, used column ev....)
    precip = np.array([data[:, 2].min(), data[:, 2].max(), data[:, 2].mean()])          #create array out of calculations
    return precip                                                                       #return product of function

print('min             max            mean')                                        #just a description
print(summarize(datafile1))                                                         # print result of function for the parameter
