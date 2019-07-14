import os
import subprocess
import math
import numpy as np
import glob
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from array import *
from io import StringIO

# define working directory
path = "C:/Users/cyber/Desktop/University/Project"
os.chdir(path)

# define directory of predictor file
folder = "data"

# define input nc file with model data
ncfile = "data/Precipitation-Daily-Catchment-RhiresD-Switzerland.nc"

# define AtmoSwing Optimizer xml file to be optimized
xmlfile = "data/output.xml"

################################################################################################################################################
# define number of values for parameter 1
n_sample = 20


#################################################################################################################################################
# define number of values for parameter 1
n_sample = 20
analogs_number_sample = np.linspace(10, 80, n_sample)

# create arrays with possible values for parameters (2 - 5)
x_min_sample = np.arange(-10, 11, 1.25)
x_points_nb_sample = np.arange(1, 26, 1)
y_min_sample = np.arange(35, 51.25, 1.25)
y_points_nb_sample = np.arange(1, 18, 1)

# define empty list to collect resulting CRPS values
C_opt = []

# define empty array to create combinations of values for AtmoSwing input
input_array = np.empty((0, 5))

###################################################################################################################################################
# create array of all possible combinations (rows) of input parameters
for sample in range(0, n_sample):# n_sample values
    for con1 in range(0, 17):# 17 values
        for con2 in range(0, 25):# 25 values
            for con3 in range(0, 13):# 13
                for con4 in range(0, 17):# 17
                    para = [str(analogs_number_sample[sample]), str(x_min_sample[con1]), str(x_points_nb_sample[con2]), str(y_min_sample[con3]), str(y_points_nb_sample[con4])]
                    print(para)
                    print(sample)
                    input_array = np.append(input_array, [para], axis=0)

#print(input_array)
print("input_array has " +str(int(input_array.size/5)) + " rows")
np.savetxt("input_array.txt", input_array, fmt="%s")

###################################################################################################################################################
# in case of 2 step run of the program, import the saved input_array file
input_array = np.loadtxt("input_array.txt")


# define array, that contains an equally balanced selection of the generated input_array (1´700´000+ vectors)

# number of vectors in new array
q = 5000

# create equally balanced indexes to select from input_array
t = np.linspace(0, int((input_array.size-1)/5), q)

# create empty array to append selection
in_run = np.empty((0, 5))

# create input_array_light (in_run)
for s in range(0, q):
    vector = [(input_array[int(t[s])])]

    in_run = np.append(in_run, vector, axis=0)

# save input_array_light
np.savetxt("input_array_light.txt", in_run, fmt="%s")


# define number of runs of AtmoSwing by counting the number of rows in the generated imput_array_light
runs = int(in_run.size/5)

##################################################################################################################################################
# change xml input file with parameters out of the input_array and run AtmoSwing in a loop
for run in range(1, runs+1):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    for elem in root.iter("analogs_number"):
            elem.text = str(in_run[run-1, 0])
    for elem in root.iter("x_min"):
            elem.text = str(in_run[run-1, 1])
    for elem in root.iter("x_points_nb"):
            elem.text = str(in_run[run-1, 2])
    for elem in root.iter("y_min"):
            elem.text = str(in_run[run-1, 3])
    for elem in root.iter("y_points_nb"):
            elem.text = str(in_run[run-1, 4])
# save the modified xml file
    tree.write("data/input.xml")
    input = ("data/input.xml")

    subprocess.call(["C:/Program Files/AtmoSwing/bin/atmoswing-optimizer.exe", "-s", "-l", "-r {}".format(run), "--file-parameters=" +input,"--predictand-db=" +ncfile, "--dir-predictors=" +folder, "--log-level=2", "--calibration-method=single", "--threads-nb=4", "--station-id=1"])

# define output file of AtmoSwing run
    outtext = "runs/{}/results/*all_station_parameters.txt".format(run)

    for result in glob.glob(outtext):
            rf = open(result, "r")
            text = rf.read()
# create a string to get the number out of it
            index = text

# create a list to count the place, where the calib is
            li = [text]
            nbli = int(li[0].find("Calib"))
            rf.close

# define the CRPS number out of the results
            CRPS = float(index[nbli+6:nbli+18])

# append current CRPS to list of results
            C_opt.append(CRPS)


            print(("run number: ")+str(run)+(" CRPS = ")+str(CRPS))


# search for minimum element index in list of CPRS and define the related parameters of input_array_light
P_opt = in_run[C_opt.index(min(C_opt))]

# save resulting CRPS list
with open("C_opt.txt", "w") as output:
    output.write(str(C_opt))

# print the resulting optimization of AtmoSwingOptimizer
print(("min CRPS is ") + str(min(C_opt)))
print(("optimized parameters are ")+str(P_opt))


# reopen the saved input_array_light
in_run = np.loadtxt("input_array_light.txt")


###################################################################################################################################################
# create lists out of input_array columns and plot
p1 = in_run[:, 0].tolist()
p2 = in_run[:, 1].tolist()
p3 = in_run[:, 2].tolist()
p4 = in_run[:, 3].tolist()
p5 = in_run[:, 4].tolist()



plt.scatter(p2, C_opt, s=10, c="r", marker="o", label="x_min")
plt.scatter(p5, C_opt, s=10, c="m", marker="o", label="y_points_nb")
plt.scatter(p3, C_opt, s=10, c="y", marker="o", label="x_points_nb")
plt.scatter(p4, C_opt, s=10, c="g", marker="o", label="y_min")
plt.scatter(p1, C_opt, s=10, c="b", marker="s", label="analogs_number")

plt.locator_params(axis="x", nbins=20)
#plt.locator_params(axis="y", nbins=4)
plt.ylabel("CRPS")
plt.legend(loc="upper left");
plt.show

