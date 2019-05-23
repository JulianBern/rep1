     # author: jb/ph
     import os
     import subprocess
     import math
     import numpy as np
     from xml.dom import minidom
     import glob

     # "C:/Program Files/AtmoSwing/bin/atmoswing-optimizer.exe" -s -l -r 1 -f data/parameters.xml --predictand-db=data/Precipitation-Daily-Catchment-RhiresD-Switzerland.nc --dir-predictors=data --log-level=2 --calibration-method=single --threads-nb=4 --station-id=1

     path = "C:/Users/cyber/Desktop/University/Project"                                                                 # define working directory
     os.chdir(path)

     folder = "data"                                                                                                    # define directory of predictor file
     xmlfile = "data/parameters.xml"                                                                                    # define xml file
     ncfile = "data/Precipitation-Daily-Catchment-RhiresD-Switzerland.nc"                                               # define predictant file

     results = "runs/41/results/*all_station_parameters.txt"                                                            # define directory of results

#cf = open(xmlfile, "w")
#file = cf.write()
#index = float(text[525:537])
#print(file)
#cf.close




#run_nb = "n"{n}
subprocess.call(["C:/Program Files/AtmoSwing/bin/atmoswing-optimizer.exe", "-s", "-l", "-r 2"--file-parameters=" +xmlfile,"--predictand-db=" +ncfile, "--dir-predictors=" +folder, "--log-level=2", "--calibration-method=single", "--threads-nb=4", "--station-id=1".format(run_nb)])




for result in glob.glob(results):                                                                                       #reading the CRPS index out of the results
     rf = open(result, "r")
     text = rf.read()
     index = float(text[525:537])
     print(index)
     rf.close





