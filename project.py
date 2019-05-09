#author: jb
import os
import math
import numpy as np
from xml.dom import minidom


analogs_number: lowerlimit="5" upperlimit="80"
x_min: lowerlimit="-10" upperlimit="10" iteration="1.25"
x_points_nb: lowerlimit="1" upperlimit="25" iteration="1"
y_min: lowerlimit="35" upperlimit="50" iteration="1.25"
y_points_nb: lowerlimit="1" upperlimit="17" iteration="1"


C:/Users/cyber/Desktop/University/Project/AtmoSwing/bin/atmoswing-optimizer.exe
    -s-l-r1-f = ('C:/Users/cyber/Desktop/University/Project/data/parameters.xml')
    predictand-db = "C:/Users/cyber/Desktop/University/Project/data/Precipitation-Daily-Catchment-RhiresD-Switzerland.nc"
    dir-predictors = "C:/Users/cyber/Desktop/University/Project/data"
    log-level=2
    calibration-method=single
    threads-nb=4
    station-id=1
