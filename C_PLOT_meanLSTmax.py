import os
import h5py
import numpy as np
import geopandas as gpd
import gdal
from osgeo import gdal_array
from osgeo import osr
import ipykernel.pylab.backend_inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from operator import add
from operator import sub
import pandas as pd
import numpy as np # for calculating standard deviation and mean
import scipy.stats as sp
import matplotlib.transforms

# define working directory of georeference files from ESA
path = r"F:\Master thesis 2020\Python\WORK_SAfr\georef"
os.chdir(path)

# LAT_NAfr import and array generation
with h5py.File('hdf5_lsasaf_msg_lat_nafr_4bytesprecision', 'r') as hdf:
    ls1 = list(hdf.keys())
    print('List of dataset in this file: \n', ls1)
    lat_n = hdf.get('LAT')
    lat_n_data = np.array(lat_n)
    print('shape of dataset1: \n', lat_n_data.shape)
# LON_NAfr import and array generation
with h5py.File('hdf5_lsasaf_msg_lon_nafr_4bytesprecision', 'r') as hdf:
    ls2 = list(hdf.keys())
    print('List of dataset in this file: \n', ls2)
    lon_n = hdf.get('LON')
    lon_n_data = np.array(lon_n)
    print('shape of dataset1: \n', lon_n_data.shape)

# define study area of north africa SEVIRI region
lat_n = lat_n_data[1074:, 1819:1916] / 10000
lon_n = lon_n_data[1074:, 1819:1916] / 10000

# LAT_SAfr import and array generation
with h5py.File('hdf5_lsasaf_msg_lat_safr_4bytesprecision', 'r') as hdf:
    ls11 = list(hdf.keys())
    print('List of dataset in this file: \n', ls11)
    lat_s = hdf.get('LAT')
    lat_s_data = np.array(lat_s)
    print('shape of dataset1: \n', lat_s_data.shape)
# LON_SAfr import and array generation
with h5py.File('hdf5_lsasaf_msg_lon_safr_4bytesprecision', 'r') as hdf:
    ls12 = list(hdf.keys())
    print('List of dataset in this file: \n', ls12)
    lon_s = hdf.get('LON')
    lon_s_data = np.array(lon_s)
    print('shape of dataset1: \n', lon_s_data.shape)

# define study area of south africa SEVIRI region
lat_s = lat_s_data[0:32, 920:1017] / 10000
lon_s = lon_s_data[0:32, 920:1017] / 10000



################ REOPEN THE EXTRACTED NP ARRAYS ########################
#SOUTH AFRICA
path = (r"F:\results\max_arrays\SAfr")
os.chdir(path)
subs_jan1_max = np.load('subs_jan1_max.npy')
subs_jan2_max = np.load('subs_jan2_max.npy')
subs_jan3_max = np.load('subs_jan3_max.npy')
subs_feb1_max = np.load('subs_feb1_max.npy')
subs_feb2_max = np.load('subs_feb2_max.npy')
subs_feb3_max = np.load('subs_feb3_max.npy')
subs_mar1_max = np.load('subs_mar1_max.npy')
subs_mar2_max = np.load('subs_mar2_max.npy')
subs_mar3_max = np.load('subs_mar3_max.npy')
subs_apr1_max = np.load('subs_apr1_max.npy')
subs_apr2_max = np.load('subs_apr2_max.npy')
subs_apr3_max = np.load('subs_apr3_max.npy')
subs_may1_max = np.load('subs_may1_max.npy')
subs_may2_max = np.load('subs_may2_max.npy')
subs_may3_max = np.load('subs_may3_max.npy')
subs_jun1_max = np.load('subs_jun1_max.npy')
subs_jun2_max = np.load('subs_jun2_max.npy')
subs_jun3_max = np.load('subs_jun3_max.npy')
subs_jul1_max = np.load('subs_jul1_max.npy')
subs_jul2_max = np.load('subs_jul2_max.npy')
subs_jul3_max = np.load('subs_jul3_max.npy')
subs_aug1_max = np.load('subs_aug1_max.npy')
subs_aug2_max = np.load('subs_aug2_max.npy')
subs_aug3_max = np.load('subs_aug3_max.npy')
subs_sep1_max = np.load('subs_sep1_max.npy')
subs_sep2_max = np.load('subs_sep2_max.npy')
subs_sep3_max = np.load('subs_sep3_max.npy')
subs_oct1_max = np.load('subs_oct1_max.npy')
subs_oct2_max = np.load('subs_oct2_max.npy')
subs_oct3_max = np.load('subs_oct3_max.npy')
subs_nov1_max = np.load('subs_nov1_max.npy')
subs_nov2_max = np.load('subs_nov2_max.npy')
subs_nov3_max = np.load('subs_nov3_max.npy')
subs_dec1_max = np.load('subs_dec1_max.npy')
subs_dec2_max = np.load('subs_dec2_max.npy')
subs_dec3_max = np.load('subs_dec3_max.npy')

extention = ["3x3 pixel = 81 km²", "1 pixel = 9 km²", "1 pixel = 9 km²", "1 pixel = 9 km²", "1 pixel = 9 km²", "1 pixel = 9 km²", "1 pixel = 9 km²", "1 pixel = 9 km²", "1 pixel = 9 km²", "1 pixel = 9 km²"]
#### define extention of aois [aoi1_9x9], [aoi1_3x3], [aoi1_1x1]
data_list = [subs_jan1_max,
subs_jan2_max,
subs_jan3_max,
subs_feb1_max,
subs_feb2_max,
subs_feb3_max,
subs_mar1_max,
subs_mar2_max,
subs_mar3_max,
subs_apr1_max,
subs_apr2_max,
subs_apr3_max,
subs_may1_max,
subs_may2_max,
subs_may3_max,
subs_jun1_max,
subs_jun2_max,
subs_jun3_max,
subs_jul1_max,
subs_jul2_max,
subs_jul3_max,
subs_aug1_max,
subs_aug2_max,
subs_aug3_max,
subs_sep1_max,
subs_sep2_max,
subs_sep3_max,
subs_oct1_max,
subs_oct2_max,
subs_oct3_max,
subs_nov1_max,
subs_nov2_max,
subs_nov3_max,
subs_dec1_max,
subs_dec2_max,
subs_dec3_max]

centre_pixels_s = [
[23, 24, 27, 28],       # Nyeri
[15, 16, 28, 29],       # NARO MORU
[21, 22, 40, 41],       # KIRARI
[10, 11, 17, 18],       #Wiyumririri
[27, 28, 14, 15],       #Murungaru
[6, 7, 25, 26],       #Ol pejeta
[4, 5, 37, 38],       #Timau
[4, 5, 57, 58],      # Meru
[19, 20, 60, 61],      # Mwingi
[10, 11, 33, 34],   # Mt. Kenya Forest Gitinga
[9, 10, 39, 40],     #lengalilla
[18, 19, 42, 43],     # mt kenya forest Chuka
[19, 20, 20, 21],   # Aberdare forest East
[18, 19, 17, 18]   # Aberdare National Park
]


s_str_list_of_aoi = ['NYERI', 'Farming NARO MORU', 'Agrofor KIRARI', 'Wiyumririri', 'Murungaru', 'OlPejeta',
                     'Timau', 'Meru', 'Mwingi', 'Mt. Kenya Forest Gitinga (East)', 'lengalilla', 'mt kenya forest Chuka (West)', 'Aberdare forest East', 'Aberdare National Park2']


# AOIs SOUTH 12
for areas in range(12, 13):  # number of aois to plot and save data
    for subarea in range(0, 10): #number of pixels and extent of aoi, (centre, extent, followed by all 8 pixels sorrounding, a, b, c, d, e, f, g, h)
        centre = centre_pixels_s[areas]
        aoi_area = [centre[0] - 1, centre[1] + 1, centre[2] - 1, centre[3] + 1]
        sub_a = [centre[0] - 1, centre[0], centre[2] - 1, centre[2]]
        sub_b = [centre[0] - 1, centre[0], centre[2], centre[3]]
        sub_c = [centre[0] - 1, centre[0], centre[3], centre[3] + 1]
        sub_d = [centre[0], centre[1], centre[2] - 1, centre[2]]
        sub_e = [centre[0], centre[1], centre[2] + 1, centre[2] + 2]
        sub_f = [centre[0] + 1, centre[1] + 1, centre[2] - 1, centre[2]]
        sub_g = [centre[0] + 1, centre[1] + 1, centre[2], centre[3]]
        sub_h = [centre[0] + 1, centre[1] + 1, centre[2] + 1, centre[2] + 2]
        sub_areas = [aoi_area, centre, sub_a, sub_b, sub_c, sub_d, sub_e, sub_f, sub_g, sub_h]
        for ar in range(0, 36):  # number of points in time, calculate the spatial mean value of every 3d array
            aoi_arrays = data_list[ar]
            print(aoi_arrays.shape)
            aoi_pre = aoi_arrays[0:, sub_areas[subarea][0]:sub_areas[subarea][1],
                      sub_areas[subarea][2]:sub_areas[subarea][3]]

            aoi1 = aoi_pre[~np.isnan(aoi_pre)]

            mean = np.mean(aoi1)
            standard_deviation = np.std(aoi1)
            distance_from_mean = abs(aoi1 - mean)
            max_deviations = 3
            not_outlier = distance_from_mean < max_deviations * standard_deviation
            no_outliers = aoi1[not_outlier]

            print(no_outliers)

            aoi = no_outliers/100

            if ar == 0:
                box0 = aoi
                mean0 = np.mean(aoi)
                median0 = np.median(box0)
                stv0 = np.std(aoi)
            if ar == 1:
                box1 = aoi
                mean1 = np.mean(aoi)
                median1 = np.median(box1)
                stv1 = np.std(aoi)
            if ar == 2:
                box2 = aoi
                mean2 = np.mean(aoi)
                median2 = np.median(box2)
                stv2 = np.std(aoi)
            if ar == 3:
                box3 = aoi
                mean3 = np.mean(aoi)
                median3 = np.median(box3)
                stv3 = np.std(aoi)
            if ar == 4:
                box4 = aoi
                mean4 = np.mean(aoi)
                median4 = np.median(box4)
                stv4 = np.std(aoi)
            if ar == 5:
                box5 = aoi
                mean5 = np.mean(aoi)
                median5 = np.median(box5)
                stv5 = np.std(aoi)
            if ar == 6:
                box6 = aoi
                mean6 = np.mean(aoi)
                median6 = np.median(box6)
                stv6 = np.std(aoi)
            if ar == 7:
                box7 = aoi
                mean7 = np.mean(aoi)
                median7 = np.median(box7)
                stv7 = np.std(aoi)
            if ar == 8:
                box8 = aoi
                mean8 = np.mean(aoi)
                median8 = np.median(box8)
                stv8 = np.std(aoi)
            if ar == 9:
                box9 = aoi
                mean9 = np.mean(aoi)
                median9 = np.median(box9)
                stv9 = np.std(aoi)
            if ar == 10:
                box10 = aoi
                mean10 = np.mean(aoi)
                median10 = np.median(box10)
                stv10 = np.std(aoi)
            if ar == 11:
                box11 = aoi
                mean11 = np.mean(aoi)
                median11 = np.median(box11)
                stv11 = np.std(aoi)
            if ar == 12:
                box12 = aoi
                mean12 = np.mean(aoi)
                median12 = np.median(box12)
                stv12 = np.std(aoi)
            if ar == 13:
                box13 = aoi
                mean13 = np.mean(aoi)
                median13 = np.median(box13)
                stv13 = np.std(aoi)
            if ar == 14:
                box14 = aoi
                mean14 = np.mean(aoi)
                median14 = np.median(box14)
                stv14 = np.std(aoi)
            if ar == 15:
                box15 = aoi
                mean15 = np.mean(aoi)
                median15 = np.median(box15)
                stv15 = np.std(aoi)
            if ar == 16:
                box16 = aoi
                mean16 = np.mean(aoi)
                median16 = np.median(box16)
                stv16 = np.std(aoi)
            if ar == 17:
                box17 = aoi
                mean17 = np.mean(aoi)
                median17 = np.median(box17)
                stv17 = np.std(aoi)
            if ar == 18:
                box18 = aoi
                mean18 = np.mean(aoi)
                median18 = np.median(box18)
                stv18 = np.std(aoi)
            if ar == 19:
                box19 = aoi
                mean19 = np.mean(aoi)
                median19 = np.median(box19)
                stv19 = np.std(aoi)
            if ar == 20:
                box20 = aoi
                mean20 = np.mean(aoi)
                median20 = np.median(box20)
                stv20 = np.std(aoi)
            if ar == 21:
                box21 = aoi
                mean21 = np.mean(aoi)
                median21 = np.median(box21)
                stv21 = np.std(aoi)
            if ar == 22:
                box22 = aoi
                mean22 = np.mean(aoi)
                median22 = np.median(box22)
                stv22 = np.std(aoi)
            if ar == 23:
                box23 =aoi
                mean23 = np.mean(aoi)
                median23 = np.median(box23)
                stv23 = np.std(aoi)
            if ar == 24:
                box24 = aoi
                mean24 = np.mean(aoi)
                median24 = np.median(box24)
                stv24 = np.std(aoi)
            if ar == 25:
                box25 = aoi
                mean25 = np.mean(aoi)
                median25 = np.median(box25)
                stv25 = np.std(aoi)
            if ar == 26:
                box26 = aoi
                mean26 = np.mean(aoi)
                median26 = np.median(box26)
                stv26 = np.std(aoi)
            if ar == 27:
                box27 = aoi
                mean27 = np.mean(aoi)
                median27 = np.median(box27)
                stv27 = np.std(aoi)
            if ar == 28:
                box28 = aoi
                mean28 = np.mean(aoi)
                median28 = np.median(box28)
                stv28 = np.std(aoi)
            if ar == 29:
                box29 = aoi
                mean29 = np.mean(aoi)
                median29 = np.median(box29)
                stv29 = np.std(aoi)
            if ar == 30:
                box30 = aoi
                mean30 = np.mean(aoi)
                median30 = np.median(box30)
                stv30 = np.std(aoi)
            if ar == 31:
                box31 = aoi
                mean31 = np.mean(aoi)
                median31 = np.median(box31)
                stv31 = np.std(aoi)
            if ar == 32:
                box32 = aoi
                mean32 = np.mean(aoi)
                median32 = np.median(box32)
                stv32 = np.std(aoi)
            if ar == 33:
                box33 = aoi
                mean33 = np.mean(aoi)
                median33 = np.median(box33)
                stv33 = np.std(aoi)
            if ar == 34:
                box34 = aoi
                mean34 = np.mean(aoi)
                median34 = np.median(box34)
                stv34 = np.std(aoi)
            if ar == 35:
                box35 = aoi
                mean35 = np.mean(aoi)
                median35 = np.median(box35)
                stv35 = np.std(aoi)
            path = (r"F:\results\annual_plots")
            os.chdir(path)
        if subarea == 0:
            extent_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            extent_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                         mean10, mean11,
                         mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                         mean21, mean22, mean23,
                         mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                         mean33, mean34,
                         mean35]
            extent_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9, median10, median11,
                   median12, median13, median14, median15, median16, median17, median18, median19, median20, median21, median22, median23,
                   median24, median25, median26, median27, median28, median29, median30, median31, median32, median33, median34,
                   median35]
            extent = [box0,
                    box1,
                    box2,
                    box3,
                    box4,
                    box5,
                    box6,
                    box7,
                    box8,
                    box9,
                    box10,
                    box11,
                    box12,
                    box13,
                    box14,
                    box15,
                    box16,
                    box17,
                    box18,
                    box19,
                    box20,
                    box21,
                    box22,
                    box23,
                    box24,
                    box25,
                    box26,
                    box27,
                    box28,
                    box29,
                    box30,
                    box31,
                    box32,
                    box33,
                    box34,
                    box35]
        if subarea == 1:
            centrepix_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                        stv15,
                        stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                        stv29, stv30,
                        stv31, stv32, stv33, stv34, stv35]
            centrepix_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                         mean10, mean11,
                         mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                         mean21, mean22, mean23,
                         mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                         mean33, mean34,
                         mean35]
            centrepix_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9, median10, median11,
                   median12, median13, median14, median15, median16, median17, median18, median19, median20, median21, median22, median23,
                   median24, median25, median26, median27, median28, median29, median30, median31, median32, median33, median34,
                   median35]
            centrepix = [box0,
                    box1,
                    box2,
                    box3,
                    box4,
                    box5,
                    box6,
                    box7,
                    box8,
                    box9,
                    box10,
                    box11,
                    box12,
                    box13,
                    box14,
                    box15,
                    box16,
                    box17,
                    box18,
                    box19,
                    box20,
                    box21,
                    box22,
                    box23,
                    box24,
                    box25,
                    box26,
                    box27,
                    box28,
                    box29,
                    box30,
                    box31,
                    box32,
                    box33,
                    box34,
                    box35]
        if subarea == 2:
            a_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                        stv15,
                        stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                        stv29, stv30,
                        stv31, stv32, stv33, stv34, stv35]
            a_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                         mean10, mean11,
                         mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                         mean21, mean22, mean23,
                         mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                         mean33, mean34,
                         mean35]
            a_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9, median10, median11,
                   median12, median13, median14, median15, median16, median17, median18, median19, median20, median21, median22, median23,
                   median24, median25, median26, median27, median28, median29, median30, median31, median32, median33, median34,
                   median35]
            a = [box0,
                    box1,
                    box2,
                    box3,
                    box4,
                    box5,
                    box6,
                    box7,
                    box8,
                    box9,
                    box10,
                    box11,
                    box12,
                    box13,
                    box14,
                    box15,
                    box16,
                    box17,
                    box18,
                    box19,
                    box20,
                    box21,
                    box22,
                    box23,
                    box24,
                    box25,
                    box26,
                    box27,
                    box28,
                    box29,
                    box30,
                    box31,
                    box32,
                    box33,
                    box34,
                    box35]
        if subarea == 3:
            b_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                        stv15,
                        stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                        stv29, stv30,
                        stv31, stv32, stv33, stv34, stv35]
            b_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                         mean10, mean11,
                         mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                         mean21, mean22, mean23,
                         mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                         mean33, mean34,
                         mean35]
            b_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9, median10, median11,
                   median12, median13, median14, median15, median16, median17, median18, median19, median20, median21, median22, median23,
                   median24, median25, median26, median27, median28, median29, median30, median31, median32, median33, median34,
                   median35]
            b = [box0,
                    box1,
                    box2,
                    box3,
                    box4,
                    box5,
                    box6,
                    box7,
                    box8,
                    box9,
                    box10,
                    box11,
                    box12,
                    box13,
                    box14,
                    box15,
                    box16,
                    box17,
                    box18,
                    box19,
                    box20,
                    box21,
                    box22,
                    box23,
                    box24,
                    box25,
                    box26,
                    box27,
                    box28,
                    box29,
                    box30,
                    box31,
                    box32,
                    box33,
                    box34,
                    box35]
        if subarea == 4:
            c_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            c_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            c_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            c = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 5:
            d_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            d_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            d_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            d = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 6:
            e_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            e_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            e_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            e = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 7:
            f_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            f_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            f_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            f = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 8:
            g_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            g_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            g_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            g = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 9:
            h_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            h_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            h_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            h = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        array = aoi_pre[0, 0:, 0:]
        # array = Safr_max
        lat = lat_s[centre[0]:centre[1],
              centre[2]:centre[3]]
        lon = lon_s[centre[0]:centre[1],
              centre[2]:centre[3]]
        # define working directory
        path = r"F:\results\tif"
        os.chdir(path)
        # adjust the coordinates of SEVIRI(center of pixel) to GTIFF export (boarder of pixel area)
        xmin, ymin, xmax, ymax = [lon.min() - 0.0189, lat.min() - 0.0135944, lon.max() + 0.0189, lat.max() + 0.0135944]
        nrows, ncols = np.shape(array)
        xres = (xmax - xmin) / float(ncols)
        yres = (ymax - ymin) / float(nrows)
        geotransform = (xmin, xres, 0, ymax, 0, -yres)
        # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
        #         top left y, rotation (0 if North is up), n-s pixel resolution)
        # I don't know why rotation is in twice???
        # output_raster = gdal.GetDriverByName('GTiff').Create('No_meanMAX_feb2.tif', ncols, nrows, 1, gdal.GDT_Float32)  # Open the file
        output_raster = gdal.GetDriverByName('GTiff').Create(s_str_list_of_aoi[areas] + '.tif', ncols, nrows, 1,
                                                             gdal.GDT_Float32)  # Open the file
        output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
        srs = osr.SpatialReference()  # Establish its coordinate encoding
        srs.ImportFromEPSG(4326)  # This one specifies WGS84 lat long.
        # Anyone know how to specify the
        # IAU2000:49900 Mars encoding?
        output_raster.SetProjection(srs.ExportToWkt())  # Exports the coordinate system
        # to the file
        output_raster.GetRasterBand(1).WriteArray(array)  # Writes my array to the raster
        output_raster.FlushCache()
    if areas == 0:
            MAX_nyeri = np.stack((extent, centrepix, a, b, c, d, e, f, g,
                                               h))  # collect all pixel datasets, all data for boxplots
            MAX_nyeri_median = np.stack((extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median,
                                      f_median, g_median, h_median))  # collect all pixel datasets
            MAX_nyeri_deviation = np.stack((extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
            MAX_nyeri_mean = np.stack((extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean,
                                    h_mean))
    if areas == 1:
                                MAX_NAROMORU = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
                                MAX_NAROMORU_median = np.stack((extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median, g_median, h_median))  # collect all pixel datasets
                                MAX_NAROMORU_deviation = np.stack((extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
                                MAX_NAROMORU_mean = np.stack((extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean))
    if areas == 2:
                                MAX_KIRARI = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
                                MAX_KIRARI_median = np.stack((extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median, g_median, h_median)) # collect all pixel datasets
                                MAX_KIRARI_deviation = np.stack((extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
                                MAX_KIRARI_mean = np.stack((extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean))
    if areas == 3:
                                MAX_Wiyumririri = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
                                        MAX_Wiyumririri_median = np.stack((extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                                            g_median, h_median))  # collect all pixel datasets
                                        MAX_Wiyumririri_deviation = np.stack((extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
                                        MAX_Wiyumririri_mean = np.stack((extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean))
        if areas == 4:
            MAX_Murungaru = np.stack((extent, centrepix, a, b, c, d, e, f, g, h)) # collect all pixel datasets
            MAX_Murungaru_median = np.stack((extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                                                g_median, h_median)) # collect all pixel datasets
            MAX_Murungaru_deviation = np.stack((extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
            MAX_Murungaru_mean = np.stack((extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
        if areas == 5:
                MAX_OlPejeta = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
                MAX_OlPejeta_median = np.stack(
                    (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                     g_median, h_median))  # collect all pixel datasets
                MAX_OlPejeta_deviation = np.stack(
                    (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
                MAX_OlPejeta_mean = np.stack(
                    (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
        if areas == 6:
                MAX_Timau = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
                MAX_Timau_median = np.stack(
                    (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                     g_median, h_median))  # collect all pixel datasets
                MAX_Timau_deviation = np.stack(
                    (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
                MAX_Timau_mean = np.stack(
                    (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
        if areas == 7:
            MAX_Meru = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
            MAX_Meru_median = np.stack(
                (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                 g_median, h_median))  # collect all pixel datasets
            MAX_Meru_deviation = np.stack(
                (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
            MAX_Meru_mean = np.stack(
                (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
        if areas == 8:
            MAX_Mwingi = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
            MAX_Mwingi_median = np.stack(
                (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                 g_median, h_median))  # collect all pixel datasets
            MAX_Mwingi_deviation = np.stack(
                (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
            MAX_Mwingi_mean = np.stack(
                (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
        if areas == 9:
            MAX_Gitinga = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
            MAX_Gitinga_median = np.stack(
                (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                 g_median, h_median))  # collect all pixel datasets
            MAX_Gitinga_deviation = np.stack(
                (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
            MAX_Gitinga_mean = np.stack(
                (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
        if areas == 10:
            MAX_Lengalilla = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
            MAX_Lengalilla_median = np.stack(
                (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                 g_median, h_median))  # collect all pixel datasets
            MAX_Lengalilla_deviation = np.stack(
                (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
            MAX_Lengalilla_mean = np.stack(
                (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
        if areas == 11:
            MAX_Chuka = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
            MAX_Chuka_median = np.stack(
                (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                 g_median, h_median))  # collect all pixel datasets
            MAX_Chuka_deviation = np.stack(
                (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
            MAX_Chuka_mean = np.stack(
                (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
        if areas == 12:
            MAX_Aberdare_forest = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
            MAX_Aberdare_forest_median = np.stack(
                (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                 g_median, h_median))  # collect all pixel datasets
            MAX_Aberdare_forest_deviation = np.stack(
                (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
            MAX_Aberdare_forest_mean = np.stack(
                (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
        if areas == 13:
                MAX_Aberdare_National = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
                MAX_Aberdare_National_median = np.stack(
                    (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
                     g_median, h_median))  # collect all pixel datasets
                MAX_Aberdare_National_deviation = np.stack(
                    (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
                MAX_Aberdare_National_mean = np.stack(
                    (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)

############################################################
############### REOPEN THE NP ARRAYS ########################
#North AFRICA
###############
##############
#############


path = (r"F:\results\max_arrays\NAfr")
os.chdir(path)
subs_jan1_max = np.load('subs_jan1_max.npy')
subs_jan2_max = np.load('subs_jan2_max.npy')
subs_jan3_max = np.load('subs_jan3_max.npy')
subs_feb1_max = np.load('subs_feb1_max.npy')
subs_feb2_max = np.load('subs_feb2_max.npy')
subs_feb3_max = np.load('subs_feb3_max.npy')
subs_mar1_max = np.load('subs_mar1_max.npy')
subs_mar2_max = np.load('subs_mar2_max.npy')
subs_mar3_max = np.load('subs_mar3_max.npy')
subs_apr1_max = np.load('subs_apr1_max.npy')
subs_apr2_max = np.load('subs_apr2_max.npy')
subs_apr3_max = np.load('subs_apr3_max.npy')
subs_may1_max = np.load('subs_may1_max.npy')
subs_may2_max = np.load('subs_may2_max.npy')
subs_may3_max = np.load('subs_may3_max.npy')
subs_jun1_max = np.load('subs_jun1_max.npy')
subs_jun2_max = np.load('subs_jun2_max.npy')
subs_jun3_max = np.load('subs_jun3_max.npy')
subs_jul1_max = np.load('subs_jul1_max.npy')
subs_jul2_max = np.load('subs_jul2_max.npy')
subs_jul3_max = np.load('subs_jul3_max.npy')
subs_aug1_max = np.load('subs_aug1_max.npy')
subs_aug2_max = np.load('subs_aug2_max.npy')
subs_aug3_max = np.load('subs_aug3_max.npy')
subs_sep1_max = np.load('subs_sep1_max.npy')
subs_sep2_max = np.load('subs_sep2_max.npy')
subs_sep3_max = np.load('subs_sep3_max.npy')
subs_oct1_max = np.load('subs_oct1_max.npy')
subs_oct2_max = np.load('subs_oct2_max.npy')
subs_oct3_max = np.load('subs_oct3_max.npy')
subs_nov1_max = np.load('subs_nov1_max.npy')
subs_nov2_max = np.load('subs_nov2_max.npy')
subs_nov3_max = np.load('subs_nov3_max.npy')
subs_dec1_max = np.load('subs_dec1_max.npy')
subs_dec2_max = np.load('subs_dec2_max.npy')
subs_dec3_max = np.load('subs_dec3_max.npy')

#### define extention of aois [aoi1_9x9], [aoi1_3x3], [aoi1_1x1]
data_list = [subs_jan1_max,
subs_jan2_max,
subs_jan3_max,
subs_feb1_max,
subs_feb2_max,
subs_feb3_max,
subs_mar1_max,
subs_mar2_max,
subs_mar3_max,
subs_apr1_max,
subs_apr2_max,
subs_apr3_max,
subs_may1_max,
subs_may2_max,
subs_may3_max,
subs_jun1_max,
subs_jun2_max,
subs_jun3_max,
subs_jul1_max,
subs_jul2_max,
subs_jul3_max,
subs_aug1_max,
subs_aug2_max,
subs_aug3_max,
subs_sep1_max,
subs_sep2_max,
subs_sep3_max,
subs_oct1_max,
subs_oct2_max,
subs_oct3_max,
subs_nov1_max,
subs_nov2_max,
subs_nov3_max,
subs_dec1_max,
subs_dec2_max,
subs_dec3_max]
labelmonths = [" ", "JAN", " ", " ", "FEB", " ", " ", "MAR", " ", " ", "APR", " ", " ", "MAY", " ", " ", "JUN", " ", " ", "JUL", " ", " ", "AUG", " ", " ", "SEP", " "," ", "OCT", " ", " ", "NOV", " ", " ", "DEC", " "]

#define the centre pixels of each AOI for the NORTH part of the study area
centre_pixels_n = [[66, 67, 29, 30],  # Klimanjo
                   [72, 73, 28, 29],  # Ol Jogi
                   [69, 70, 21, 22],  # Sosian
                   [61, 62, 28, 29],  #Oldonyiro
                   [74, 75, 40, 41],  #Lewa
                   [45, 46, 20, 21],  #Maralal
                   [63, 64, 31, 32],  # Ol Lentille
                   [49, 50, 35, 36],  # Wamba
                   [14, 15, 51, 52],  #Logologo
                   [4, 5, 63, 64],  # Shura
                   [24, 25, 72, 73],  # Duma
                    [41, 42, 66, 67],  # Merti
                    [75, 76, 44, 45]]  # Nkando

n_str_list_of_aoi = ['Kimanjo',
                    'Ol Jogi',
                    'Sosian',
                     'Oldonyiro',
                     'Lewa', 'Maralal', 'Ol Lentille', 'WambaC', 'Logologo', 'Shura', 'Duma1', 'Merti1', '2Nkando']

for areas in range(0, 13):  # number of aois to plot and save data
    for subarea in range(0, 10): #number of pixels and extent of aoi, (centre, extent, followed by all 8 pixels sorrounding, a, b, c, d, e, f, g, h)
        centre = centre_pixels_n[areas]
        aoi_area = [centre[0] - 1, centre[1] + 1, centre[2] - 1, centre[3] + 1]
        sub_a = [centre[0] - 1, centre[0], centre[2] - 1, centre[2]]
        sub_b = [centre[0] - 1, centre[0], centre[2], centre[3]]
        sub_c = [centre[0] - 1, centre[0], centre[3], centre[3] + 1]
        sub_d = [centre[0], centre[1], centre[2] - 1, centre[2]]
        sub_e = [centre[0], centre[1], centre[2] + 1, centre[2] + 2]
        sub_f = [centre[0] + 1, centre[1] + 1, centre[2] - 1, centre[2]]
        sub_g = [centre[0] + 1, centre[1] + 1, centre[2], centre[3]]
        sub_h = [centre[0] + 1, centre[1] + 1, centre[2] + 1, centre[2] + 2]
        sub_areas = [aoi_area, centre, sub_a, sub_b, sub_c, sub_d, sub_e, sub_f, sub_g, sub_h]
        for ar in range(0, 36):  # number of points in time, calculate the spatial mean value of every 3d array
            aoi_arrays = data_list[ar]
            print(aoi_arrays.shape)
            aoi_pre = aoi_arrays[0:, sub_areas[subarea][0]:sub_areas[subarea][1],
                      sub_areas[subarea][2]:sub_areas[subarea][3]]

            aoi1 = aoi_pre[~np.isnan(aoi_pre)]

            mean = np.mean(aoi1)
            standard_deviation = np.std(aoi1)
            distance_from_mean = abs(aoi1 - mean)
            max_deviations = 3
            not_outlier = distance_from_mean < max_deviations * standard_deviation
            no_outliers = aoi1[not_outlier]

            print(no_outliers)

            aoi = no_outliers/100

            if ar == 0:
                box0 = aoi
                mean0 = np.mean(aoi)
                median0 = np.median(box0)
                stv0 = np.std(aoi)
            if ar == 1:
                box1 = aoi
                mean1 = np.mean(aoi)
                median1 = np.median(box1)
                stv1 = np.std(aoi)
            if ar == 2:
                box2 = aoi
                mean2 = np.mean(aoi)
                median2 = np.median(box2)
                stv2 = np.std(aoi)
            if ar == 3:
                box3 = aoi
                mean3 = np.mean(aoi)
                median3 = np.median(box3)
                stv3 = np.std(aoi)
            if ar == 4:
                box4 = aoi
                mean4 = np.mean(aoi)
                median4 = np.median(box4)
                stv4 = np.std(aoi)
            if ar == 5:
                box5 = aoi
                mean5 = np.mean(aoi)
                median5 = np.median(box5)
                stv5 = np.std(aoi)
            if ar == 6:
                box6 = aoi
                mean6 = np.mean(aoi)
                median6 = np.median(box6)
                stv6 = np.std(aoi)
            if ar == 7:
                box7 = aoi
                mean7 = np.mean(aoi)
                median7 = np.median(box7)
                stv7 = np.std(aoi)
            if ar == 8:
                box8 = aoi
                mean8 = np.mean(aoi)
                median8 = np.median(box8)
                stv8 = np.std(aoi)
            if ar == 9:
                box9 = aoi
                mean9 = np.mean(aoi)
                median9 = np.median(box9)
                stv9 = np.std(aoi)
            if ar == 10:
                box10 = aoi
                mean10 = np.mean(aoi)
                median10 = np.median(box10)
                stv10 = np.std(aoi)
            if ar == 11:
                box11 = aoi
                mean11 = np.mean(aoi)
                median11 = np.median(box11)
                stv11 = np.std(aoi)
            if ar == 12:
                box12 = aoi
                mean12 = np.mean(aoi)
                median12 = np.median(box12)
                stv12 = np.std(aoi)
            if ar == 13:
                box13 = aoi
                mean13 = np.mean(aoi)
                median13 = np.median(box13)
                stv13 = np.std(aoi)
            if ar == 14:
                box14 = aoi
                mean14 = np.mean(aoi)
                median14 = np.median(box14)
                stv14 = np.std(aoi)
            if ar == 15:
                box15 = aoi
                mean15 = np.mean(aoi)
                median15 = np.median(box15)
                stv15 = np.std(aoi)
            if ar == 16:
                box16 = aoi
                mean16 = np.mean(aoi)
                median16 = np.median(box16)
                stv16 = np.std(aoi)
            if ar == 17:
                box17 = aoi
                mean17 = np.mean(aoi)
                median17 = np.median(box17)
                stv17 = np.std(aoi)
            if ar == 18:
                box18 = aoi
                mean18 = np.mean(aoi)
                median18 = np.median(box18)
                stv18 = np.std(aoi)
            if ar == 19:
                box19 = aoi
                mean19 = np.mean(aoi)
                median19 = np.median(box19)
                stv19 = np.std(aoi)
            if ar == 20:
                box20 = aoi
                mean20 = np.mean(aoi)
                median20 = np.median(box20)
                stv20 = np.std(aoi)
            if ar == 21:
                box21 = aoi
                mean21 = np.mean(aoi)
                median21 = np.median(box21)
                stv21 = np.std(aoi)
            if ar == 22:
                box22 = aoi
                mean22 = np.mean(aoi)
                median22 = np.median(box22)
                stv22 = np.std(aoi)
            if ar == 23:
                box23 =aoi
                mean23 = np.mean(aoi)
                median23 = np.median(box23)
                stv23 = np.std(aoi)
            if ar == 24:
                box24 = aoi
                mean24 = np.mean(aoi)
                median24 = np.median(box24)
                stv24 = np.std(aoi)
            if ar == 25:
                box25 = aoi
                mean25 = np.mean(aoi)
                median25 = np.median(box25)
                stv25 = np.std(aoi)
            if ar == 26:
                box26 = aoi
                mean26 = np.mean(aoi)
                median26 = np.median(box26)
                stv26 = np.std(aoi)
            if ar == 27:
                box27 = aoi
                mean27 = np.mean(aoi)
                median27 = np.median(box27)
                stv27 = np.std(aoi)
            if ar == 28:
                box28 = aoi
                mean28 = np.mean(aoi)
                median28 = np.median(box28)
                stv28 = np.std(aoi)
            if ar == 29:
                box29 = aoi
                mean29 = np.mean(aoi)
                median29 = np.median(box29)
                stv29 = np.std(aoi)
            if ar == 30:
                box30 = aoi
                mean30 = np.mean(aoi)
                median30 = np.median(box30)
                stv30 = np.std(aoi)
            if ar == 31:
                box31 = aoi
                mean31 = np.mean(aoi)
                median31 = np.median(box31)
                stv31 = np.std(aoi)
            if ar == 32:
                box32 = aoi
                mean32 = np.mean(aoi)
                median32 = np.median(box32)
                stv32 = np.std(aoi)
            if ar == 33:
                box33 = aoi
                mean33 = np.mean(aoi)
                median33 = np.median(box33)
                stv33 = np.std(aoi)
            if ar == 34:
                box34 = aoi
                mean34 = np.mean(aoi)
                median34 = np.median(box34)
                stv34 = np.std(aoi)
            if ar == 35:
                box35 = aoi
                mean35 = np.mean(aoi)
                median35 = np.median(box35)
                stv35 = np.std(aoi)

        path = (r"F:\results\annual_plots")
        os.chdir(path)
        if subarea == 0:
            extent_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            extent_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                         mean10, mean11,
                         mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                         mean21, mean22, mean23,
                         mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                         mean33, mean34,
                         mean35]
            extent_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9, median10, median11,
                   median12, median13, median14, median15, median16, median17, median18, median19, median20, median21, median22, median23,
                   median24, median25, median26, median27, median28, median29, median30, median31, median32, median33, median34,
                   median35]
            extent = [box0,
                    box1,
                    box2,
                    box3,
                    box4,
                    box5,
                    box6,
                    box7,
                    box8,
                    box9,
                    box10,
                    box11,
                    box12,
                    box13,
                    box14,
                    box15,
                    box16,
                    box17,
                    box18,
                    box19,
                    box20,
                    box21,
                    box22,
                    box23,
                    box24,
                    box25,
                    box26,
                    box27,
                    box28,
                    box29,
                    box30,
                    box31,
                    box32,
                    box33,
                    box34,
                    box35]
        if subarea == 1:
            centrepix_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                        stv15,
                        stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                        stv29, stv30,
                        stv31, stv32, stv33, stv34, stv35]
            centrepix_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                         mean10, mean11,
                         mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                         mean21, mean22, mean23,
                         mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                         mean33, mean34,
                         mean35]
            centrepix_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9, median10, median11,
                   median12, median13, median14, median15, median16, median17, median18, median19, median20, median21, median22, median23,
                   median24, median25, median26, median27, median28, median29, median30, median31, median32, median33, median34,
                   median35]
            centrepix = [box0,
                    box1,
                    box2,
                    box3,
                    box4,
                    box5,
                    box6,
                    box7,
                    box8,
                    box9,
                    box10,
                    box11,
                    box12,
                    box13,
                    box14,
                    box15,
                    box16,
                    box17,
                    box18,
                    box19,
                    box20,
                    box21,
                    box22,
                    box23,
                    box24,
                    box25,
                    box26,
                    box27,
                    box28,
                    box29,
                    box30,
                    box31,
                    box32,
                    box33,
                    box34,
                    box35]
        if subarea == 2:
            a_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                        stv15,
                        stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                        stv29, stv30,
                        stv31, stv32, stv33, stv34, stv35]
            a_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                         mean10, mean11,
                         mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                         mean21, mean22, mean23,
                         mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                         mean33, mean34,
                         mean35]
            a_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9, median10, median11,
                   median12, median13, median14, median15, median16, median17, median18, median19, median20, median21, median22, median23,
                   median24, median25, median26, median27, median28, median29, median30, median31, median32, median33, median34,
                   median35]
            a = [box0,
                    box1,
                    box2,
                    box3,
                    box4,
                    box5,
                    box6,
                    box7,
                    box8,
                    box9,
                    box10,
                    box11,
                    box12,
                    box13,
                    box14,
                    box15,
                    box16,
                    box17,
                    box18,
                    box19,
                    box20,
                    box21,
                    box22,
                    box23,
                    box24,
                    box25,
                    box26,
                    box27,
                    box28,
                    box29,
                    box30,
                    box31,
                    box32,
                    box33,
                    box34,
                    box35]
        if subarea == 3:
            b_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                        stv15,
                        stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                        stv29, stv30,
                        stv31, stv32, stv33, stv34, stv35]
            b_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                         mean10, mean11,
                         mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                         mean21, mean22, mean23,
                         mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                         mean33, mean34,
                         mean35]
            b_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9, median10, median11,
                   median12, median13, median14, median15, median16, median17, median18, median19, median20, median21, median22, median23,
                   median24, median25, median26, median27, median28, median29, median30, median31, median32, median33, median34,
                   median35]
            b = [box0,
                    box1,
                    box2,
                    box3,
                    box4,
                    box5,
                    box6,
                    box7,
                    box8,
                    box9,
                    box10,
                    box11,
                    box12,
                    box13,
                    box14,
                    box15,
                    box16,
                    box17,
                    box18,
                    box19,
                    box20,
                    box21,
                    box22,
                    box23,
                    box24,
                    box25,
                    box26,
                    box27,
                    box28,
                    box29,
                    box30,
                    box31,
                    box32,
                    box33,
                    box34,
                    box35]
        if subarea == 4:
            c_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            c_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            c_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            c = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 5:
            d_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            d_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            d_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            d = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 6:
            e_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            e_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            e_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            e = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 7:
            f_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            f_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            f_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            f = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 8:
            g_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            g_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            g_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            g = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        if subarea == 9:
            h_std = [stv0, stv1, stv2, stv3, stv4, stv5, stv6, stv7, stv8, stv9, stv10, stv11, stv12, stv13, stv14,
                         stv15,
                         stv16, stv17, stv18, stv19, stv20, stv21, stv22, stv23, stv24, stv25, stv26, stv27, stv28,
                         stv29, stv30,
                         stv31, stv32, stv33, stv34, stv35]
            h_mean = [mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9,
                          mean10, mean11,
                          mean12, mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20,
                          mean21, mean22, mean23,
                          mean24, mean25, mean26, mean27, mean28, mean29, mean30, mean31, mean32,
                          mean33, mean34,
                          mean35]
            h_median = [median0, median1, median2, median3, median4, median5, median6, median7, median8, median9,
                            median10, median11,
                            median12, median13, median14, median15, median16, median17, median18, median19, median20,
                            median21, median22, median23,
                            median24, median25, median26, median27, median28, median29, median30, median31, median32,
                            median33, median34,
                            median35]
            h = [box0,
                     box1,
                     box2,
                     box3,
                     box4,
                     box5,
                     box6,
                     box7,
                     box8,
                     box9,
                     box10,
                     box11,
                     box12,
                     box13,
                     box14,
                     box15,
                     box16,
                     box17,
                     box18,
                     box19,
                     box20,
                     box21,
                     box22,
                     box23,
                     box24,
                     box25,
                     box26,
                     box27,
                     box28,
                     box29,
                     box30,
                     box31,
                     box32,
                     box33,
                     box34,
                     box35]
        #array = aoi_pre[0, 0:, 0:]
        # array = Safr_max

        #lat = lat_n[centre[0]:centre[1],
        #      centre[2]:centre[3]]
        #lon = lon_n[centre[0]:centre[1],
        #      centre[2]:centre[3]]
        # define working directory
        #path = r"F:\results\tif"
        #os.chdir(path)
        # adjust the coordinates of SEVIRI(center of pixel) to GTIFF export (boarder of pixel area)
        #xmin, ymin, xmax, ymax = [lon.min() - 0.0189, lat.min() - 0.0135944, lon.max() + 0.0189, lat.max() + 0.0135944]
        #nrows, ncols = np.shape(array)
        #xres = (xmax - xmin) / float(ncols)
        #yres = (ymax - ymin) / float(nrows)
        #geotransform = (xmin, xres, 0, ymax, 0, -yres)
        # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
        #         top left y, rotation (0 if North is up), n-s pixel resolution)
        # I don't know why rotation is in twice???
        # output_raster = gdal.GetDriverByName('GTiff').Create('No_meanMAX_feb2.tif', ncols, nrows, 1, gdal.GDT_Float32)  # Open the file
        #output_raster = gdal.GetDriverByName('GTiff').Create(n_str_list_of_aoi[areas] + '.tif', ncols, nrows, 1,
        #                                                     gdal.GDT_Float32)  # Open the file
        #output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
        #srs = osr.SpatialReference()  # Establish its coordinate encoding
        #srs.ImportFromEPSG(4326)  # This one specifies WGS84 lat long.
        # Anyone know how to specify the
        # IAU2000:49900 Mars encoding?
        #output_raster.SetProjection(srs.ExportToWkt())  # Exports the coordinate system
        # to the file
        #output_raster.GetRasterBand(1).WriteArray(array)  # Writes my array to the raster
        #output_raster.FlushCache()


    if areas == 0:
        #MAX_Kimanjo = np.stack((extent, centrepix, a, b, c, d, e, f, g,h))  # collect all pixel datasets, all data for boxplots
        MAX_Kimanjo_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median,
             f_median, g_median, h_median))  # collect all pixel datasets
        MAX_Kimanjo_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Kimanjo_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean,
             h_mean))
    if areas == 1:
        #MAX_oljogi = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_oljogi_median = np.stack((extent_median, centrepix_median, a_median, b_median, c_median, d_median,
                                            e_median, f_median, g_median, h_median))  # collect all pixel datasets
        MAX_oljogi_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_oljogi_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean))
    if areas == 2:
        #MAX_Sosian = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Sosian_median = np.stack((extent_median, centrepix_median, a_median, b_median, c_median, d_median,
                                          e_median, f_median, g_median, h_median))  # collect all pixel datasets
        MAX_Sosian_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Sosian_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean))
    if areas == 3:
        #MAX_Oldonyiro = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Oldonyiro_median = np.stack((extent_median, centrepix_median, a_median, b_median, c_median, d_median,
                                            e_median, f_median, g_median, h_median))  # collect all pixel datasets
        MAX_Oldonyiro_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Oldonyiro_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean))
    if areas == 4:
        #MAX_Lewa = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Lewa_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
             g_median, h_median))  # collect all pixel datasets
        MAX_Lewa_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Lewa_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean))
    if areas == 5:
        #MAX_Maralal = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Maralal_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
             g_median, h_median))  # collect all pixel datasets
        MAX_Maralal_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Maralal_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
    if areas == 6:
        #MAX_OlLentille = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_OlLentille_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
             g_median, h_median))  # collect all pixel datasets
        MAX_OlLentille_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_OlLentille_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
    if areas == 7:
        #MAX_Wamba = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Wamba_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
             g_median, h_median))  # collect all pixel datasets
        MAX_Wamba_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Wamba_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
    if areas == 8:
        #MAX_Logologo = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Logologo_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
             g_median, h_median))  # collect all pixel datasets
        MAX_Logologo_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Logologo_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
    if areas == 9:
        #MAX_Shura = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Shura_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
             g_median, h_median))  # collect all pixel datasets
        MAX_Shura_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Shura_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
    if areas == 10:
        #MAX_Duma = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Duma_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
             g_median, h_median))  # collect all pixel datasets
        MAX_Duma_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Duma_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
    if areas == 11:
        #MAX_Merti = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Merti_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
             g_median, h_median))  # collect all pixel datasets
        MAX_Merti_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Merti_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)
    if areas == 12:
        #MAX_Nkando = np.stack((extent, centrepix, a, b, c, d, e, f, g, h))  # collect all pixel datasets
        MAX_Nkando_median = np.stack(
            (extent_median, centrepix_median, a_median, b_median, c_median, d_median, e_median, f_median,
             g_median, h_median))  # collect all pixel datasets
        MAX_Nkando_deviation = np.stack(
            (extent_std, centrepix_std, a_std, b_std, c_std, d_std, e_std, f_std, g_std, h_std))
        MAX_Nkando_mean = np.stack(
            (extent_mean, centrepix_mean, a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean, h_mean), axis=-0)





# save arrays of all aoi____________
#path = (r"C:\Users\icx\Desktop\MA_Q3\arraysAOIs")
#os.chdir(path)
#for sample in range(0, len(all_arrays_Feb_diu)):
#    np.save(all_arrays_Jun_dev_NAMES[sample], all_arrays_Jun_dev[sample])



### build lists of data to plot in loop (check if all arrays are here)
all_arrays_MAX_mean_NAMES = ["MAX_Merti_mean", "MAX_Duma_mean", "MAX_Shura_mean", "MAX_Logologo_mean", "MAX_Meru_mean", "MAX_Mwingi_mean", "MAX_Wamba_mean", "MAX_OlLentille_mean", "MAX_Oldonyiro_mean", "MAX_Nkando_mean",
                  "MAX_Lewa_mean", "MAX_oljogi_mean", "MAX_Chuka_mean", "MAX_nyeri_mean", "MAX_Sosian_mean", "MAX_OlPejeta_mean", "MAX_KIRARI_mean", "MAX_Kimanjo_mean",
                  "MAX_NAROMORU_mean", "MAX_Maralal_mean", "MAX_Wiyumririri_mean", "MAX_Murungaru_mean", "MAX_Timau_mean", "MAX_Gitinga_mean", "MAX_Aberdare_forest_mean", "MAX_Aberdare_National_mean",
                  "MAX_Lengalilla_mean"]
all_arrays_MAX_dev_NAMES = ["MAX_Merti_deviation", "MAX_Duma_deviation", "MAX_Shura_deviation", "MAX_Logologo_deviation", "MAX_Meru_deviation", "MAX_Mwingi_deviation", "MAX_Wamba_deviation",
                      "MAX_OlLentille_deviation", "MAX_Oldonyiro_deviation", "MAX_Nkando_deviation",
                  "MAX_Lewa_deviation", "MAX_oljogi_deviation", "MAX_Chuka_deviation", "MAX_nyeri_deviation", "MAX_Sosian_deviation", "MAX_OlPejeta_deviation", "MAX_KIRARI_deviation", "MAX_Kimanjo_deviation",
                  "MAX_NAROMORU_deviation", "MAX_Maralal_deviation", "MAX_Wiyumririri_deviation", "MAX_Murungaru_deviation", "MAX_Timau_deviation", "MAX_Gitinga_deviation", "MAX_Aberdare_forest_deviation", "MAX_Aberdare_National_deviation",
                  "MAX_Lengalilla_deviation"]
all_arrays_Feb_diu_NAMES = ["diurnal_Merti_mean", "diurnal_Duma_mean", "diurnal_Shura_mean", "diurnal_Logologo_mean", "diurnal_Meru_mean", "diurnal_Mwingi_mean", "diurnal_Wamba_mean", "diurnal_OlLentille_mean", "diurnal_Oldonyiro_mean", "diurnal_Nkando_mean",
                  "diurnal_Lewa_mean", "diurnal_oljogi_mean", "diurnal_Chuka_mean", "diurnal_nyeri_mean", "diurnal_Sosian_mean", "diurnal_OlPejeta_mean", "diurnal_KIRARI_mean", "diurnal_Klimanjo_mean",
                  "diurnal_NAROMORU_mean", "diurnal_Maralal_mean", "diurnal_Wiyumririri_mean", "diurnal_Murungaru_mean", "diurnal_Timau_mean", "diurnal_Gitinga_mean", "diurnal_Aberdare_forest_mean", "diurnal_Aberdare_National_mean",
                  "diurnal_Lengalilla_mean"]
all_arrays_Feb_dev_NAMES = ["diurnal_Merti_deviation", "diurnal_Duma_deviation", "diurnal_Shura_deviation", "diurnal_Logologo_deviation", "diurnal_Meru_deviation", "diurnal_Mwingi_deviation", "diurnal_Wamba_deviation", "diurnal_OlLentille_deviation", "diurnal_Oldonyiro_deviation", "diurnal_Nkando_deviation",
                  "diurnal_Lewa_deviation", "diurnal_oljogi_deviation", "diurnal_Chuka_deviation", "diurnal_nyeri_deviation", "diurnal_Sosian_deviation", "diurnal_OlPejeta_deviation", "diurnal_KIRARI_deviation", "diurnal_Klimanjo_deviation",
                  "diurnal_NAROMORU_deviation", "diurnal_Maralal_deviation", "diurnal_Wiyumririri_deviation", "diurnal_Murungaru_deviation", "diurnal_Timau_deviation", "diurnal_Gitinga_deviation", "diurnal_Aberdare_forest_deviation", "diurnal_Aberdare_National_deviation",
                  "diurnal_Lengalilla_deviation"]
all_arrays_Jun_diu_NAMES = ["Jdiurnal_Merti_mean", "Jdiurnal_Duma_mean", "Jdiurnal_Shura_mean", "Jdiurnal_Logologo_mean", "Jdiurnal_Meru_mean", "Jdiurnal_Mwingi_mean", "Jdiurnal_Wamba_mean", "Jdiurnal_OlLentille_mean", "Jdiurnal_Oldonyiro_mean", "Jdiurnal_Nkando_mean",
                  "Jdiurnal_Lewa_mean", "Jdiurnal_oljogi_mean", "Jdiurnal_Chuka_mean", "Jdiurnal_nyeri_mean", "Jdiurnal_Sosian_mean", "Jdiurnal_OlPejeta_mean", "Jdiurnal_KIRARI_mean", "Jdiurnal_Klimanjo_mean",
                  "Jdiurnal_NAROMORU_mean", "Jdiurnal_Maralal_mean", "Jdiurnal_Wiyumririri_mean", "Jdiurnal_Murungaru_mean", "Jdiurnal_Timau_mean", "Jdiurnal_Gitinga_mean", "Jdiurnal_Aberdare_forest_mean", "Jdiurnal_Aberdare_National_mean",
                  "Jdiurnal_Lengalilla_mean"]
all_arrays_Jun_dev_NAMES = ["Jdiurnal_Merti_deviation", "Jdiurnal_Duma_deviation", "Jdiurnal_Shura_deviation", "Jdiurnal_Logologo_deviation", "Jdiurnal_Meru_deviation", "Jdiurnal_Mwingi_deviation", "Jdiurnal_Wamba_deviation", "Jdiurnal_OlLentille_deviation", "Jdiurnal_Oldonyiro_deviation", "Jdiurnal_Nkando_deviation",
                  "Jdiurnal_Lewa_deviation", "Jdiurnal_oljogi_deviation", "Jdiurnal_Chuka_deviation", "Jdiurnal_nyeri_deviation", "Jdiurnal_Sosian_deviation", "Jdiurnal_OlPejeta_deviation", "Jdiurnal_KIRARI_deviation", "Jdiurnal_Klimanjo_deviation",
                  "Jdiurnal_NAROMORU_deviation", "Jdiurnal_Maralal_deviation", "Jdiurnal_Wiyumririri_deviation", "Jdiurnal_Murungaru_deviation", "Jdiurnal_Timau_deviation", "Jdiurnal_Gitinga_deviation", "Jdiurnal_Aberdare_forest_deviation", "Jdiurnal_Aberdare_National_deviation",
                  "Jdiurnal_Lengalilla_deviation"]



### OPEN ### RESULTS ###############################################################################################
# load results
path = (r"C:\Users\icx\Desktop\MA_Q3\arraysAOIs")
os.chdir(path)

a = 24
b = 26
print("AOI1 = "+all_arrays_MAX_mean_NAMES[a])
print("AOI2 = "+all_arrays_MAX_mean_NAMES[b])

aoi1_mannual = np.load(all_arrays_MAX_mean_NAMES[a] + '.npy')# ANNUAL MAX LST VAR
aoi2_mannual = np.load(all_arrays_MAX_mean_NAMES[b] + '.npy')# ANNUAL MAX LST VAR
aoi1_dev_annual = np.load(all_arrays_MAX_dev_NAMES[a] + '.npy')# ANNUAL MAX LST VAR
aoi2_dev_annual = np.load(all_arrays_MAX_dev_NAMES[b] + '.npy')# ANNUAL MAX LST VAR
aoi1_feb = np.load(all_arrays_Feb_diu_NAMES[a] + '.npy')# Feb mean LST VAR
aoi1_feb_dev = np.load(all_arrays_Feb_dev_NAMES[a] + '.npy')# Feb dev LST VAR
aoi2_feb = np.load(all_arrays_Feb_diu_NAMES[b] + '.npy')# Feb mean LST VAR
aoi2_feb_dev = np.load(all_arrays_Feb_dev_NAMES[b] + '.npy')# Feb mean LST VAR
aoi1_jun = np.load(all_arrays_Jun_diu_NAMES[a] + '.npy')# Jun mean LST VAR
aoi1_jun_dev = np.load(all_arrays_Jun_dev_NAMES[a] + '.npy')# Jun dev LST VAR
aoi2_jun = np.load(all_arrays_Jun_diu_NAMES[b] + '.npy')# Jun mean LST VAR
aoi2_jun_dev = np.load(all_arrays_Jun_dev_NAMES[b] + '.npy')# Jun mean LST VAR

##################################################################################################################################
#KEY NUMBERS FOR MA
AOI1 = aoi1_mannual[0]
AOI1_dev = aoi1_dev_annual[0]
AOI2 = aoi2_mannual[0]
AOI2_dev = aoi2_dev_annual[0]

### ANNUAL MAX ###
#stats
#mean Difference annual
meanLST_AOI1 = np.nanmean(AOI1)
maxLST_AOI1 = np.nanmax(AOI1)
minLST_AOI1 = np.nanmin(AOI1)
meanLST_AOI2 = np.nanmean(AOI2)
maxLST_AOI2 = np.nanmax(AOI2)
minLST_AOI2 = np.nanmin(AOI2)
DIF_mean_annual_LST = np.nanmean(AOI1-AOI2)
MAXdif_annual_LST = np.nanmin(AOI1-AOI2)
MINdif_annual_LST = np.nanmax(AOI1-AOI2)
range_max_AOI1 = np.nanmax(AOI1) - np.nanmin(AOI1)
range_max_AOI2 = np.nanmax(AOI2) - np.nanmin(AOI2)
AOI1_dev_mean = np.nanmean(AOI1_dev)
AOI2_dev_mean = np.nanmean(AOI2_dev)

# ANNUAL ###
# aoi1
print("LST MAX ANNUAL:")
print(all_arrays_MAX_mean_NAMES[a]+"aoi1 MEAN = " + str(meanLST_AOI1)+ ", mean SD = " + str(AOI1_dev_mean)+", "+ "range = " + str(range_max_AOI1))
print("aoi1 MAX = " + str(maxLST_AOI1)+", "+ "SD = " + str(AOI1_dev[np.argmax(AOI1)])+", "+ "month = " + str(np.argmax(AOI1)*1/3+1))
print("aoi1 MIN = " + str(minLST_AOI1)+", "+ "SD = " + str(AOI1_dev[np.argmin(AOI1)])+", "+ "month = " + str(np.argmin(AOI1)*1/3+1))
print("-------")
# aoi2
print(all_arrays_MAX_mean_NAMES[b]+"aoi2 MEAN = " + str(meanLST_AOI2)+ ", mean SD = " + str(AOI2_dev_mean)+", "+ "range = " + str(range_max_AOI2))
print("aoi2 MAX = " + str(maxLST_AOI2)+", "+ "SD = " + str(AOI2_dev[np.argmax(AOI2)])+", "+ "month = " + str(np.argmax(AOI2)*1/3+1))
print("aoi2 MIN = " + str(minLST_AOI2)+", "+ "SD = " + str(AOI2_dev[np.argmin(AOI2)])+", "+ "month = " + str(np.argmin(AOI2)*1/3+1))
print("-------")
# compare them
print("mean Difference = " + str(DIF_mean_annual_LST))
print("smallest Difference = " + str(MINdif_annual_LST)+", "+ "month = " + str(np.argmax(AOI1-AOI2)*1/3+1))
print("greatest Difference = " + str(MAXdif_annual_LST)+", "+ "month = " + str(np.argmin(AOI1-AOI2)*1/3+1)+ " "+ all_arrays_MAX_mean_NAMES[a]+ " "+str(AOI1[np.argmin(AOI1-AOI2)])+" SD "+str(AOI1_dev[np.argmin(AOI1-AOI2)])
      + " "+ all_arrays_MAX_mean_NAMES[b]+ " "+str(AOI2[np.argmin(AOI1-AOI2)])+" SD "+str(AOI2_dev[np.argmin(AOI1-AOI2)]))
print("Difference between Maxima = " + str(maxLST_AOI2-maxLST_AOI1))
########################################
# feb
AOI1 = aoi1_feb[0]
AOI1 = np.concatenate((AOI1[84:], AOI1[:84]), axis=0, out=None)
AOI1_dev = aoi1_feb_dev[0]
AOI1_dev = np.concatenate((AOI1_dev[84:], AOI1_dev[:84]), axis=0, out=None)
AOI2 = aoi2_feb[0]
AOI2 = np.concatenate((AOI2[84:], AOI2[:84]), axis=0, out=None)
AOI2_dev = aoi2_feb_dev[0]
AOI2_dev = np.concatenate((AOI2_dev[84:], AOI2_dev[:84]), axis=0, out=None)


meanLST_AOI1 = np.nanmean(AOI1)
maxLST_AOI1 = np.nanmax(AOI1)
minLST_AOI1 = np.nanmin(AOI1)
meanLST_AOI2 = np.nanmean(AOI2)
maxLST_AOI2 = np.nanmax(AOI2)
minLST_AOI2 = np.nanmin(AOI2)
DIF_mean_annual_LST = np.nanmean(AOI1-AOI2)
MAXdif_annual_LST = np.nanmin(AOI1-AOI2)
MINdif_annual_LST = np.nanmax(AOI1-AOI2)
range_max_AOI1 = np.nanmax(AOI1) - np.nanmin(AOI1)
range_max_AOI2 = np.nanmax(AOI2) - np.nanmin(AOI2)
AOI1_dev_mean = np.nanmean(AOI1_dev)
AOI2_dev_mean = np.nanmean(AOI2_dev)

print("LST MEAN DIURNAL FEB:")
print(all_arrays_MAX_mean_NAMES[a]+" aoi1 MEAN = " + str(meanLST_AOI1)+ ", mean SD = " + str(AOI1_dev_mean)+", "+ "range = " + str(range_max_AOI1))
print("aoi1 MAX = " + str(maxLST_AOI1)+", "+ "SD = " + str(AOI1_dev[np.argmax(AOI1)])+", "+ "time = " + str(np.argmax(AOI1)*0.25))
print("aoi1 MIN = " + str(minLST_AOI1)+", "+ "SD = " + str(AOI1_dev[np.argmin(AOI1)])+", "+ "time = " + str(np.argmin(AOI1)*0.25))
print("-------")
# aoi2
print(all_arrays_MAX_mean_NAMES[b]+" aoi2 MEAN = " + str(meanLST_AOI2)+ ", mean SD = " + str(AOI2_dev_mean)+", "+ "range = " + str(range_max_AOI2))
print("aoi2 MAX = " + str(maxLST_AOI2)+", "+ "SD = " + str(AOI2_dev[np.argmax(AOI2)])+", "+ "local time = " + str(np.argmax(AOI2)*0.25))
print("aoi2 MIN = " + str(minLST_AOI2)+", "+ "SD = " + str(AOI2_dev[np.argmin(AOI2)])+", "+ "local time = " + str(np.argmin(AOI2)*0.25))
print("-------")
# compare them
print("mean Difference = " + str(DIF_mean_annual_LST))
print("smallest Difference = " + str(MINdif_annual_LST)+", "+ "local time = " + str(np.argmax(AOI1-AOI2)*0.25))
print("greatest Difference = " + str(MAXdif_annual_LST)+", "+ "local time = " + str(np.argmin(AOI1-AOI2)*0.25)+ " "+ all_arrays_MAX_mean_NAMES[a]+ " "+str(AOI1[np.argmin(AOI1-AOI2)])+" SD "+str(AOI1_dev[np.argmin(AOI1-AOI2)])
      + " "+ all_arrays_MAX_mean_NAMES[b]+ " "+str(AOI2[np.argmin(AOI1-AOI2)])+" SD "+str(AOI2_dev[np.argmin(AOI1-AOI2)]))
print("Difference between Maxima = " + str(maxLST_AOI2-maxLST_AOI1))
##########################################
# jun
AOI1 = aoi1_jun[0]
AOI1 = np.concatenate((AOI1[84:], AOI1[:84]), axis=0, out=None)
AOI1_dev = aoi1_jun_dev[0]
AOI1_dev = np.concatenate((AOI1_dev[84:], AOI1_dev[:84]), axis=0, out=None)
AOI2 = aoi2_jun[0]
AOI2 = np.concatenate((AOI2[84:], AOI2[:84]), axis=0, out=None)
AOI2_dev = aoi2_jun_dev[0]
AOI2_dev = np.concatenate((AOI2_dev[84:], AOI2_dev[:84]), axis=0, out=None)

meanLST_AOI1 = np.nanmean(AOI1)
maxLST_AOI1 = np.nanmax(AOI1)
minLST_AOI1 = np.nanmin(AOI1)
meanLST_AOI2 = np.nanmean(AOI2)
maxLST_AOI2 = np.nanmax(AOI2)
minLST_AOI2 = np.nanmin(AOI2)
DIF_mean_annual_LST = np.nanmean(AOI1-AOI2)
MAXdif_annual_LST = np.nanmin(AOI1-AOI2)
MINdif_annual_LST = np.nanmax(AOI1-AOI2)
range_max_AOI1 = np.nanmax(AOI1) - np.nanmin(AOI1)
range_max_AOI2 = np.nanmax(AOI2) - np.nanmin(AOI2)
AOI1_dev_mean = np.nanmean(AOI1_dev)
AOI2_dev_mean = np.nanmean(AOI2_dev)

# ANNUAL ###
# aoi1
print("LST MEAN DIURNAL JUN:")
print(all_arrays_MAX_mean_NAMES[a]+" aoi1 MEAN = " + str(meanLST_AOI1)+ ", mean SD = " + str(AOI1_dev_mean)+", "+ "range = " + str(range_max_AOI1))
print("aoi1 MAX = " + str(maxLST_AOI1)+", "+ "SD = " + str(AOI1_dev[np.argmax(AOI1)])+", "+ "time = " + str(np.argmax(AOI1)*0.25))
print("aoi1 MIN = " + str(minLST_AOI1)+", "+ "SD = " + str(AOI1_dev[np.argmin(AOI1)])+", "+ "time = " + str(np.argmin(AOI1)*0.25))
print("-------")
# aoi2
print(all_arrays_MAX_mean_NAMES[b]+" aoi2 MEAN = " + str(meanLST_AOI2)+ ", mean SD = " + str(AOI2_dev_mean)+", "+ "range = " + str(range_max_AOI2))
print("aoi2 MAX = " + str(maxLST_AOI2)+", "+ "SD = " + str(AOI2_dev[np.argmax(AOI2)])+", "+ "local time = " + str(np.argmax(AOI2)*0.25))
print("aoi2 MIN = " + str(minLST_AOI2)+", "+ "SD = " + str(AOI2_dev[np.argmin(AOI2)])+", "+ "local time = " + str(np.argmin(AOI2)*0.25))
print("-------")
# compare them
print("mean Difference = " + str(DIF_mean_annual_LST))
print("smallest Difference = " + str(MINdif_annual_LST)+", "+ "local time = " + str(np.argmax(AOI1-AOI2)*0.25))
print("greatest Difference = " + str(MAXdif_annual_LST)+", "+ "local time = " + str(np.argmin(AOI1-AOI2)*0.25)+ " "+ all_arrays_MAX_mean_NAMES[a]+ " "+str(AOI1[np.argmin(AOI1-AOI2)])+" SD "+str(AOI1_dev[np.argmin(AOI1-AOI2)])
      + " "+ all_arrays_MAX_mean_NAMES[b]+ " "+str(AOI2[np.argmin(AOI1-AOI2)])+" SD "+str(AOI2_dev[np.argmin(AOI1-AOI2)]))
print("Difference between Maxima = " + str(maxLST_AOI2-maxLST_AOI1))


































### build lists of data to plot in loop (check if all arrays are here)
all_arrays_MAX_mean = [MAX_Merti_mean, MAX_Duma_mean, MAX_Shura_mean, MAX_Logologo_mean, MAX_Meru_mean, MAX_Mwingi_mean, MAX_Wamba_mean, MAX_OlLentille_mean, MAX_Oldonyiro_mean, MAX_Nkando_mean,
                  MAX_Lewa_mean, MAX_oljogi_mean, MAX_Chuka_mean, MAX_nyeri_mean, MAX_Sosian_mean, MAX_OlPejeta_mean, MAX_KIRARI_mean, MAX_Kimanjo_mean,
                  MAX_NAROMORU_mean, MAX_Maralal_mean, MAX_Wiyumririri_mean, MAX_Murungaru_mean, MAX_Timau_mean, MAX_Gitinga_mean, MAX_Aberdare_forest_mean, MAX_Aberdare_National_mean,
                  MAX_Lengalilla_mean]
all_arrays_MAX_dev = [MAX_Merti_deviation, MAX_Duma_deviation, MAX_Shura_deviation, MAX_Logologo_deviation, MAX_Meru_deviation, MAX_Mwingi_deviation, MAX_Wamba_deviation,
                      MAX_OlLentille_deviation, MAX_Oldonyiro_deviation, MAX_Nkando_deviation,
                  MAX_Lewa_deviation, MAX_oljogi_deviation, MAX_Chuka_deviation, MAX_nyeri_deviation, MAX_Sosian_deviation, MAX_OlPejeta_deviation, MAX_KIRARI_deviation, MAX_Kimanjo_deviation,
                  MAX_NAROMORU_deviation, MAX_Maralal_deviation, MAX_Wiyumririri_deviation, MAX_Murungaru_deviation, MAX_Timau_deviation, MAX_Gitinga_deviation, MAX_Aberdare_forest_deviation, MAX_Aberdare_National_deviation,
                  MAX_Lengalilla_deviation]
all_arrays_Feb_diu = [diurnal_Merti_mean, diurnal_Duma_mean, diurnal_Shura_mean, diurnal_Logologo_mean, diurnal_Meru_mean, diurnal_Mwingi_mean, diurnal_Wamba_mean, diurnal_OlLentille_mean, diurnal_Oldonyiro_mean, diurnal_Nkando_mean,
                  diurnal_Lewa_mean, diurnal_oljogi_mean, diurnal_Chuka_mean, diurnal_nyeri_mean, diurnal_Sosian_mean, diurnal_OlPejeta_mean, diurnal_KIRARI_mean, diurnal_Klimanjo_mean,
                  diurnal_NAROMORU_mean, diurnal_Maralal_mean, diurnal_Wiyumririri_mean, diurnal_Murungaru_mean, diurnal_Timau_mean, diurnal_Gitinga_mean, diurnal_Aberdare_forest_mean, diurnal_Aberdare_National_mean,
                  diurnal_Lengalilla_mean]
all_arrays_Feb_dev = [diurnal_Merti_deviation, diurnal_Duma_deviation, diurnal_Shura_deviation, diurnal_Logologo_deviation, diurnal_Meru_deviation, diurnal_Mwingi_deviation, diurnal_Wamba_deviation, diurnal_OlLentille_deviation, diurnal_Oldonyiro_deviation, diurnal_Nkando_deviation,
                  diurnal_Lewa_deviation, diurnal_oljogi_deviation, diurnal_Chuka_deviation, diurnal_nyeri_deviation, diurnal_Sosian_deviation, diurnal_OlPejeta_deviation, diurnal_KIRARI_deviation, diurnal_Klimanjo_deviation,
                  diurnal_NAROMORU_deviation, diurnal_Maralal_deviation, diurnal_Wiyumririri_deviation, diurnal_Murungaru_deviation, diurnal_Timau_deviation, diurnal_Gitinga_deviation, diurnal_Aberdare_forest_deviation, diurnal_Aberdare_National_deviation,
                  diurnal_Lengalilla_deviation]
all_arrays_Jun_diu = [Jdiurnal_Merti_mean, Jdiurnal_Duma_mean, Jdiurnal_Shura_mean, Jdiurnal_Logologo_mean, Jdiurnal_Meru_mean, Jdiurnal_Mwingi_mean, Jdiurnal_Wamba_mean, Jdiurnal_OlLentille_mean, Jdiurnal_Oldonyiro_mean, Jdiurnal_Nkando_mean,
                  Jdiurnal_Lewa_mean, Jdiurnal_oljogi_mean, Jdiurnal_Chuka_mean, Jdiurnal_nyeri_mean, Jdiurnal_Sosian_mean, Jdiurnal_OlPejeta_mean, Jdiurnal_KIRARI_mean, Jdiurnal_Klimanjo_mean,
                  Jdiurnal_NAROMORU_mean, Jdiurnal_Maralal_mean, Jdiurnal_Wiyumririri_mean, Jdiurnal_Murungaru_mean, Jdiurnal_Timau_mean, Jdiurnal_Gitinga_mean, Jdiurnal_Aberdare_forest_mean, Jdiurnal_Aberdare_National_mean,
                  Jdiurnal_Lengalilla_mean]
all_arrays_Jun_dev = [Jdiurnal_Merti_deviation, Jdiurnal_Duma_deviation, Jdiurnal_Shura_deviation, Jdiurnal_Logologo_deviation, Jdiurnal_Meru_deviation, Jdiurnal_Mwingi_deviation, Jdiurnal_Wamba_deviation, Jdiurnal_OlLentille_deviation, Jdiurnal_Oldonyiro_deviation, Jdiurnal_Nkando_deviation,
                  Jdiurnal_Lewa_deviation, Jdiurnal_oljogi_deviation, Jdiurnal_Chuka_deviation, Jdiurnal_nyeri_deviation, Jdiurnal_Sosian_deviation, Jdiurnal_OlPejeta_deviation, Jdiurnal_KIRARI_deviation, Jdiurnal_Klimanjo_deviation,
                  Jdiurnal_NAROMORU_deviation, Jdiurnal_Maralal_deviation, Jdiurnal_Wiyumririri_deviation, Jdiurnal_Murungaru_deviation, Jdiurnal_Timau_deviation, Jdiurnal_Gitinga_deviation, Jdiurnal_Aberdare_forest_deviation, Jdiurnal_Aberdare_National_deviation,
                  Jdiurnal_Lengalilla_deviation]
##################################################################################################################################







#read excel to get labels from AOIS list
db_aoi = pd.read_excel(r"C:\Users\icx\Desktop\MA_Q3\AOIs_MA_LST_JB.xlsx")
place = list(db_aoi.place)
rainfall = list(db_aoi.precipitation)
zone = list(db_aoi.zone)
altitude = list(db_aoi.meanaltitude)
county = list(db_aoi.county)
long = list(db_aoi.long)
lat = list(db_aoi.lat)
LCM = list(db_aoi.LCM)

colorlist = ["red", "gold", "darkorange", "darkviolet", "mediumblue", "lime", "darkred", "fuchsia", "deepskyblue"]

for sample in range(0, 9):
    print(colorlist[sample]+ matplotlib.colors.to_hex(colorlist[sample], keep_alpha=False))

    print("green" + matplotlib.colors.to_hex("green", keep_alpha=False))
    print("red" + matplotlib.colors.to_hex("red", keep_alpha=False))
    print("blue" + matplotlib.colors.to_hex("blue", keep_alpha=False))
    print("darkgreen" + matplotlib.colors.to_hex("darkgreen", keep_alpha=False))
    print("navy" + matplotlib.colors.to_hex("navy", keep_alpha=False))
    #SINGEL AOI PLOTTING APPENDIX
    ######################################################################################################################################################################


#SINGLE AOI
#define ANNUAL arrays
a = 26
b = a+1
for area in range(a, b):
    aoi = all_arrays_MAX_mean[area]
    aoi_dev = all_arrays_MAX_dev[area]
    # define DIURNAL ARRAYS
    #feb
    aoi_diu = all_arrays_Feb_diu[area]
    aoi_diu_dev = all_arrays_Feb_dev[area]
    #define DIURNAL ARRAYS
    #jun
    j_aoi_diu = all_arrays_Jun_diu[area]
    j_aoi_diu_dev = all_arrays_Jun_dev[area]
    title = LCM[area] + ", AOI #" + str(area+1) + " " + place[area] + ", " + county[area] + " County, Centre: LAT " + str(lat[area]) + ", LON " + str(long[area])
    subtitle = "[3x3 km], " + zone[area] + " Zone, " + "Mean Altitude: " +  str(altitude[area]) + " m, " +  "2012 - 2014 Seasonal $Precipitation_{mean}$: " + str(int(rainfall[area])) + " mm"
    #########################################################################
    # plot AOI all data
    import matplotlib.transforms
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D    "], fontsize=8)
    #ax1.set_xticks([0, 2.5,  5.5,  8.5, 11.5, 14.5, 17.5, 20.5, 23.5, 26.5, 29.5, 32.5])
    #ax1.set_xticklabels(["     Jan", "", "", "", "","       Jun", "", "", "", "", "", "      Dez"], fontsize=9)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # all 3x3 sd ranges
    for sample in range(1, 10):
        y1_min = list(map(sub, aoi[sample], aoi_dev[sample]))
        y1_max = list(map(add, aoi[sample], aoi_dev[sample]))
        plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=colorlist[sample-1], alpha=0.1, label="std dev")

    for sample in range(2, 10):
        singlepix = ax1.plot(t, aoi[sample], color=colorlist[sample-1], linewidth=0.8, label="")
    centre = ax1.plot(t, aoi[1], color=colorlist[0], linewidth=0.8, label="3x3 km")
    #aoi9x9 = ax1.plot(t, aoi[0], color="black", linewidth=1, label="9x9 km", alpha=0.8)
    #centre = ax1.plot(t, aoi[1], color="red", linewidth=1.5, alpha=0.7, label="3x3 km central pixel")
    ax1.set_ylim([-5,75])
    #plt.legend(fontsize=7)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue
    all9x9 = aoi_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    
    centre = aoi_diu[1]
    centre_dev = aoi_diu_dev[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    reordered_centre_dev = np.concatenate((centre_dev[84:], centre_dev[:84]), axis=0, out=None)
    
    a3 = aoi_diu[2]
    a3_dev = aoi_diu_dev[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a3re_dev = np.concatenate((a3_dev[84:], a3_dev[:84]), axis=0, out=None)
    
    a4 = aoi_diu[3]
    a4_dev = aoi_diu_dev[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a4re_dev = np.concatenate((a4_dev[84:], a4_dev[:84]), axis=0, out=None)
    
    a5 = aoi_diu[4]
    a5_dev = aoi_diu_dev[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a5re_dev = np.concatenate((a5_dev[84:], a5_dev[:84]), axis=0, out=None)
    
    a6 = aoi_diu[5]
    a6_dev = aoi_diu_dev[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a6re_dev = np.concatenate((a6_dev[84:], a6_dev[:84]), axis=0, out=None)
    
    a7 = aoi_diu[6]
    a7_dev = aoi_diu_dev[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a7re_dev = np.concatenate((a7_dev[84:], a7_dev[:84]), axis=0, out=None)
    
    a8 = aoi_diu[7]
    a8_dev = aoi_diu_dev[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a8re_dev = np.concatenate((a8_dev[84:], a8_dev[:84]), axis=0, out=None)
    
    a9 = aoi_diu[8]
    a9_dev = aoi_diu_dev[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a9re_dev = np.concatenate((a9_dev[84:], a9_dev[:84]), axis=0, out=None)
    
    a10 = aoi_diu[9]
    a10_dev = aoi_diu_dev[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    a10re_dev = np.concatenate((a10_dev[84:], a10_dev[:84]), axis=0, out=None)
    
    #dev 9x9
    #centre9x9dev = aoi_diu_dev[0]
    #reordered9x9dev = np.concatenate((centre9x9dev[84:], centre9x9dev[:84]), axis=0, out=None)
    all_3x3 = [reordered_centre, a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    all_3x3_dev = [reordered_centre_dev, a3re_dev, a4re_dev, a5re_dev, a6re_dev, a7re_dev, a8re_dev, a9re_dev, a10re_dev]
    
    for sample in range(0, 9):
        y1_min = list(map(sub, all_3x3[sample], all_3x3_dev[sample]))
        y1_max = list(map(add, all_3x3[sample], all_3x3_dev[sample]))
        plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=colorlist[sample], alpha=0.1, label="std dev")

    for sample in range(1, 9):
        singlepix = ax2.plot(t2, all_3x3[sample], color=colorlist[sample], linewidth=0.8, label="")
    centre = ax2.plot(t2, all_3x3[0], color=colorlist[0], linewidth=0.8, label="3x3 km")
    #aoi9x9 = ax2.plot(t2, aoi_diu[0], color="black", linewidth=1, label="9x9 km", alpha=0.3)

    ax2.set_ylim([-5,75])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue
    all9x9 = j_aoi_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)

    centre = j_aoi_diu[1]
    centre_dev = j_aoi_diu_dev[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    reordered_centre_dev = np.concatenate((centre_dev[84:], centre_dev[:84]), axis=0, out=None)

    a3 = j_aoi_diu[2]
    a3_dev = j_aoi_diu_dev[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a3re_dev = np.concatenate((a3_dev[84:], a3_dev[:84]), axis=0, out=None)

    a4 = j_aoi_diu[3]
    a4_dev = j_aoi_diu_dev[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a4re_dev = np.concatenate((a4_dev[84:], a4_dev[:84]), axis=0, out=None)

    a5 = j_aoi_diu[4]
    a5_dev = j_aoi_diu_dev[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a5re_dev = np.concatenate((a5_dev[84:], a5_dev[:84]), axis=0, out=None)

    a6 = j_aoi_diu[5]
    a6_dev = j_aoi_diu_dev[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a6re_dev = np.concatenate((a6_dev[84:], a6_dev[:84]), axis=0, out=None)

    a7 = j_aoi_diu[6]
    a7_dev = j_aoi_diu_dev[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a7re_dev = np.concatenate((a7_dev[84:], a7_dev[:84]), axis=0, out=None)

    a8 = j_aoi_diu[7]
    a8_dev = j_aoi_diu_dev[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a8re_dev = np.concatenate((a8_dev[84:], a8_dev[:84]), axis=0, out=None)

    a9 = j_aoi_diu[8]
    a9_dev = j_aoi_diu_dev[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a9re_dev = np.concatenate((a9_dev[84:], a9_dev[:84]), axis=0, out=None)

    a10 = j_aoi_diu[9]
    a10_dev = j_aoi_diu_dev[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    a10re_dev = np.concatenate((a10_dev[84:], a10_dev[:84]), axis=0, out=None)

    # dev 9x9
    # centre9x9dev = j_aoi_diu_dev[0]
    # reordered9x9dev = np.concatenate((centre9x9dev[84:], centre9x9dev[:84]), axis=0, out=None)
    all_3x3 = [reordered_centre, a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    all_3x3_dev = [reordered_centre_dev, a3re_dev, a4re_dev, a5re_dev, a6re_dev, a7re_dev, a8re_dev, a9re_dev,
                   a10re_dev]

    for sample in range(0, 9):
        y1_min = list(map(sub, all_3x3[sample], all_3x3_dev[sample]))
        y1_max = list(map(add, all_3x3[sample], all_3x3_dev[sample]))
        plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=colorlist[sample-1], alpha=0.1, label="" if sample == 2 else "")

    for sample in range(1, 9):
        singlepix = ax3.plot(t2, all_3x3[sample], color=colorlist[sample], linewidth=0.8, label="")
    centre = ax3.plot(t2, all_3x3[0], color=colorlist[0], linewidth=0.8, label="")
    #aoi9x9 = ax3.plot(t2, j_aoi_diu[0], color="black", linewidth=1, label="9x9 km", alpha=0.3)
    ax3.set_ylim([-5,75])
    #plt.legend(fontsize=7)





filename = str(a+1) + str(place[a]) +".png"
path = os.path.join(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_single", filename)
plt.savefig(path, dpi=400)







































# was not used
#
#
#
#
#
#
#


#SEMI HUMIDS##################################################################################################################################################
title = "Largescale Agriculture Nkando (Meru) vs. Wild Life Conservancy Lewa (Meru) vs. Agroforestry Nyeri (Nyeri)"
subtitle = "Semi-Humid Zone: Average Seasonal Precipitation (2012 - 2014): 450 - 510 mm"

#deifne ax lim y-ax LST
lim = 60
legsize = 7

label1 = "#10 Nkando (1,645 m)"
label2 = "#11 Lewa (1,639 m)"
label3 = "#14 Nyeri (1,766 m)"

aoi1 = MAX_Nkando_mean
aoi2 = MAX_Lewa_mean
aoi3 = MAX_nyeri_mean

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Nkando_mean
aoi2_diu = diurnal_Lewa_mean
aoi3_diu = diurnal_nyeri_mean
#jun
j_aoi1_diu = Jdiurnal_Nkando_mean
j_aoi2_diu = Jdiurnal_Lewa_mean
j_aoi3_diu = Jdiurnal_nyeri_mean

color1 = "red"
color11 = "darkred"
color2 = "darkorange"
color22 = "peru"
color3 = "blue"
color33 = "navy"

#Color1=red
#Color2=
#########################################################################
# plot AOI all data
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(5, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    ax1.set_xticklabels(["                    J F M A M J J A S O N D"], fontsize=8)
    #ax1.set_xticks([0, 2.5,  5.5,  8.5, 11.5, 14.5, 17.5, 20.5, 23.5, 26.5, 29.5, 32.5])
    #ax1.set_xticklabels(["     Jan", "", "", "", "","       Jun", "", "", "", "", "", "      Dez"], fontsize=9)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    for sample in range(2, aoi1.shape[0]):
        singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1, label="9x9 km", alpha=0.9)
    # AOI 3
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    for sample in range(2, aoi3.shape[0]):
        singlepix = ax1.plot(t, aoi3[sample], color=color33, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre2 = ax1.plot(t, aoi3[1], color = color33, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi3[0], color=color3, linewidth=1, label="9x9 km", alpha=0.9)
    ax1.set_ylim([0,lim])
    #plt.legend(fontsize=7)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    for sample in range(2, aoi2.shape[0]):
        singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1, label="9x9 km", alpha=0.9)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = aoi1_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = aoi1_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = aoi1_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = aoi1_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = aoi1_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = aoi1_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = aoi1_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = aoi1_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = aoi1_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    #dev 9x9
    #centre3x3dev = aoi1_diu_dev[1]
    #reordered3x3dev1 = np.concatenate((centre3x3dev[84:], centre3x3dev[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="LST standard deviation, 3x3 km (central pixel)")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9, color=color1, linewidth=1, label="9x9 km", alpha=0.9)

    #AOI 3
    all9x9 = aoi3_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = aoi3_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = aoi3_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = aoi3_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = aoi3_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = aoi3_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = aoi3_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = aoi3_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = aoi3_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = aoi3_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    #dev 9x9
    #centre3x3dev = aoi3_diu_dev[1]
    #reordered3x3dev = np.concatenate((centre3x3dev[84:], centre3x3dev[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="LST standard deviation, 3x3 km (central pixel)")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix3 = ax2.plot(t2, rest_of_3x3[sample], color=color33, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre3 = ax2.plot(t2, reordered_centre, color = color33, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9, color=color3, linewidth=1, label="9x9 km", alpha=0.9)


    #AOI 2
    all9x9 = aoi2_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = aoi2_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = aoi2_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = aoi2_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = aoi2_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = aoi2_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = aoi2_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = aoi2_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = aoi2_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = aoi2_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    #dev 9x9
    #centre3x3dev = aoi2_diu_dev[1]
    #reordered3x3dev = np.concatenate((centre3x3dev[84:], centre3x3dev[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="LST standard deviation, 3x3 km (central pixel)")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9, color=color2, linewidth=1, label="9x9 km", alpha=0.9)

    ax2.set_ylim([0,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = j_aoi1_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = j_aoi1_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = j_aoi1_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = j_aoi1_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = j_aoi1_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = j_aoi1_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = j_aoi1_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = j_aoi1_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = j_aoi1_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9, color=color1, linewidth=1, alpha=0.9, label=label1)

    # reorder time issue AOI 3
    all9x9 = j_aoi3_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = j_aoi3_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = j_aoi3_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = j_aoi3_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = j_aoi3_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = j_aoi3_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = j_aoi3_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = j_aoi3_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = j_aoi3_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = j_aoi3_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix3 = ax3.plot(t2, rest_of_3x3[sample], color=color33, linewidth=0.2)
    centre3 = ax3.plot(t2, reordered_centre, color = color33, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9, color=color3, linewidth=1, alpha=0.9, label = label3)

    # reorder time issue AOI 2
    all9x9 = j_aoi2_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = j_aoi2_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = j_aoi2_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = j_aoi2_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = j_aoi2_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = j_aoi2_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = j_aoi2_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = j_aoi2_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = j_aoi2_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = j_aoi2_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9, color=color2, linewidth=1, alpha=0.9, label= label2)
    ax3.set_ylim([0,lim])
    plt.legend(fontsize=legsize)


plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\semiHumid_areas.png", dpi = 400)












### SD 9x9 integrated not complete
#semi_arid areas##################################################################################################################################################
title = "Grassland/Rangeland Laikipia Plateau: Kimanjo vs. Sosian vs. Ol Pejeta"
subtitle = "Semi-Arid Zone: Average Seasonal Precipitation (2012 -2014): 340 mm"

#deifne ax lim y-ax LST
lim1_min = 0
lim1_max = 60

lim = 60
legsize = 7

label1 = "#18 Kimanjo (1,791 m)"
label2 = "#15 Sosian (1,766 m)"
label3 = "#16 Ol Pejeta (1,776 m)"

aoi1 = MAX_Kimanjo_mean
aoi1_dev = MAX_Kimanjo_deviation
aoi2 = MAX_Sosian_mean
aoi2_dev = MAX_Sosian_deviation
aoi3 = MAX_OlPejeta_mean
aoi3_dev = MAX_OlPejeta_deviation

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Klimanjo_mean
aoi1_diu_dev = diurnal_Klimanjo_deviation

aoi2_diu = diurnal_Sosian_mean
aoi2_diu_dev = diurnal_Sosian_deviation

aoi3_diu = diurnal_OlPejeta_mean
aoi3_diu_dev = diurnal_OlPejeta_deviation

#jun
j_aoi1_diu = Jdiurnal_Klimanjo_mean
j_aoi1_diu_dev = Jdiurnal_Klimanjo_deviation

j_aoi2_diu = Jdiurnal_Sosian_mean
j_aoi2_diu_dev = Jdiurnal_Sosian_deviation

j_aoi3_diu = Jdiurnal_OlPejeta_mean
j_aoi3_diu_dev = Jdiurnal_OlPejeta_deviation

color1 = "red"
color11 = "darkred"
color2 = "darkorange"
color22 = "peru"
color3 = "blue"
color33 = "navy"

#Color1=red
#Color2=
#########################################################################
# plot AOI all data
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(5, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    ax1.set_xticklabels(["                    J F M A M J J A S O N D"], fontsize=8)
    #ax1.set_xticks([0, 2.5,  5.5,  8.5, 11.5, 14.5, 17.5, 20.5, 23.5, 26.5, 29.5, 32.5])
    #ax1.set_xticklabels(["     Jan", "", "", "", "","       Jun", "", "", "", "", "", "      Dez"], fontsize=9)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    # 9x9 sd
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.3, label="")

    # 9x9 sd
    y1_min = list(map(sub, aoi2[0], aoi2_dev[0]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.3, label="")

    # 9x9 sd
    y1_min = list(map(sub, aoi3[0], aoi3_dev[0]))
    y1_max = list(map(add, aoi3[0], aoi3_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color3, alpha=0.3, label="")

    for sample in range(2, aoi1.shape[0]):
        singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1, label="9x9 km", alpha=0.9)
    # AOI 3
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    for sample in range(2, aoi3.shape[0]):
        singlepix = ax1.plot(t, aoi3[sample], color=color33, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre2 = ax1.plot(t, aoi3[1], color = color33, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi3[0], color=color3, linewidth=1, label="9x9 km", alpha=0.9)
    ax1.set_ylim([lim1_min,lim1_max])
    #plt.legend(fontsize=7)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    for sample in range(2, aoi2.shape[0]):
        singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1, label="9x9 km", alpha=0.9)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi3_diu[0]
    all9x9_dev = aoi3_diu_dev[0]
    reordered9x9_3 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_3 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    centre = aoi1_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = aoi1_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = aoi1_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = aoi1_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = aoi1_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = aoi1_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = aoi1_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = aoi1_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = aoi1_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    #dev 9x9
    #centre3x3dev = aoi1_diu_dev[1]
    #reordered3x3dev1 = np.concatenate((centre3x3dev[84:], centre3x3dev[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="LST standard deviation, 3x3 km (central pixel)")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t, y1_min, y1_max, color=color1, alpha=0.1, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t, y1_min, y1_max, color=color2, alpha=0.1, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_3, reordered9x9dev_3))
    y1_max = list(map(add, reordered9x9_3, reordered9x9dev_3))
    plt.subplot(ax2).fill_between(t, y1_min, y1_max, color=color3, alpha=0.1, label="")


    for sample in range(0, 8):
        singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9, color=color1, linewidth=1, label="9x9 km", alpha=0.9)

    #AOI 3
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi3_diu[0]
    all9x9_dev = j_aoi3_diu_dev[0]
    reordered9x9_3 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_3 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    centre = aoi3_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = aoi3_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = aoi3_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = aoi3_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = aoi3_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = aoi3_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = aoi3_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = aoi3_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = aoi3_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    #dev 9x9
    #centre3x3dev = aoi3_diu_dev[1]
    #reordered3x3dev = np.concatenate((centre3x3dev[84:], centre3x3dev[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="LST standard deviation, 3x3 km (central pixel)")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]


    for sample in range(0, 8):
        singlepix3 = ax2.plot(t2, rest_of_3x3[sample], color=color33, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre3 = ax2.plot(t2, reordered_centre, color = color33, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9, color=color3, linewidth=1, label="9x9 km", alpha=0.9)


    #AOI 2
    all9x9 = aoi2_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = aoi2_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = aoi2_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = aoi2_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = aoi2_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = aoi2_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = aoi2_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = aoi2_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = aoi2_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = aoi2_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    #dev 9x9
    #centre3x3dev = aoi2_diu_dev[1]
    #reordered3x3dev = np.concatenate((centre3x3dev[84:], centre3x3dev[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="LST standard deviation, 3x3 km (central pixel)")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]

    for sample in range(0, 8):
        singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9, color=color2, linewidth=1, label="9x9 km", alpha=0.9)

    ax2.set_ylim([0,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi3_diu[0]
    all9x9_dev = aoi3_diu_dev[0]
    reordered9x9_3 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_3 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    centre = j_aoi1_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = j_aoi1_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = j_aoi1_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = j_aoi1_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = j_aoi1_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = j_aoi1_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = j_aoi1_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = j_aoi1_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = j_aoi1_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t, y1_min, y1_max, color=color1, alpha=0.1, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t, y1_min, y1_max, color=color2, alpha=0.1, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_3, reordered9x9dev_3))
    y1_max = list(map(add, reordered9x9_3, reordered9x9dev_3))
    plt.subplot(ax2).fill_between(t, y1_min, y1_max, color=color3, alpha=0.1, label="")


    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9, color=color1, linewidth=1, alpha=0.9, label=label1)

    # reorder time issue AOI 3
    all9x9 = j_aoi3_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = j_aoi3_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = j_aoi3_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = j_aoi3_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = j_aoi3_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = j_aoi3_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = j_aoi3_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = j_aoi3_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = j_aoi3_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = j_aoi3_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix3 = ax3.plot(t2, rest_of_3x3[sample], color=color33, linewidth=0.2)
    centre3 = ax3.plot(t2, reordered_centre, color = color33, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9, color=color3, linewidth=1, alpha=0.9, label = label3)

    # reorder time issue AOI 2
    all9x9 = j_aoi2_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = j_aoi2_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = j_aoi2_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = j_aoi2_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = j_aoi2_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = j_aoi2_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = j_aoi2_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = j_aoi2_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = j_aoi2_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = j_aoi2_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]

    for sample in range(0, 8):
        singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9, color=color2, linewidth=1, alpha=0.9, label= label2)
    ax3.set_ylim([0,lim])
    plt.legend(fontsize=legsize)

plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\grassland_semiArid_areas_plateau.png", dpi = 400)








# 3 comparison aoi
#HIHGLAND##################################################################################################################################################
title = "Grassland/Rangeland Laikipia Plateau: Kimanjo vs. Sosian vs. Ol Pejeta"
subtitle = "Semi-Arid Zone: Average Seasonal Precipitation (2012-14): 340 mm"

#deifne ax lim y-ax LST
lim1_min = 0
lim1_max = 50

lim = 50
legsize = 7

label1 = "#18 Kimanjo (1,791 m)"
label2 = "#15 Sosian (1,766 m)"
label3 = "#16 Ol Pejeta (1,776 m)"

aoi1 = MAX_Timau_mean
aoi2 = MAX_Lengalilla_mean
aoi3 = MAX_Gitinga_mean

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Timau_mean
aoi2_diu = diurnal_Lengalilla_mean
aoi3_diu = diurnal_Gitinga_mean
#jun
j_aoi1_diu = Jdiurnal_Timau_mean
j_aoi2_diu = Jdiurnal_Lengalilla_mean
j_aoi3_diu = Jdiurnal_Gitinga_mean

color1 = "red"
color11 = "darkred"
color2 = "darkorange"
color22 = "peru"
color3 = "blue"
color33 = "navy"

#Color1=red
#Color2=
#########################################################################
# plot AOI all data
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(5, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    ax1.set_xticklabels(["                    J F M A M J J A S O N D"], fontsize=8)
    #ax1.set_xticks([0, 2.5,  5.5,  8.5, 11.5, 14.5, 17.5, 20.5, 23.5, 26.5, 29.5, 32.5])
    #ax1.set_xticklabels(["     Jan", "", "", "", "","       Jun", "", "", "", "", "", "      Dez"], fontsize=9)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Decadal_{mean}$ $LST_{max}$",fontsize=8.5)

    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    for sample in range(2, aoi1.shape[0]):
        singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1, label="9x9 km", alpha=0.9)
    # AOI 3
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    for sample in range(2, aoi3.shape[0]):
        singlepix = ax1.plot(t, aoi3[sample], color=color33, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre2 = ax1.plot(t, aoi3[1], color = color33, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi3[0], color=color3, linewidth=1, label="9x9 km", alpha=0.9)
    ax1.set_ylim([lim1_min,lim1_max])
    #plt.legend(fontsize=7)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    for sample in range(2, aoi2.shape[0]):
        singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1, label="9x9 km", alpha=0.9)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("$February_{mean}$ diurnal $LST_{var}$",fontsize=8.5)
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = aoi1_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = aoi1_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = aoi1_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = aoi1_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = aoi1_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = aoi1_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = aoi1_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = aoi1_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = aoi1_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    #dev 9x9
    #centre3x3dev = aoi1_diu_dev[1]
    #reordered3x3dev1 = np.concatenate((centre3x3dev[84:], centre3x3dev[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="LST standard deviation, 3x3 km (central pixel)")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9, color=color1, linewidth=1, label="9x9 km", alpha=0.9)

    #AOI 3
    all9x9 = aoi3_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = aoi3_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = aoi3_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = aoi3_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = aoi3_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = aoi3_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = aoi3_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = aoi3_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = aoi3_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = aoi3_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    #dev 9x9
    #centre3x3dev = aoi3_diu_dev[1]
    #reordered3x3dev = np.concatenate((centre3x3dev[84:], centre3x3dev[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="LST standard deviation, 3x3 km (central pixel)")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix3 = ax2.plot(t2, rest_of_3x3[sample], color=color33, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre3 = ax2.plot(t2, reordered_centre, color = color33, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9, color=color3, linewidth=1, label="9x9 km", alpha=0.9)


    #AOI 2
    all9x9 = aoi2_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = aoi2_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = aoi2_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = aoi2_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = aoi2_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = aoi2_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = aoi2_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = aoi2_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = aoi2_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = aoi2_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)
    #dev 9x9
    #centre3x3dev = aoi2_diu_dev[1]
    #reordered3x3dev = np.concatenate((centre3x3dev[84:], centre3x3dev[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="LST standard deviation, 3x3 km (central pixel)")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9, color=color2, linewidth=1, label="9x9 km", alpha=0.9)

    ax2.set_ylim([0,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("$June_{mean}$ diurnal $LST_{var}$",fontsize=8.5)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = j_aoi1_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = j_aoi1_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = j_aoi1_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = j_aoi1_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = j_aoi1_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = j_aoi1_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = j_aoi1_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = j_aoi1_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = j_aoi1_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9, color=color1, linewidth=1, alpha=0.9, label=label1)

    # reorder time issue AOI 3
    all9x9 = j_aoi3_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = j_aoi3_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = j_aoi3_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = j_aoi3_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = j_aoi3_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = j_aoi3_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = j_aoi3_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = j_aoi3_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = j_aoi3_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = j_aoi3_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix3 = ax3.plot(t2, rest_of_3x3[sample], color=color33, linewidth=0.2)
    centre3 = ax3.plot(t2, reordered_centre, color = color33, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9, color=color3, linewidth=1, alpha=0.9, label = label3)

    # reorder time issue AOI 2
    all9x9 = j_aoi2_diu[0]
    reordered9x9 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    centre = j_aoi2_diu[1]
    reordered_centre = np.concatenate((centre[84:], centre[:84]), axis=0, out=None)
    a3 = j_aoi2_diu[2]
    a3re = np.concatenate((a3[84:], a3[:84]), axis=0, out=None)
    a4 = j_aoi2_diu[3]
    a4re = np.concatenate((a4[84:], a4[:84]), axis=0, out=None)
    a5 = j_aoi2_diu[4]
    a5re = np.concatenate((a5[84:], a5[:84]), axis=0, out=None)
    a6 = j_aoi2_diu[5]
    a6re = np.concatenate((a6[84:], a6[:84]), axis=0, out=None)
    a7 = j_aoi2_diu[6]
    a7re = np.concatenate((a7[84:], a7[:84]), axis=0, out=None)
    a8 = j_aoi2_diu[7]
    a8re = np.concatenate((a8[84:], a8[:84]), axis=0, out=None)
    a9 = j_aoi2_diu[8]
    a9re = np.concatenate((a9[84:], a9[:84]), axis=0, out=None)
    a10 = j_aoi2_diu[9]
    a10re = np.concatenate((a10[84:], a10[:84]), axis=0, out=None)

    # dev centrepix 3x3
    #y1_min = list(map(sub, reordered_centre, reordered3x3dev))
    #y1_max = list(map(add, reordered_centre, reordered3x3dev))
    #plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    rest_of_3x3 = [a3re, a4re, a5re, a6re, a7re, a8re, a9re, a10re]
    for sample in range(0, 8):
        singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9, color=color2, linewidth=1, alpha=0.9, label= label2)
    ax3.set_ylim([0,lim])
    plt.legend(fontsize=legsize)

plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\highlands.png", dpi = 400)

















#2 AOIS ----- FOREST vs agroforestry
#
# ##################################################################################################################################################
title = "Agroforestry Kirari, Embu County (1,780 m) vs. Rain Forest Chuka, Tharaka-Nithi County (1,733 m)"
subtitle = "[9x9 km], Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"

#deifne ax lim y-ax LST
lim = 60
lim2 = -5
legsize = 6.5

label1 = "Agroforestry"
label2 = "Rain Forest"
#label3 = "#16 Ol Pejeta (1,776 m)"

aoi1 = MAX_KIRARI_mean
aoi1_dev = MAX_KIRARI_deviation

aoi2 = MAX_Chuka_mean
aoi2_dev = MAX_Chuka_deviation

dif1 = aoi1[0] - aoi2[0]

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_KIRARI_mean
aoi1_diu_dev = diurnal_KIRARI_deviation

aoi2_diu = diurnal_Chuka_mean
aoi2_diu_dev = diurnal_Chuka_deviation

dif2 = aoi1_diu[0] - aoi2_diu[0]
dif2 = np.concatenate((dif2[84:], dif2[:84]), axis=0, out=None)

#jun
j_aoi1_diu = Jdiurnal_KIRARI_mean
j_aoi1_diu_dev = Jdiurnal_KIRARI_deviation

j_aoi2_diu = Jdiurnal_Chuka_mean
j_aoi2_diu_dev = Jdiurnal_Chuka_deviation

dif3 = j_aoi1_diu[0] - j_aoi2_diu[0]
dif3 = np.concatenate((dif3[84:], dif3[:84]), axis=0, out=None)


color1 = "blue"
color11 = "navy"
color2 = "green"
color22 = "green"
#color3 = "blue"
#color33 = "navy"

#Color1=red
#Color2=
#########################################################################
# plot AOI all data

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    #ax1.set_xticklabels(["                        J F M A M J J A S O N D"], fontsize=9)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D     "], fontsize=8)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # 9x9 sd AOI1
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.2, label="SD")

    # 9x9 sd AOI2
    y1_min = list(map(sub, aoi2[0], aoi2_dev[1]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[1]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.2, label="SD")
    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi1.shape[0]):
     #   singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi2.shape[0]):
    #    singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif1 = ax1.plot(t, dif1, color="black", linewidth=1.3, label="dif", alpha=0.95)

    ax1.set_ylim([lim2,lim])
#plt.legend(fontsize=legsize)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="")

    #for sample in range(0, 8):
    #    singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)


    #for sample in range(0, 8):
    #    singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif2 = ax2.plot(t2, dif2, color="black", linewidth=1.3, label="dif", alpha=0.95)
    ax2.set_ylim([lim2,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="range of SD "+label1)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="range of SD "+label2)


    #for sample in range(0, 8):
    #    singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    #centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, alpha=0.95, label="")
    #for sample in range(0, 8):
    #    singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    #centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, alpha=0.95, label= "")
    dif3 = ax3.plot(t2, dif3, color="black", linewidth=1.3, label="", alpha=0.95)
    ax3.set_ylim([lim2,lim])
    plt.legend(fontsize=legsize)


plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\humid_areas_2.png", dpi = 400)






#2 AOIS ----- HIGHLAND AGRICULTURE vs forest
#
# ##################################################################################################################################################
title = "Largescale Agriculture Timau, Meru County (2,634 m) vs. Rain Forest Gitinga, Meru & Nyeri County (2,642 m)"
subtitle = "[9x9 km], Sub-Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"

#deifne ax lim y-ax LST
lim = 60
lim2 = -5
legsize = 6.5

color1 = "red"
color11 = "darkred"
color2 = "green"
color22 = "green"
#color3 = "blue"
#color33 = "navy"

label1 = "Agriculture"
label2 = "Rain Forest"
#label3 = "#16 Ol Pejeta (1,776 m)"

aoi1 = MAX_Timau_mean
aoi1_dev = MAX_Timau_deviation

aoi2 = MAX_Gitinga_mean
aoi2_dev = MAX_Gitinga_deviation

dif1 = aoi1[0] - aoi2[0]

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Timau_mean
aoi1_diu_dev = diurnal_Timau_deviation

aoi2_diu = diurnal_Gitinga_mean
aoi2_diu_dev = diurnal_Gitinga_deviation

dif2 = aoi1_diu[0] - aoi2_diu[0]
dif2 = np.concatenate((dif2[84:], dif2[:84]), axis=0, out=None)

#jun
j_aoi1_diu = Jdiurnal_Timau_mean
j_aoi1_diu_dev = Jdiurnal_Timau_deviation

j_aoi2_diu = Jdiurnal_Gitinga_mean
j_aoi2_diu_dev = Jdiurnal_Gitinga_deviation

dif3 = j_aoi1_diu[0] - j_aoi2_diu[0]
dif3 = np.concatenate((dif3[84:], dif3[:84]), axis=0, out=None)


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    #ax1.set_xticklabels(["                        J F M A M J J A S O N D"], fontsize=9)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D     "], fontsize=8)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # 9x9 sd AOI1
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.2, label="SD")

    # 9x9 sd AOI2
    y1_min = list(map(sub, aoi2[0], aoi2_dev[1]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[1]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.2, label="SD")
    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi1.shape[0]):
     #   singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi2.shape[0]):
    #    singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif1 = ax1.plot(t, dif1, color="black", linewidth=1.3, label="dif", alpha=0.95)

    ax1.set_ylim([lim2,lim])
#plt.legend(fontsize=legsize)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="")

    #for sample in range(0, 8):
    #    singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)


    #for sample in range(0, 8):
    #    singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif2 = ax2.plot(t2, dif2, color="black", linewidth=1.3, label="dif", alpha=0.95)
    ax2.set_ylim([lim2,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="range of SD "+label1)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="range of SD "+label2)


    #for sample in range(0, 8):
    #    singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    #centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, alpha=0.95, label="")
    #for sample in range(0, 8):
    #    singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    #centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, alpha=0.95, label= "")
    dif3 = ax3.plot(t2, dif3, color="black", linewidth=1.3, label="", alpha=0.95)
    ax3.set_ylim([lim2,lim])
    plt.legend(fontsize=legsize)


plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\sub-humid_areas_highland2.png", dpi = 400)





# ##################################################################################################################################################
title = "Agriculture Naro Moru, Nyeri County (1,937 m) vs. Maralal National Sanctuary, Samburu County (1,949 m)"
subtitle = "[9x9 km], Semi-Humid/Semi-Arid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 312 - 381 mm"
#"[9x9 km], Sub-Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"
#deifne ax lim y-ax LST
lim = 60
lim2 = -5
legsize = 6.5

label1 = "Agriculture"
label2 = "Sanctuary"
#label3 = "#16 Ol Pejeta (1,776 m)"

aoi1 = MAX_NAROMORU_mean
aoi1_dev = MAX_NAROMORU_deviation

aoi2 = MAX_Maralal_mean
aoi2_dev = MAX_Maralal_deviation

dif1 = aoi1[0] - aoi2[0]

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_NAROMORU_mean
aoi1_diu_dev = diurnal_NAROMORU_deviation

aoi2_diu = diurnal_Maralal_mean
aoi2_diu_dev = diurnal_Maralal_deviation

dif2 = aoi1_diu[0] - aoi2_diu[0]
dif2 = np.concatenate((dif2[84:], dif2[:84]), axis=0, out=None)

#jun
j_aoi1_diu = Jdiurnal_NAROMORU_mean
j_aoi1_diu_dev = Jdiurnal_NAROMORU_deviation

j_aoi2_diu = Jdiurnal_Maralal_mean
j_aoi2_diu_dev = Jdiurnal_Maralal_deviation

dif3 = j_aoi1_diu[0] - j_aoi2_diu[0]
dif3 = np.concatenate((dif3[84:], dif3[:84]), axis=0, out=None)



color1 = "red"
color11 = "darkred"
color2 = "blue"
color22 = "navy"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    #ax1.set_xticklabels(["                        J F M A M J J A S O N D"], fontsize=9)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D     "], fontsize=8)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # 9x9 sd AOI1
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.2, label="SD")

    # 9x9 sd AOI2
    y1_min = list(map(sub, aoi2[0], aoi2_dev[1]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[1]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.2, label="SD")
    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi1.shape[0]):
     #   singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi2.shape[0]):
    #    singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif1 = ax1.plot(t, dif1, color="black", linewidth=1.3, label="dif", alpha=0.95)

    ax1.set_ylim([lim2,lim])
#plt.legend(fontsize=legsize)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="")

    #for sample in range(0, 8):
    #    singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)


    #for sample in range(0, 8):
    #    singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif2 = ax2.plot(t2, dif2, color="black", linewidth=1.3, label="dif", alpha=0.95)
    ax2.set_ylim([lim2,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="range of SD "+label1)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="range of SD "+label2)


    #for sample in range(0, 8):
    #    singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    #centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, alpha=0.95, label="")
    #for sample in range(0, 8):
    #    singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    #centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, alpha=0.95, label= "")
    dif3 = ax3.plot(t2, dif3, color="black", linewidth=1.3, label="", alpha=0.95)
    ax3.set_ylim([lim2,lim])
    plt.legend(fontsize=legsize)


plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\semi_aridhumid_areas_plateau.png", dpi = 400)




# ##################################################################################################################################################
title = "Aberdare Forest, Nyeri County (3,000 m) vs. Degraded Forest Lengalilla Mt. Kenya, Meru County (3,360 m)"
subtitle = "[9x9 km], Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 560 - 582 mm"
#"[9x9 km], Sub-Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"
#deifne ax lim y-ax LST
lim = 50
lim2 = -5
legsize = 6.5

label1 = "Degraded Forest"
label2 = "Aberdare Forest"
#label3 = "#16 Ol Pejeta (1,776 m)"

aoi1 = MAX_Lengalilla_mean
aoi1_dev = MAX_Lengalilla_deviation

aoi2 = MAX_Aberdare_forest_mean
aoi2_dev = MAX_Aberdare_forest_deviation

dif1 = aoi1[0] - aoi2[0]

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Lengalilla_mean
aoi1_diu_dev = diurnal_Lengalilla_deviation

aoi2_diu = diurnal_Aberdare_forest_mean
aoi2_diu_dev = diurnal_Aberdare_forest_deviation

dif2 = aoi1_diu[0] - aoi2_diu[0]
dif2 = np.concatenate((dif2[84:], dif2[:84]), axis=0, out=None)

#jun
j_aoi1_diu = Jdiurnal_Lengalilla_mean
j_aoi1_diu_dev = Jdiurnal_Lengalilla_deviation

j_aoi2_diu = Jdiurnal_Aberdare_forest_mean
j_aoi2_diu_dev = Jdiurnal_Aberdare_forest_deviation

dif3 = j_aoi1_diu[0] - j_aoi2_diu[0]
dif3 = np.concatenate((dif3[84:], dif3[:84]), axis=0, out=None)


color1 = "red"
color11 = "darkred"
color2 = "green"
color22 = "green"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    #ax1.set_xticklabels(["                        J F M A M J J A S O N D"], fontsize=9)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D     "], fontsize=8)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # 9x9 sd AOI1
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.2, label="SD")

    # 9x9 sd AOI2
    y1_min = list(map(sub, aoi2[0], aoi2_dev[1]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[1]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.2, label="SD")
    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi1.shape[0]):
     #   singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi2.shape[0]):
    #    singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif1 = ax1.plot(t, dif1, color="black", linewidth=1.3, label="dif", alpha=0.95)

    ax1.set_ylim([lim2,lim])
#plt.legend(fontsize=legsize)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="")

    #for sample in range(0, 8):
    #    singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)


    #for sample in range(0, 8):
    #    singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif2 = ax2.plot(t2, dif2, color="black", linewidth=1.3, label="dif", alpha=0.95)
    ax2.set_ylim([lim2,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="range of SD "+label1)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="range of SD "+label2)


    #for sample in range(0, 8):
    #    singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    #centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, alpha=0.95, label="")
    #for sample in range(0, 8):
    #    singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    #centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, alpha=0.95, label= "")
    dif3 = ax3.plot(t2, dif3, color="black", linewidth=1.3, label="", alpha=0.95)
    ax3.set_ylim([lim2,lim])
    plt.legend(fontsize=legsize)


plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\foresxt_degradation_humid.png", dpi = 400)


# ##################################################################################################################################################
title = "Agriculture Nkando, Meru County (1,645 m) vs. Lewa Wild Life Conservancy, Meru County (1,649 m)"
subtitle = "[9x9 km], Semi-Arid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 450 - 510 mm"
#"[9x9 km], Sub-Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"
#deifne ax lim y-ax LST
lim = 60
lim2 = -3
legsize = 6.5

label1 = "Agriculture"
label2 = "Conservancy"
#label3 = "#16 Ol Pejeta (1,776 m)"


aoi1 = MAX_Nkando_mean
aoi1_dev = MAX_Nkando_deviation

aoi2 = MAX_Lewa_mean
aoi2_dev = MAX_Lewa_deviation

dif1 = aoi1[0] - aoi2[0]

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Nkando_mean
aoi1_diu_dev = diurnal_Nkando_deviation

aoi2_diu = diurnal_Lewa_mean
aoi2_diu_dev = diurnal_Lewa_deviation

dif2 = aoi1_diu[0] - aoi2_diu[0]
dif2 = np.concatenate((dif2[84:], dif2[:84]), axis=0, out=None)

#jun
j_aoi1_diu = Jdiurnal_Nkando_mean
j_aoi1_diu_dev = Jdiurnal_Nkando_deviation

j_aoi2_diu = Jdiurnal_Lewa_mean
j_aoi2_diu_dev = Jdiurnal_Lewa_deviation

dif3 = j_aoi1_diu[0] - j_aoi2_diu[0]
dif3 = np.concatenate((dif3[84:], dif3[:84]), axis=0, out=None)


color1 = "red"
color11 = "darkred"
color2 = "blue"
color22 = "navy"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    #ax1.set_xticklabels(["                        J F M A M J J A S O N D"], fontsize=9)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D     "], fontsize=8)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # 9x9 sd AOI1
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.2, label="SD")

    # 9x9 sd AOI2
    y1_min = list(map(sub, aoi2[0], aoi2_dev[1]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[1]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.2, label="SD")
    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi1.shape[0]):
     #   singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi2.shape[0]):
    #    singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif1 = ax1.plot(t, dif1, color="black", linewidth=1.3, label="dif", alpha=0.95)

    ax1.set_ylim([lim2,lim])
#plt.legend(fontsize=legsize)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="")

    #for sample in range(0, 8):
    #    singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)


    #for sample in range(0, 8):
    #    singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif2 = ax2.plot(t2, dif2, color="black", linewidth=1.3, label="dif", alpha=0.95)
    ax2.set_ylim([lim2,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="range of SD "+label1)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="range of SD "+label2)


    #for sample in range(0, 8):
    #    singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    #centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, alpha=0.95, label="")
    #for sample in range(0, 8):
    #    singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    #centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, alpha=0.95, label= "")
    dif3 = ax3.plot(t2, dif3, color="black", linewidth=1.3, label="", alpha=0.95)
    ax3.set_ylim([lim2,lim])
    plt.legend(fontsize=legsize)


plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\semiarid_middlezone.png", dpi = 400)





# ##################################################################################################################################################
title = "Grassland Wamba, Samburu County (1,232 m) vs. Ol Lentille Wild Life Conservancy, Isiolo County (1,397 m)"
subtitle = "[9x9 km], Arid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 380 - 410 mm"
#"[9x9 km], Sub-Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"
#deifne ax lim y-ax LST
lim = 65
lim2 = -3
legsize = 6.5

label1 = "Grassland"
label2 = "Conservancy"
#label3 = "#16 Ol Pejeta (1,776 m)"


aoi1 = MAX_Wamba_mean
aoi1_dev = MAX_Wamba_deviation

aoi2 = MAX_OlLentille_mean
aoi2_dev = MAX_OlLentille_deviation

dif1 = aoi1[0] - aoi2[0]

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Wamba_mean
aoi1_diu_dev = diurnal_Wamba_deviation

aoi2_diu = diurnal_OlLentille_mean
aoi2_diu_dev = diurnal_OlLentille_deviation

dif2 = aoi1_diu[0] - aoi2_diu[0]
dif2 = np.concatenate((dif2[84:], dif2[:84]), axis=0, out=None)

#jun
j_aoi1_diu = Jdiurnal_Wamba_mean
j_aoi1_diu_dev = Jdiurnal_Wamba_deviation

j_aoi2_diu = Jdiurnal_OlLentille_mean
j_aoi2_diu_dev = Jdiurnal_OlLentille_deviation

dif3 = j_aoi1_diu[0] - j_aoi2_diu[0]
dif3 = np.concatenate((dif3[84:], dif3[:84]), axis=0, out=None)

color1 = "red"
color11 = "darkred"
color2 = "green"
color22 = "green"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    #ax1.set_xticklabels(["                        J F M A M J J A S O N D"], fontsize=9)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D     "], fontsize=8)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # 9x9 sd AOI1
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.2, label="SD")

    # 9x9 sd AOI2
    y1_min = list(map(sub, aoi2[0], aoi2_dev[1]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[1]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.2, label="SD")
    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi1.shape[0]):
     #   singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi2.shape[0]):
    #    singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif1 = ax1.plot(t, dif1, color="black", linewidth=1.3, label="dif", alpha=0.95)

    ax1.set_ylim([lim2,lim])
#plt.legend(fontsize=legsize)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="")

    #for sample in range(0, 8):
    #    singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)


    #for sample in range(0, 8):
    #    singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif2 = ax2.plot(t2, dif2, color="black", linewidth=1.3, label="dif", alpha=0.95)
    ax2.set_ylim([lim2,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="range of SD "+label1)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="range of SD "+label2)


    #for sample in range(0, 8):
    #    singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    #centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, alpha=0.95, label="")
    #for sample in range(0, 8):
    #    singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    #centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, alpha=0.95, label= "")
    dif3 = ax3.plot(t2, dif3, color="black", linewidth=1.3, label="", alpha=0.95)
    ax3.set_ylim([lim2,lim])
    plt.legend(fontsize=legsize)



plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\arid_middlezone.png", dpi = 400)




# ##################################################################################################################################################
title = "Mixed Crop and Livestock Areas Mwingi, Kitui County (669 m) vs. Meru National Park, Meru County (609 m)"
subtitle = "[9x9 km], Arid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 250 mm"
#"[9x9 km], Sub-Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"
#deifne ax lim y-ax LST
lim = 70
lim2 = -3
legsize = 6.5

label1 = "Mixed Areas"
label2 = "National Park"
#label3 = "#16 Ol Pejeta (1,776 m)"


aoi1 = MAX_Mwingi_mean
aoi1_dev = MAX_Mwingi_deviation

aoi2 = MAX_Meru_mean
aoi2_dev = MAX_Meru_deviation

dif1 = aoi1[0] - aoi2[0]

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Mwingi_mean
aoi1_diu_dev = diurnal_Mwingi_deviation

aoi2_diu = diurnal_Meru_mean
aoi2_diu_dev = diurnal_Meru_deviation

dif2 = aoi1_diu[0] - aoi2_diu[0]
dif2 = np.concatenate((dif2[84:], dif2[:84]), axis=0, out=None)

#jun
j_aoi1_diu = Jdiurnal_Mwingi_mean
j_aoi1_diu_dev = Jdiurnal_Mwingi_deviation

j_aoi2_diu = Jdiurnal_Meru_mean
j_aoi2_diu_dev = Jdiurnal_Meru_deviation

dif3 = j_aoi1_diu[0] - j_aoi2_diu[0]
dif3 = np.concatenate((dif3[84:], dif3[:84]), axis=0, out=None)


color1 = "red"
color11 = "darkred"
color2 = "green"
color22 = "green"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    #ax1.set_xticklabels(["                        J F M A M J J A S O N D"], fontsize=9)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D     "], fontsize=8)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # 9x9 sd AOI1
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.2, label="SD")

    # 9x9 sd AOI2
    y1_min = list(map(sub, aoi2[0], aoi2_dev[1]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[1]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.2, label="SD")
    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi1.shape[0]):
     #   singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi2.shape[0]):
    #    singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif1 = ax1.plot(t, dif1, color="black", linewidth=1.3, label="dif", alpha=0.95)

    ax1.set_ylim([lim2,lim])
#plt.legend(fontsize=legsize)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="")

    #for sample in range(0, 8):
    #    singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)


    #for sample in range(0, 8):
    #    singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif2 = ax2.plot(t2, dif2, color="black", linewidth=1.3, label="dif", alpha=0.95)
    ax2.set_ylim([lim2,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="range of SD "+label1)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="range of SD "+label2)


    #for sample in range(0, 8):
    #    singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    #centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, alpha=0.95, label="")
    #for sample in range(0, 8):
    #    singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    #centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, alpha=0.95, label= "")
    dif3 = ax3.plot(t2, dif3, color="black", linewidth=1.3, label="", alpha=0.95)
    ax3.set_ylim([lim2,lim])
    plt.legend(fontsize=legsize)


plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\arid_lowland.png", dpi = 400)




# ##################################################################################################################################################
title = "Livestock-only Areas Logologo, Marsabit County (419 m) vs. Shura, Marsabit County (450 m)"
subtitle = "[9x9 km], Very Arid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 160 mm"
#"[9x9 km], Sub-Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"
#deifne ax lim y-ax LST
lim = 70
lim2 = -3
legsize = 6

label1 = "Logologo"
label2 = "Shura"
#label3 = "#16 Ol Pejeta (1,776 m)"


#
aoi1 = MAX_Logologo_mean
aoi1_dev = MAX_Logologo_deviation

aoi2 = MAX_Shura_mean
aoi2_dev = MAX_Shura_deviation

dif1 = aoi1[0] - aoi2[0]

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Logologo_mean
aoi1_diu_dev = diurnal_Logologo_deviation

aoi2_diu = diurnal_Shura_mean
aoi2_diu_dev = diurnal_Shura_deviation

dif2 = aoi1_diu[0] - aoi2_diu[0]
dif2 = np.concatenate((dif2[84:], dif2[:84]), axis=0, out=None)

#jun
j_aoi1_diu = Jdiurnal_Logologo_mean
j_aoi1_diu_dev = Jdiurnal_Logologo_deviation

j_aoi2_diu = Jdiurnal_Shura_mean
j_aoi2_diu_dev = Jdiurnal_Shura_deviation

dif3 = j_aoi1_diu[0] - j_aoi2_diu[0]
dif3 = np.concatenate((dif3[84:], dif3[:84]), axis=0, out=None)

color1 = "darkred"
color11 = "darkred"
color2 = "navy"
color22 = "saddlebrown"

#plot
#plot
# plot AOI all data
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    #ax1.set_xticklabels(["                        J F M A M J J A S O N D"], fontsize=9)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D     "], fontsize=8)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # 9x9 sd AOI1
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.2, label="SD")

    # 9x9 sd AOI2
    y1_min = list(map(sub, aoi2[0], aoi2_dev[1]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[1]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.2, label="SD")
    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi1.shape[0]):
     #   singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi2.shape[0]):
    #    singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif1 = ax1.plot(t, dif1, color="black", linewidth=1.3, label="dif", alpha=0.95)

    ax1.set_ylim([lim2,lim])
#plt.legend(fontsize=legsize)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="")

    #for sample in range(0, 8):
    #    singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)


    #for sample in range(0, 8):
    #    singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif2 = ax2.plot(t2, dif2, color="black", linewidth=1.3, label="dif", alpha=0.95)
    ax2.set_ylim([lim2,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="range of SD "+label1)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="range of SD "+label2)


    #for sample in range(0, 8):
    #    singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    #centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, alpha=0.95, label="")
    #for sample in range(0, 8):
    #    singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    #centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, alpha=0.95, label= "")
    dif3 = ax3.plot(t2, dif3, color="black", linewidth=1.3, label="", alpha=0.95)
    ax3.set_ylim([lim2,lim])
    plt.legend(fontsize=legsize)



plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\veryarid_lowland.png", dpi = 400)



# ##################################################################################################################################################
title = "Grassland Kimanjo, Laikipia County (1,791 m) vs. Wild Life Conservancy Ol Pejeta, Laikipia County (1,776 m)"
subtitle = "[9x9 km], Semi-Arid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 340 mm"
#"[9x9 km], Sub-Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"
#deifne ax lim y-ax LST
lim = 60
lim2 = -3
legsize = 6.5

label1 = "Grassland"
label2 = "Conservancy"
#label3 = "#16 Ol Pejeta (1,776 m)"


#
aoi1 = MAX_Kimanjo_mean
aoi1_dev = MAX_Kimanjo_deviation

aoi2 = MAX_OlPejeta_mean
aoi2_dev = MAX_OlPejeta_deviation

dif1 = aoi1[0] - aoi2[0]
# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Klimanjo_mean
aoi1_diu_dev = diurnal_Klimanjo_deviation

aoi2_diu = diurnal_OlPejeta_mean
aoi2_diu_dev = diurnal_OlPejeta_deviation
dif2 = aoi1_diu[0]-aoi2_diu[0]
dif2 = np.concatenate((dif2[84:], dif2[:84]), axis=0, out=None)

#jun
j_aoi1_diu = Jdiurnal_Klimanjo_mean
j_aoi1_diu_dev = Jdiurnal_Klimanjo_deviation

j_aoi2_diu = Jdiurnal_OlPejeta_mean
j_aoi2_diu_dev = Jdiurnal_OlPejeta_deviation
dif3 = j_aoi1_diu[0]-j_aoi2_diu[0]
dif3 = np.concatenate((dif3[84:], dif3[:84]), axis=0, out=None)


color1 = "darkred"
color11 = "darkred"
color2 = "darkgreen"
color22 = "saddlebrown"

#plot
#plot
# plot AOI all data
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.set_size_inches(6.3, 2.5)
    fig.subplots_adjust(left=0.113, right=0.98, top=0.755, bottom=0.13)
    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    t = list(range(0, 36))
    tx = list(range(8, 35, 9))
    ax1.set_xticks(tx)
    #ax1.set_xticklabels(["                        J F M A M J J A S O N D"], fontsize=9)
    ax1.set_xticklabels(["     J F M   A M J     ", "","     J A S   O N D     "], fontsize=8)
    #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax1.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    ax1.set_ylabel('SEVIRI Land Surface Temperature (°C)', color="black", fontsize=8.5)
    ax1.set_xlabel('month', color="black", fontsize=8)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax1.grid(which='both')
    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax1.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax1.set_facecolor("white")
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    # Create offset transform by 5 points in x direction
    dx = 6/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax1.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax1.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    #ax1
    ax1.set_title("$Annual_{mean}LST_{max}$ var",fontsize=8)

    # 9x9 sd AOI1
    y1_min = list(map(sub, aoi1[0], aoi1_dev[0]))
    y1_max = list(map(add, aoi1[0], aoi1_dev[0]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color1, alpha=0.2, label="SD")

    # 9x9 sd AOI2
    y1_min = list(map(sub, aoi2[0], aoi2_dev[1]))
    y1_max = list(map(add, aoi2[0], aoi2_dev[1]))
    plt.subplot(ax1).fill_between(t, y1_min, y1_max, color=color2, alpha=0.2, label="SD")
    # AOI 1
    #y1_min = list(map(sub, aoi1[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi1[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi1.shape[0]):
     #   singlepix = ax1.plot(t, aoi1[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax1.plot(t, aoi1[1], color = color11, linewidth=0.2, label="3x3 km central pixel")
    aoi9x91 = ax1.plot(t, aoi1[0], color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)
    # AOI 2
    #y1_min = list(map(sub, aoi2[1], aoi_dev[1]))
    #y1_max = list(map(add, aoi2[1], aoi_dev[1]))
    #plt.subplot(ax1).fill_between(t, y1_min, y1_max, color="firebrick", alpha=0.3, label="std dev central pixel")
    #for sample in range(2, aoi2.shape[0]):
    #    singlepix = ax1.plot(t, aoi2[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax1.plot(t, aoi2[1], color = color22, linewidth=0.2, label="3x3 km central pixel")
    aoi9x92 = ax1.plot(t, aoi2[0], color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif1 = ax1.plot(t, dif1, color="black", linewidth=1.3, label="dif", alpha=0.95)

    ax1.set_ylim([lim2,lim])
#plt.legend(fontsize=legsize)
    #####################################################################################################
    #ax2
    t2 = list(range(0, 96))
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax2.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax2.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax2.set_yticks(major_ticks)
    ax2.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax2.grid(which='both')
    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax2.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax2.set_facecolor("white")
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax2.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    #ax2.set_ylabel('$Land Surface Temperature_{mean}$ (°C)', color="black", fontsize=22)
    # Create offset transform by 5 points in x direction
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax2.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax2.set_title("February diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI1
    # reorder time issue AOI1
    all9x9 = aoi1_diu[0]
    all9x9_dev = aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = aoi2_diu[0]
    all9x9_dev = aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="")

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax2).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="")

    #for sample in range(0, 8):
    #    singlepix = ax2.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre1 = ax2.plot(t2, reordered_centre, color = color11, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, label="9x9 km", alpha=0.95)


    #for sample in range(0, 8):
    #    singlepix2 = ax2.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2, label="3x3 km" if sample == 2 else "")
    #centre2 = ax2.plot(t2, reordered_centre, color = color22, linewidth=0.2, label="3x3 km (central pixel)")
    aoi9x9 = ax2.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, label="9x9 km", alpha=0.95)
    dif2 = ax2.plot(t2, dif2, color="black", linewidth=1.3, label="dif", alpha=0.95)
    ax2.set_ylim([lim2,lim])
    #plt.legend(fontsize=16)
    #############################
    #ax3
    t2 = list(range(0, 96))
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 95])
    ax3.set_xticklabels(["", "4", "8", "12", "16", "20", ""], fontsize=8)
    #ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax3.set_yticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=10)
    major_ticks = np.arange(0, 71, 10)
    minor_ticks = np.arange(0, 71, 5)
    ax3.set_yticks(major_ticks)
    ax3.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax3.grid(which='both')
    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.7, color="darkgrey", linewidth=0.2)
    ax3.grid(which='major', alpha=0.7, color="black", linewidth=0.3)
    ax3.set_facecolor("white")
    ax3.spines['bottom'].set_color('black')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(0.5)
    ax3.spines['bottom'].set_linewidth(0.5)
    ax3.spines['top'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['top'].set_linewidth(0.5)
    ax3.spines['right'].set_linewidth(0.5)
    # apply offset transform to all x ticklabels.
    dx = 5/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax3.set_xlabel('local time (UTC+03:00)', color="black", fontsize=8)
    dx = 0/72.; dy = 5/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax3.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ###################################################################################################################
    ax3.set_title("June diurnal $LST_{mean}$ var",fontsize=8)
    # reorder time issue AOI 1
    all9x9 = j_aoi1_diu[0]
    all9x9_dev = j_aoi1_diu_dev[0]
    reordered9x9_1 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_1 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    all9x9 = j_aoi2_diu[0]
    all9x9_dev = j_aoi2_diu_dev[0]
    reordered9x9_2 = np.concatenate((all9x9[84:], all9x9[:84]), axis=0, out=None)
    reordered9x9dev_2 = np.concatenate((all9x9_dev[84:], all9x9_dev[:84]), axis=0, out=None)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_1, reordered9x9dev_1))
    y1_max = list(map(add, reordered9x9_1, reordered9x9dev_1))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color1, alpha=0.2, label="range of SD "+label1)

    # 9x9 sd
    y1_min = list(map(sub, reordered9x9_2, reordered9x9dev_2))
    y1_max = list(map(add, reordered9x9_2, reordered9x9dev_2))
    plt.subplot(ax3).fill_between(t2, y1_min, y1_max, color=color2, alpha=0.2, label="range of SD "+label2)


    #for sample in range(0, 8):
    #    singlepix1 = ax3.plot(t2, rest_of_3x3[sample], color=color11, linewidth=0.2)
    #centre1 = ax3.plot(t2, reordered_centre, color = color11, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_1, color=color1, linewidth=1.3, alpha=0.95, label="")
    #for sample in range(0, 8):
    #    singlepix2 = ax3.plot(t2, rest_of_3x3[sample], color=color22, linewidth=0.2)
    #centre2 = ax3.plot(t2, reordered_centre, color = color22, linewidth=0.2)
    aoi9x9 = ax3.plot(t2, reordered9x9_2, color=color2, linewidth=1.3, alpha=0.95, label= "")
    dif3 = ax3.plot(t2, dif3, color="black", linewidth=1.3, label="", alpha=0.95)
    ax3.set_ylim([lim2,lim])
    plt.legend(fontsize=legsize)


plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\plateau.png", dpi = 400)






















































# ##################################################################################################################################################
title = "Livestock-only Areas Isiolo: Merti vs. Duma"
subtitle = "[9x9 km], Very Arid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 150 mm"
#"[9x9 km], Sub-Humid Zone, 2012 - 2014 Seasonal $Precipitation_{mean}$: 670 - 680 mm"
#deifne ax lim y-ax LST
lim = 70
lim2 = -2
legsize = 6

label1 = "#2 Merti (363 m)"
label2 = "#1 Duma (300 m)"
#label3 = "#16 Ol Pejeta (1,776 m)"

aoi1 = MAX_Merti_mean
#aoi2 = MAX_Lengalilla_mean
aoi2 = MAX_Duma_mean

# reorder DIURNAL ARRAYS
#feb
aoi1_diu = diurnal_Merti_mean
#aoi2_diu = diurnal_Lengalilla_mean
aoi2_diu = diurnal_Duma_mean
#jun
j_aoi1_diu = Jdiurnal_Merti_mean
#j_aoi2_diu = Jdiurnal_Lengalilla_mean
j_aoi2_diu = Jdiurnal_Duma_mean

color1 = "red"
color11 = "darkred"
color2 = "blue"
color22 = "navy"


plt.savefig(r"C:\Users\icx\Desktop\FINAL PLOTS\Aois_comparison\veryarid_lowland_low2.png", dpi = 400)


































































































































###########################
###########################
############################
#plot mean Lines in one graph, 2 figures, HIGHLAND 3 AOIS, LOWLAND 3 AOIS(published HP!)
for areas in range(1, 2):
    # define comparisons of data plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.045, right=0.98, top=0.85, bottom=0.2)
    fig.suptitle('SEVIRI Mean Annual Land Surface Temperature for Different Land Use Types in Kenya (2012-2014)',
                 fontsize=21)
    width_in_inches = 80
    height_in_inches = 40
    dots_per_inch = 30
    t = list(range(0, 36))
    ax1.set_xticks([0, 2.5,  5.5,  8.5, 11.5, 14.5, 17.5, 20.5, 23.5, 26.5, 29.5, 32.5])
    ax1.set_xticklabels(["          Jan", "           Feb", "            Mar", "           Apr", "           May",
                        "            Jun", "           Jul", "           Aug", "           Sep", "           Oct",
                        "           Nov", "            Dec"], fontsize=14)
    ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    ax1.set_yticklabels(["0", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70"], fontsize=14)
###################################################################################################################
    ax1.set_title(
        "Laikipia Plateau " + "(Extent = METEOSAT 2nd SEVIRI " + extention[areas] + ")",
        fontsize=14)
    nyeri = ax1.plot(t, MAX_s_lineplots_mean[areas], color = scolor1, linewidth=2.5)
    ax1.errorbar(t, MAX_s_lineplots_mean[areas],  yerr=MAX_s_deviation[areas],
                linewidth=2, color=scolor1, alpha=0.4, capsize=6, capthick=2)

    ol_jogi = ax1.plot(t, MAX_n_lineplots_mean[areas+6], color = ncolor3, linewidth=2.5)
    ax1.errorbar(t, MAX_n_lineplots_mean[areas+6],  yerr=MAX_n_deviation[areas+6],
                linewidth=2, color=ncolor3, alpha=0.4, capsize=6, capthick=2)

    mukogodo = ax1.plot(t, MAX_n_lineplots_mean[areas + 9], color=ncolor4, linewidth=2.5)
    ax1.errorbar(t, MAX_n_lineplots_mean[areas + 9],  yerr=MAX_n_deviation[areas+9],
                linewidth=2, color=ncolor4, alpha=0.4, capsize=6, capthick=2)
#####################################################################################################
    fig.text(0.035, 0.1245, n_str_list_of_aoi[areas + 9],
             backgroundcolor=ncolor4,
             color='black', weight='roman', size='large')
    fig.text(0.035, 0.0845, n_str_list_of_aoi[areas + 6],
             backgroundcolor=ncolor3,
             color='black', weight='roman', size='large')
    fig.text(0.035, 0.0445, s_str_list_of_aoi[areas],
             backgroundcolor=scolor1,
             color='black', weight='roman', size='large')
    #fig.text(0.70, 0.1645, s_str_list_of_aoi[areas-3],
             #backgroundcolor=scolor2,
             #color='black', weight='roman', size='large')
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    #ax1.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.7)
    #ax1.set_xlabel('per 10-day-period (~' + number_of_values[areas] + ' daily LST max values per period)', fontsize=15)
    ax1.set_ylabel("SEVIRI mean daily LST max [°C]", fontsize=15)
    #custom_ylim = (0, 60)
    # Setting the values for all axes.
    #plt.setp(ax1, ylim=custom_ylim)
############################
    t2 = list(range(0, 36))
    ax2.set_xticks([0, 2.5,  5.5,  8.5, 11.5, 14.5, 17.5, 20.5, 23.5, 26.5, 29.5, 32.5])
    ax2.set_xticklabels(["          Jan", "           Feb", "            Mar", "           Apr", "           May",
                        "            Jun", "           Jul", "           Aug", "           Sep", "           Oct",
                        "           Nov", "            Dec"], fontsize=14)
    ax2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    ax2.set_yticklabels(["0", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70"], fontsize=14)
###################################################################################################################
    ax2.set_title(
        "Lowland " + "(Extent = METEOSAT 2nd SEVIRI " + extention[areas] + ")",
        fontsize=14)
    swamps = ax2.plot(t2, MAX_n_lineplots_mean[areas+12], color = ncolor5, linewidth=3)
    meruparkN = ax2.plot(t2, MAX_s_lineplots_mean[areas+6], color = scolor3, linewidth=3)
    rangelandSwamps = ax2.plot(t2, MAX_n_lineplots_mean[areas + 15], color=ncolor6, linewidth=3)
#####################################################################################################
    fig.text(0.55, 0.1245, n_str_list_of_aoi[areas + 15],
             backgroundcolor=ncolor6,
             color='black', weight='roman', size='large')
    fig.text(0.55, 0.0845, s_str_list_of_aoi[areas+6],
             backgroundcolor=scolor3,
             color='black', weight='roman', size='large')
    fig.text(0.55, 0.0445, n_str_list_of_aoi[areas+12],
             backgroundcolor=ncolor5,
             color='black', weight='roman', size='large')
    #fig.text(0.70, 0.1645, s_str_list_of_aoi[areas-3],
             #backgroundcolor=scolor2,
             #color='black', weight='roman', size='large')
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax2.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.7)
    ax2.set_ylabel("SEVIRI mean daily LST max [°C]", fontsize=15)
    #ax1.set_xlabel('per 10-day-period (~' + number_of_values[areas] + ' daily LST max values per period)', fontsize=15)
    #ax1.set_ylabel("SEVIRI mean daily LST max [°C]", fontsize=15)
    custom_ylim = (0, 60)
    # Setting the values for all axes.
    plt.setp(ax2, ylim=custom_ylim)
    plt.show()



