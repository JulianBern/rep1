# author: Bernegger Julian

# this code reads the pre produced hdf5 file of north / south part of the study area and exports one np.array(30, x, x) with 1 LSTmax dataset per day of the specific decade

import os
import h5py
import numpy as np
import geopandas as gpd
import gdal
from osgeo import gdal_array
from osgeo import osr
import matplotlib.pylab as plt

subs_jan1 = list(range(20120101,20120111)) + list(range(20130101,20130111)) + list(range(20140101,20140111))# create list of substrings to SORT all possible 10 day periods (3x12=36 classes)
subs_jan2 = list(range(20120111,20120121)) + list(range(20130111,20130121)) + list(range(20140111,20140121))
subs_jan3 = list(range(20120121,20120132)) + list(range(20130121,20130132)) + list(range(20140121,20140132))
subs_feb1 = list(range(20120201,20120211)) + list(range(20130201,20130211)) + list(range(20140201,20140211))
subs_feb2 = list(range(20120211,20120221)) + list(range(20130211,20130221)) + list(range(20140211,20140221))
subs_feb3 = list(range(20120221,20120232)) + list(range(20130221,20130232)) + list(range(20140221,20140232))
subs_mar1 = list(range(20120301,20120311)) + list(range(20130301,20130311)) + list(range(20140301,20140311))
subs_mar2 = list(range(20120311,20120321)) + list(range(20130311,20130321)) + list(range(20140311,20140321))
subs_mar3 = list(range(20120321,20120332)) + list(range(20130321,20130332)) + list(range(20140321,20140332))
subs_apr1 = list(range(20120401,20120411)) + list(range(20130401,20130411)) + list(range(20140401,20140411))
subs_apr2 = list(range(20120411,20120421)) + list(range(20130411,20130421)) + list(range(20140411,20140421))
subs_apr3 = list(range(20120421,20120432)) + list(range(20130421,20130432)) + list(range(20140421,20140432))
subs_may1 = list(range(20120501,20120511)) + list(range(20130501,20130511)) + list(range(20140501,20140511))
subs_may2 = list(range(20120511,20120521)) + list(range(20130511,20130521)) + list(range(20140511,20140521))
subs_may3 = list(range(20120521,20120532)) + list(range(20130521,20130532)) + list(range(20140521,20140532))
subs_jun1 = list(range(20120601,20120611)) + list(range(20130601,20130611)) + list(range(20140601,20140611))
subs_jun2 = list(range(20120611,20120621)) + list(range(20130611,20130621)) + list(range(20140611,20140621))
subs_jun3 = list(range(20120621,20120632)) + list(range(20130621,20130632)) + list(range(20140621,20140632))
subs_jul1 = list(range(20120701,20120711)) + list(range(20130701,20130711)) + list(range(20140701,20140711))
subs_jul2 = list(range(20120711,20120721)) + list(range(20130711,20130721)) + list(range(20140711,20140721))
subs_jul3 = list(range(20120721,20120732)) + list(range(20130721,20130732)) + list(range(20140721,20140732))
subs_aug1 = list(range(20120801,20120811)) + list(range(20130801,20130811)) + list(range(20140801,20140811))
subs_aug2 = list(range(20120811,20120821)) + list(range(20130811,20130821)) + list(range(20140811,20140821))
subs_aug3 = list(range(20120821,20120832)) + list(range(20130821,20130832)) + list(range(20140821,20140832))
subs_sep1 = list(range(20120901,20120911)) + list(range(20130901,20130911)) + list(range(20140901,20140911))
subs_sep2 = list(range(20120911,20120921)) + list(range(20130911,20130921)) + list(range(20140911,20140921))
subs_sep3 = list(range(20120921,20120932)) + list(range(20130921,20130932)) + list(range(20140921,20140932))
subs_oct1 = list(range(20121001,20121011)) + list(range(20131001,20131011)) + list(range(20141001,20141011))
subs_oct2 = list(range(20121011,20121021)) + list(range(20131011,20131021)) + list(range(20141011,20141021))
subs_oct3 = list(range(20121021,20121032)) + list(range(20131021,20131032)) + list(range(20141021,20141032))
subs_nov1 = list(range(20121101,20121111)) + list(range(20131101,20131111)) + list(range(20141101,20141111))
subs_nov2 = list(range(20121111,20121121)) + list(range(20131111,20131121)) + list(range(20141111,20141121))
subs_nov3 = list(range(20121121,20121132)) + list(range(20131121,20131132)) + list(range(20141121,20141132))
subs_dec1 = list(range(20121201,20121211)) + list(range(20131201,20131211)) + list(range(20141201,20141211))
subs_dec2 = list(range(20121211,20121221)) + list(range(20131211,20131221)) + list(range(20141211,20141221))
subs_dec3 = list(range(20121221,20121232)) + list(range(20131221,20131232)) + list(range(20141221,20141232))

strings = ["subs_jan1", "subs_jan2", "subs_jan3", "subs_feb1", "subs_feb2", "subs_feb3", "subs_mar1", "subs_mar2", "subs_mar3", "subs_apr1", "subs_apr2", "subs_apr3", "subs_may1", "subs_may2", "subs_may3", "subs_jun1", "subs_jun2", "subs_jun3", "subs_jul1", "subs_jul2", "subs_jul3", "subs_aug1", "subs_aug2", "subs_aug3", "subs_sep1", "subs_sep2", "subs_sep3", "subs_oct1", "subs_oct2", "subs_oct3", "subs_nov1", "subs_nov2", "subs_nov3", "subs_dec1", "subs_dec2", "subs_dec3"]# define the classes of 10 day period
stringcall = [subs_jan1, subs_jan2, subs_jan3, subs_feb1, subs_feb2, subs_feb3, subs_mar1, subs_mar2, subs_mar3, subs_apr1, subs_apr2, subs_apr3, subs_may1, subs_may2, subs_may3, subs_jun1, subs_jun2, subs_jun3, subs_jul1, subs_jul2, subs_jul3, subs_aug1, subs_aug2, subs_aug3, subs_sep1, subs_sep2, subs_sep3, subs_oct1, subs_oct2, subs_oct3, subs_nov1, subs_nov2, subs_nov3, subs_dec1, subs_dec2, subs_dec3]# define the classes of 10 day period

path = r"F:\Data\MA_own_hdf5"# import the prepared HDF5 file with datasets sorted by 10 day period in year
os.chdir(path)
q = h5py.File('LSTmax_SAfr_2012_2014.hdf5', 'r')
base_items = list(q.items())
print(base_items)



# create list indexes to sort values after equal days in the year
list1 = [0, 10, 20]
list2 = [1, 11, 21]
list3 = [2, 12, 22]
list4 = [3, 13, 23]
list5 = [4, 14, 24]
list6 = [5, 15, 25]
list7 = [6, 16, 26]
list8 = [7, 17, 27]
list9 = [8, 18, 28]
list10 = [9, 19, 29]
#extra for 11 day periods
list11 = [10, 21, 32]
listcall = [list1,
list2,
list3,
list4,
list5,
list6,
list7,
list8,
list9,
list10, list11]



month = ['1jan',
 '2feb',
 '3mar',
 '4apr',
 '5may',
 '6jun',
 '7jul',
 '8aug',
 '91sep',
 '92oct',
 '93nov',
 '94dec']
for fol in range(0, 1):# define month to calculate, and save arrays of daily max datasets to to specific folder
    d0 = np.empty((0, 32, 97))
    d1 = np.empty((0, 32, 97))
    d2 = np.empty((0, 32, 97))
    d3 = np.empty((0, 32, 97))
    d4 = np.empty((0, 32, 97))
    d5 = np.empty((0, 32, 97))
    d6 = np.empty((0, 32, 97))
    d7 = np.empty((0, 32, 97))
    d8 = np.empty((0, 32, 97))
    d9 = np.empty((0, 32, 97))
    d10 = np.empty((0, 32, 97))
    d11 = np.empty((0, 32, 97))
    d12 = np.empty((0, 32, 97))
    d13 = np.empty((0, 32, 97))
    d14 = np.empty((0, 32, 97))
    d15 = np.empty((0, 32, 97))
    d16 = np.empty((0, 32, 97))
    d17 = np.empty((0, 32, 97))
    d18 = np.empty((0, 32, 97))
    d19 = np.empty((0, 32, 97))
    d20 = np.empty((0, 32, 97))
    d21 = np.empty((0, 32, 97))
    d22 = np.empty((0, 32, 97))
    d23 = np.empty((0, 32, 97))
    d24 = np.empty((0, 32, 97))
    d25 = np.empty((0, 32, 97))
    d26 = np.empty((0, 32, 97))
    d27 = np.empty((0, 32, 97))
    d28 = np.empty((0, 32, 97))
    d29 = np.empty((0, 32, 97))
    d30 = np.empty((0, 32, 97))
    all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21,
                     d22, d23, d24, d25, d26, d27, d28, d29, d30]
    max_period1_array = np.empty((0, 32, 97))  # define arrays to append daily max arrays
    max_period2_array = np.empty((0, 32, 97))
    max_period3_array = np.empty((0, 32, 97))
    path = (r"F:\Data\MA_data_LST_SAfr_months")
    os.chdir(path)
    path = os.path.join(path, month[fol])  # define the directory of the specific month to collect
    os.chdir(path)
    folder = os.listdir(path)
    for period in range(0, 2):# loop over 10 day periods
        decade = 3*(fol+1)-3+period
        for day in range(0, 10):  # 10 days of each ten 10 period
            let10 = [stringcall[decade][listcall[day][0]], stringcall[decade][listcall[day][1]], stringcall[decade][listcall[day][2]]]
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 0 and period == 0:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d0 = np.append(all_day_array[day], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                             d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d0[d0 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d0, axis=0)# sort out nan values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 1 and period == 0:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d1 = np.append(all_day_array[day], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                             d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d1[d1 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d1, axis=0)# sort out nan values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 2 and period == 0:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d2 = np.append(all_day_array[day], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                             d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d2[d2 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d2, axis=0)# sort out nan values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 3 and period == 0:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d3 = np.append(all_day_array[day], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d3[d3 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d3, axis=0)# sort out nan values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 4 and period == 0:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d4 = np.append(all_day_array[day], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d4[d4 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d4, axis=0)# sort out nan values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 5 and period == 0:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d5 = np.append(all_day_array[day], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d5[d5 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d5, axis=0)# sort out nan values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 6 and period == 0:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d6 = np.append(all_day_array[day], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d6[d6 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d6, axis=0)# sort out nan values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 7 and period == 0:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d7 = np.append(all_day_array[day], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d7[d7 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d7, axis=0)# sort out nan values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 8 and period == 0:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d8 = np.append(all_day_array[day], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d8[d8 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d8, axis=0)# sort out nan values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
            for data in range(0, 3):# defines the year (2012, 2013 or 2014)
                select = [i for i in folder if "_" + str(let10[data]) in i]# collects the data from the opened hdf5 file
                if day == 9 and period == 0:# if data represents the 10th day of the 1st period of the month, the code will do the following
                    for line in range(0, len(select)):# for all the datasets of that 10-day interval
                        group = q.get(strings[decade])# open the subgroup of the hdf5 file
                        result = np.array(group.get(select[line]))# import the dataset and turn it into a np.array
                        d9 = np.append(all_day_array[day], [result], axis=0)# append all the datasets to the array of day 10 of the month
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                        # refresh the definition of all the arrays, representing days
                    d9[d9 == -8000] = 'nan'  # turn all the -8000 values into 'nan' expressions
                    max_d = np.nanmax(d9, axis=0)# calculate maxLST values and sort out 'nan' values
                    max_period1_array = np.append(max_period1_array, [max_d], axis=0)
                    # append the resulting LSTmax array to the array of the first 10-day period of that month
                    path = (r"F:\results\max_arrays\SAfr")# define the directory to export the LSTmax arrays
                    os.chdir(path)
                    np.save(strings[decade] + "_" + "max", max_period1_array)# export the array, representing LSTmax of the 10-day interval
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 0 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d10 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d10[d10 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d10, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 1 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d11 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d11[d11 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d11, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 2 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d12 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d12[d12 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d12, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 3 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d13 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d13[d13 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d13, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 4 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d14 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d14[d14 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d14, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 5 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d15 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d15[d15 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d15, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 6 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d16 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d16[d16 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d16, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 7 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d17 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d17[d17 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d17, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 8 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d18 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d18[d18 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d18, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 9 and period == 1:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d19 = np.append(all_day_array[day+10], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d19[d19 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d19, axis=0)# sort out nan values
                    max_period2_array = np.append(max_period2_array, [max_d], axis=0)
                    path = (r"F:\results\max_arrays\SAfr")
                    os.chdir(path)
                    np.save(strings[decade] + "_" + "max", max_period2_array)
    for period in range(2, 3):# loop over the 11 day periods (3rd of month)
        decade = 3*(fol+1)-3+period
        for day in range(0, 11):  # 10 days of each ten 10 period
            let10 = [stringcall[decade][listcall[day][0]], stringcall[decade][listcall[day][1]], stringcall[decade][listcall[day][2]]]
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 0 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d20 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                             d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d20[d20 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d20, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 1 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d21 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                             d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d21[d21 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d21, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 2 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d22 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                             d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d22[d22 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d22, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 3 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d23 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d23[d23 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d23, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 4 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d24 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d24[d24 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d24, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 5 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d25 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d25[d25 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d25, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 6 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d26 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d26[d26 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d26, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 7 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d27 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d27[d27 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d27, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 8 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d28 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d28[d28 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d28, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 9 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d29 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d29[d29 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d29, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            for data in range(0, 3):
                select = [i for i in folder if "_" + str(let10[data]) in i]
                if day == 10 and period == 2:
                    for line in range(0, len(select)):
                        group = q.get(strings[decade])
                        result = np.array(group.get(select[line]))
                        d30 = np.append(all_day_array[day+20], [result], axis=0)
                        all_day_array = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                                         d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
                    d30[d30 == -8000] = 'nan'  # or use np.nan
                    max_d = np.nanmax(d30, axis=0)# sort out nan values
                    max_period3_array = np.append(max_period3_array, [max_d], axis=0)
            path = (r"F:\results\max_arrays\SAfr")
            os.chdir(path)
            np.save(strings[decade] + "_" + "max", max_period3_array)


# load results
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




######################################################################################################
   #####south AFRICA#####
d0 = np.empty((0, 77, 97))
d1 = np.empty((0, 32, 97))
d2 = np.empty((0, 32, 97))
d3 = np.empty((0, 32, 97))
d4 = np.empty((0, 32, 97))
d5 = np.empty((0, 32, 97))
d6 = np.empty((0, 32, 97))
d7 = np.empty((0, 32, 97))
d8 = np.empty((0, 32, 97))
d9 = np.empty((0, 32, 97))
d10 = np.empty((0, 32, 97))
d11 = np.empty((0, 32, 97))
d12 = np.empty((0, 32, 97))
d13 = np.empty((0, 32, 97))
d14 = np.empty((0, 32, 97))
d15 = np.empty((0, 32, 97))
d16 = np.empty((0, 32, 97))
d17 = np.empty((0, 32, 97))
d18 = np.empty((0, 32, 97))
d19 = np.empty((0, 32, 97))
d20 = np.empty((0, 32, 97))
d21 = np.empty((0, 32, 97))
d22 = np.empty((0, 32, 97))
d23 = np.empty((0, 32, 97))
d24 = np.empty((0, 32, 97))
d25 = np.empty((0, 32, 97))
d26 = np.empty((0, 32, 97))
d27 = np.empty((0, 32, 97))
d28 = np.empty((0, 32, 97))
d29 = np.empty((0, 32, 97))
d30 = np.empty((0, 32, 97))


    d0 = np.empty((0, 77, 97))
    d1 = np.empty((0, 77, 97))
    d2 = np.empty((0, 77, 97))
    d3 = np.empty((0, 77, 97))
    d4 = np.empty((0, 77, 97))
    d5 = np.empty((0, 77, 97))
    d6 = np.empty((0, 77, 97))
    d7 = np.empty((0, 77, 97))
    d8 = np.empty((0, 77, 97))
    d9 = np.empty((0, 77, 97))
    d10 = np.empty((0, 77, 97))
    d11 = np.empty((0, 77, 97))
    d12 = np.empty((0, 77, 97))
    d13 = np.empty((0, 77, 97))
    d14 = np.empty((0, 77, 97))
    d15 = np.empty((0, 77, 97))
    d16 = np.empty((0, 77, 97))
    d17 = np.empty((0, 77, 97))
    d18 = np.empty((0, 77, 97))
    d19 = np.empty((0, 77, 97))
    d20 = np.empty((0, 77, 97))
    d21 = np.empty((0, 77, 97))
    d22 = np.empty((0, 77, 97))
    d23 = np.empty((0, 77, 97))
    d24 = np.empty((0, 77, 97))
    d25 = np.empty((0, 77, 97))
    d26 = np.empty((0, 77, 97))
    d27 = np.empty((0, 77, 97))
    d28 = np.empty((0, 77, 97))
    d29 = np.empty((0, 77, 97))
    d30 = np.empty((0, 77, 97))