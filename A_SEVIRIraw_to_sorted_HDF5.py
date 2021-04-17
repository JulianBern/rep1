# author: Bernegger Julian

# This code imports SEVIRI LST HDF5 files from defined directories, that contain selected datasets from the same month of the year (e.g. MAY 2012/2013/2014).
# After that, the data is saved into a new HDF5 file, one for North Hemisphere one for South, containing 36 groups for each 10-day-period of the year.
# Temporal resolution = 15min = 96 datasets per day, 2012-2014 contain ~2880 datasets per 10-day-periods

import os
import h5py
import numpy as np
import geopandas as gpd
import gdal
from osgeo import gdal_array
from osgeo import osr
import matplotlib.pylab as plt

# SOUTH AFRICA
# create list of substrings to SORT all possible 10 day periods (3x12=36 classes)
subs_jan1 = list(range(20120101,20120111)) + list(range(20130101,20130111)) + list(range(20140101,20140111))
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

###################################################################################################################
# DECADE GROUP ___ CREATING HDF5 FILE --- SAVE SEVIRI DATA TO IT --- SOUTH AFRICA PART OF STUDY AREA
################################################################################################################
path = r"F:\MA_own_hdf5"# define working directory for my HDF5 file
os.chdir(path)
q = h5py.File('LSTmax_SAfr_2012_2014.hdf5','w')# create hdf5 file

t0 = q.create_group(strings[0])# create groups in hdf5 file
t0.name
t1 = q.create_group(strings[1])
t1.name
t2 = q.create_group(strings[2])
t2.name
t3 = q.create_group(strings[3])
t3.name
t4 = q.create_group(strings[4])
t4.name
t5 = q.create_group(strings[5])
t5.name
t6 = q.create_group(strings[6])
t6.name
t7 = q.create_group(strings[7])
t7.name
t8 = q.create_group(strings[8])
t8.name
t9 = q.create_group(strings[9])
t9.name
t10 = q.create_group(strings[10])
t10.name
t11 = q.create_group(strings[11])
t11.name
t12 = q.create_group(strings[12])
t12.name
t13 = q.create_group(strings[13])
t13.name
t14 = q.create_group(strings[14])
t14.name
t15 = q.create_group(strings[15])
t15.name
t16 = q.create_group(strings[16])
t16.name
t17 = q.create_group(strings[17])
t17.name
t18 = q.create_group(strings[18])
t18.name
t19 = q.create_group(strings[19])
t19.name
t20 = q.create_group(strings[20])
t20.name
t21 = q.create_group(strings[21])
t21.name
t22 = q.create_group(strings[22])
t22.name
t23 = q.create_group(strings[23])
t23.name
t24 = q.create_group(strings[24])
t24.name
t25 = q.create_group(strings[25])
t25.name
t26 = q.create_group(strings[26])
t26.name
t27 = q.create_group(strings[27])
t27.name
t28 = q.create_group(strings[28])
t28.name
t29 = q.create_group(strings[29])
t29.name
t30 = q.create_group(strings[30])
t30.name
t31 = q.create_group(strings[31])
t31.name
t32 = q.create_group(strings[32])
t32.name
t33 = q.create_group(strings[33])
t33.name
t34 = q.create_group(strings[34])
t34.name
t35 = q.create_group(strings[35])
t35.name

### SOUTH AFRICA ###
### creation of datasets in hdf5 file
for fol in range(0, 12):# loop over every data folder of monthly sorted data
    path = ("F:\MA_data_LST_SAfr_months")  # define the directory of the specific month to collect
    os.chdir(path)
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
    path = os.path.join(path, month[fol])  # define the directory of the specific month to collect
    os.chdir(path)
    folder = os.listdir(path)
    for sample in range(3*(fol+1)-3, 3*(fol+1)):# number of groups, 10 day periods (36)
        subs = stringcall[sample]
        for sie in range(0, len(subs)):# number of filenames is 30, if study period is 3 years (10 per year)
            select = [i for i in folder if "_" + str(subs[sie]) in i]
            print(select)
            for sample1 in range(0, len(select)):  # import files with specific NAME as hdf and cut it to SOUTH AFRICA REGION OF STUDY AREA and create np.array store to HDF5
                filename = select[sample1]
                with h5py.File(filename, 'r') as hdf:
                    ls4 = list(hdf.keys())
                    print('List of dataset in this file: \n', ls4)
                    lst = hdf.get('LST')
                    lst = lst[0:32, 920:1017]# define study area extent
                    lst_data = np.array(lst)
                    if sample == 0:
                        dset0 = t0.create_dataset(filename, data=lst_data, )
                    if sample == 1:
                        dset1 = t1.create_dataset(filename, data=lst_data, )
                    if sample == 2:
                        dset2 = t2.create_dataset(filename, data=lst_data, )
                    if sample == 3:
                        dset3 = t3.create_dataset(filename, data=lst_data, )
                    if sample == 4:
                        dset4 = t4.create_dataset(filename, data=lst_data, )
                    if sample == 5:
                        dset5 = t5.create_dataset(filename, data=lst_data, )
                    if sample == 6:
                        dset6 = t6.create_dataset(filename, data=lst_data, )
                    if sample == 7:
                        dset7 = t7.create_dataset(filename, data=lst_data, )
                    if sample == 8:
                        dset8 = t8.create_dataset(filename, data=lst_data, )
                    if sample == 9:
                        dset9 = t9.create_dataset(filename, data=lst_data, )
                    if sample == 10:
                        dset10 = t10.create_dataset(filename, data=lst_data, )
                    if sample == 11:
                        dset11 = t11.create_dataset(filename, data=lst_data, )
                    if sample == 12:
                        dset12 = t12.create_dataset(filename, data=lst_data, )
                    if sample == 13:
                        dset13 = t13.create_dataset(filename, data=lst_data, )
                    if sample == 14:
                        dset14 = t14.create_dataset(filename, data=lst_data, )
                    if sample == 15:
                        dset15 = t15.create_dataset(filename, data=lst_data, )
                    if sample == 16:
                        dset16 = t16.create_dataset(filename, data=lst_data, )
                    if sample == 17:
                        dset17 = t17.create_dataset(filename, data=lst_data, )
                    if sample == 18:
                        dset18 = t18.create_dataset(filename, data=lst_data, )
                    if sample == 19:
                        dset19 = t19.create_dataset(filename, data=lst_data, )
                    if sample == 20:
                        dset20 = t20.create_dataset(filename, data=lst_data, )
                    if sample == 21:
                        dset21 = t21.create_dataset(filename, data=lst_data, )
                    if sample == 22:
                        dset22 = t22.create_dataset(filename, data=lst_data, )
                    if sample == 23:
                        dset23 = t23.create_dataset(filename, data=lst_data, )
                    if sample == 24:
                        dset24 = t24.create_dataset(filename, data=lst_data, )
                    if sample == 25:
                        dset25 = t25.create_dataset(filename, data=lst_data, )
                    if sample == 26:
                        dset26 = t26.create_dataset(filename, data=lst_data, )
                    if sample == 27:
                        dset27 = t27.create_dataset(filename, data=lst_data, )
                    if sample == 28:
                        dset28 = t28.create_dataset(filename, data=lst_data, )
                    if sample == 29:
                        dset29 = t29.create_dataset(filename, data=lst_data, )
                    if sample == 30:
                        dset30 = t30.create_dataset(filename, data=lst_data, )
                    if sample == 31:
                        dset31 = t31.create_dataset(filename, data=lst_data, )
                    if sample == 32:
                        dset32 = t32.create_dataset(filename, data=lst_data, )
                    if sample == 33:
                        dset33 = t33.create_dataset(filename, data=lst_data, )
                    if sample == 34:
                        dset34 = t34.create_dataset(filename, data=lst_data, )
                    if sample == 35:
                        dset35 = t35.create_dataset(filename, data=lst_data, )

                    print('shape of dataset1: \n', lst_data.shape)
                    lsX = list(q.keys())
                    print('List of dataset in this file: \n', lsX)
q.close()
###################################################################################################################
# LIST GROUPS, MEMBERS AND DATASETS OF MY HDF5 FILE
path = r"F:\MA_own_hdf5"
os.chdir(path)
q = h5py.File('LSTmax_SAfr_2012_2014.hdf5', 'r')
base_items = list(q.items())
print(base_items)

q.close()



################################################################################################
# NORTH AFRICA

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

###################################################################################################################
# DECADE GROUP ___ CREATING HDF5 FILE --- SAVE SEVIRI DATA TO IT --- SOUTH AFRICA PART OF STUDY AREA
################################################################################################################
path = r"F:\MA_own_hdf5"# define working directory for my HDF5 file
os.chdir(path)
q = h5py.File('LSTmax_NAfr_2012_2014.hdf5','w')# create hdf5 file

t0 = q.create_group(strings[0])# create groups in hdf5 file
t0.name
t1 = q.create_group(strings[1])
t1.name
t2 = q.create_group(strings[2])
t2.name
t3 = q.create_group(strings[3])
t3.name
t4 = q.create_group(strings[4])
t4.name
t5 = q.create_group(strings[5])
t5.name
t6 = q.create_group(strings[6])
t6.name
t7 = q.create_group(strings[7])
t7.name
t8 = q.create_group(strings[8])
t8.name
t9 = q.create_group(strings[9])
t9.name
t10 = q.create_group(strings[10])
t10.name
t11 = q.create_group(strings[11])
t11.name
t12 = q.create_group(strings[12])
t12.name
t13 = q.create_group(strings[13])
t13.name
t14 = q.create_group(strings[14])
t14.name
t15 = q.create_group(strings[15])
t15.name
t16 = q.create_group(strings[16])
t16.name
t17 = q.create_group(strings[17])
t17.name
t18 = q.create_group(strings[18])
t18.name
t19 = q.create_group(strings[19])
t19.name
t20 = q.create_group(strings[20])
t20.name
t21 = q.create_group(strings[21])
t21.name
t22 = q.create_group(strings[22])
t22.name
t23 = q.create_group(strings[23])
t23.name
t24 = q.create_group(strings[24])
t24.name
t25 = q.create_group(strings[25])
t25.name
t26 = q.create_group(strings[26])
t26.name
t27 = q.create_group(strings[27])
t27.name
t28 = q.create_group(strings[28])
t28.name
t29 = q.create_group(strings[29])
t29.name
t30 = q.create_group(strings[30])
t30.name
t31 = q.create_group(strings[31])
t31.name
t32 = q.create_group(strings[32])
t32.name
t33 = q.create_group(strings[33])
t33.name
t34 = q.create_group(strings[34])
t34.name
t35 = q.create_group(strings[35])
t35.name

### NORTH AFRICA ###
### creation of datasets in hdf5 file
for fol in range(0, 12):# loop over every data folder of monthly sorted data
    path = ("F:\MA_data_LST_NAfr_months")  # define the directory of the specific month to collect
    os.chdir(path)
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
    path = os.path.join(path, month[fol])  # define the directory of the specific month to collect
    os.chdir(path)
    folder = os.listdir(path)
    for sample in range(3*(fol+1)-3, 3*(fol+1)):# number of groups, 10 day periods (36)
        subs = stringcall[sample]
        for sie in range(0, 1):# number of filenames is 30, if study period is 3 years (10 per year)
            select = [i for i in folder if "_" + str(subs[sie]) in i]
            print(select)
            for sample1 in range(0, len(select)):  # import files with specific NAME as hdf and cut it to SOUTH AFRICA REGION OF STUDY AREA and create np.array store to HDF5
                filename = select[sample1]
                with h5py.File(filename, 'r') as hdf:
                    ls4 = list(hdf.keys())
                    print('List of dataset in this file: \n', ls4)
                    lst = hdf.get('LST')
                    lst = lst[1073:, 1819:1916]# define study area extent
                    lst_data = np.array(lst)
                    if sample == 0:
                        dset0 = t0.create_dataset(filename, data=lst_data, )
                    if sample == 1:
                        dset1 = t1.create_dataset(filename, data=lst_data, )
                    if sample == 2:
                        dset2 = t2.create_dataset(filename, data=lst_data, )
                    if sample == 3:
                        dset3 = t3.create_dataset(filename, data=lst_data, )
                    if sample == 4:
                        dset4 = t4.create_dataset(filename, data=lst_data, )
                    if sample == 5:
                        dset5 = t5.create_dataset(filename, data=lst_data, )
                    if sample == 6:
                        dset6 = t6.create_dataset(filename, data=lst_data, )
                    if sample == 7:
                        dset7 = t7.create_dataset(filename, data=lst_data, )
                    if sample == 8:
                        dset8 = t8.create_dataset(filename, data=lst_data, )
                    if sample == 9:
                        dset9 = t9.create_dataset(filename, data=lst_data, )
                    if sample == 10:
                        dset10 = t10.create_dataset(filename, data=lst_data, )
                    if sample == 11:
                        dset11 = t11.create_dataset(filename, data=lst_data, )
                    if sample == 12:
                        dset12 = t12.create_dataset(filename, data=lst_data, )
                    if sample == 13:
                        dset13 = t13.create_dataset(filename, data=lst_data, )
                    if sample == 14:
                        dset14 = t14.create_dataset(filename, data=lst_data, )
                    if sample == 15:
                        dset15 = t15.create_dataset(filename, data=lst_data, )
                    if sample == 16:
                        dset16 = t16.create_dataset(filename, data=lst_data, )
                    if sample == 17:
                        dset17 = t17.create_dataset(filename, data=lst_data, )
                    if sample == 18:
                        dset18 = t18.create_dataset(filename, data=lst_data, )
                    if sample == 19:
                        dset19 = t19.create_dataset(filename, data=lst_data, )
                    if sample == 20:
                        dset20 = t20.create_dataset(filename, data=lst_data, )
                    if sample == 21:
                        dset21 = t21.create_dataset(filename, data=lst_data, )
                    if sample == 22:
                        dset22 = t22.create_dataset(filename, data=lst_data, )
                    if sample == 23:
                        dset23 = t23.create_dataset(filename, data=lst_data, )
                    if sample == 24:
                        dset24 = t24.create_dataset(filename, data=lst_data, )
                    if sample == 25:
                        dset25 = t25.create_dataset(filename, data=lst_data, )
                    if sample == 26:
                        dset26 = t26.create_dataset(filename, data=lst_data, )
                    if sample == 27:
                        dset27 = t27.create_dataset(filename, data=lst_data, )
                    if sample == 28:
                        dset28 = t28.create_dataset(filename, data=lst_data, )
                    if sample == 29:
                        dset29 = t29.create_dataset(filename, data=lst_data, )
                    if sample == 30:
                        dset30 = t30.create_dataset(filename, data=lst_data, )
                    if sample == 31:
                        dset31 = t31.create_dataset(filename, data=lst_data, )
                    if sample == 32:
                        dset32 = t32.create_dataset(filename, data=lst_data, )
                    if sample == 33:
                        dset33 = t33.create_dataset(filename, data=lst_data, )
                    if sample == 34:
                        dset34 = t34.create_dataset(filename, data=lst_data, )
                    if sample == 35:
                        dset35 = t35.create_dataset(filename, data=lst_data, )

                    print('shape of dataset1: \n', lst_data.shape)
                    lsX = list(q.keys())
                    print('List of dataset in this file: \n', lsX)
q.close()
###################################################################################################################
# LIST GROUPS, MEMBERS AND DATASETS OF MY HDF5 FILE
path = r"F:\MA_own_hdf5"
os.chdir(path)
q = h5py.File('LSTmax_NAfr_2012_2014.hdf5', 'r')
base_items = list(q.items())
print(base_items)

q.close()