"""
Data prep mod
UMC server
"""
#%% Packages
# data
import netCDF4 as nc
import numpy as np

import pandas as pd

# models 
import tensorflow as tf
print(tf.__version__)
#tf.compat.v1.disable_eager_execution()
#tf.disable_eager_execution()
tf.compat.v1.enable_eager_execution()

# must be True
print(tf.executing_eagerly())

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

import os

import helper_functions as hf
from skimage.metrics import structural_similarity as ssim

import importlib as imp

#%% Loading data
print('Load data')
U_WRF_3km = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_in/U_WRF_3km.npy')
V_WRF_3km = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_in/V_WRF_3km.npy')
T_WRF_3km = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_in/T_WRF_3km.npy')
U_WRF_3km_test = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_in/U_WRF_3km_test.npy')
V_WRF_9km_test = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_in/V_WRF_3km_test.npy') 
T_WRF_3km_test = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_in/T_WRF_3km_test.npy')
print('Loading data done')
#%% MR_HR data

# Stacking wind data
Wind_WRF_3km = np.squeeze(np.stack([U_WRF_3km, V_WRF_3km], axis = -1))
Wind_WRF_3km_test = np.squeeze(np.stack([U_WRF_3km_test, V_WRF_9km_test], axis = -1))
    
print(Wind_WRF_3km.shape, Wind_WRF_3km_test.shape)

# getting the generated MR data
# temperature
T_mr = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_out/temperature/LR_MR/dataSR.npy')      
T_mr_test = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_out/temperature/LR_MR/test/dataSR.npy')

# wind
Wind_mr = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_out/wind/LR_MR/dataSR.npy')      
Wind_mr_test = np.load('/hpc/shared/uu_imau_ocean/kovacs/SR_new/data_out/wind/LR_MR/test/dataSR.npy')

# Train
hf.generate_TFRecords(filename = '/hpc/shared/uu_imau_ocean/kovacs/SR_new/TFRecords/T_mr_hr.tfrecord',
                   data_HR = T_WRF_3km,
                   data_LR = T_mr,
                   mode = 'train')

# Test
hf.generate_TFRecords(filename = '/hpc/shared/uu_imau_ocean/kovacs/SR_new/TFRecords/T_mr_hr_test.tfrecord',
                   data_HR = None,
                   data_LR = T_mr_test,
                   mode = 'test')

# Train
hf.generate_TFRecords(filename = '/hpc/shared/uu_imau_ocean/kovacs/SR_new/TFRecords/Wind_mr_hr.tfrecord',
                   data_HR = Wind_WRF_3km,
                   data_LR = Wind_mr,
                   mode = 'train')

# Test
hf.generate_TFRecords(filename = '/hpc/shared/uu_imau_ocean/kovacs/SR_new/TFRecords/Wind_mr_hr_test.tfrecord',
                   data_HR = None,
                   data_LR = Wind_mr_test,
                   mode = 'test')
