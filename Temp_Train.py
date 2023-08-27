"""
Temperature TRAIN larger training size
Stengel + van Rijk combo
UMC Server
LR_MR or MR_HR
"""

#%% Part TRAIN na pretrain
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

#tensorflow_version 1.x

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PhIREGANs import PhIREGANs

#%% Training
# epoch 7 of LR_MR
#standard: alpha_advers=0.001
#dropout = 0.8

alpha = 0.001
dropout_rate = 0.8

model_name = '/hpc/shared/uu_imau_ocean/kovacs/More_Time/temperature/MR_HR/'
run_id = 'alpha' + str(alpha) + '_dropout' + str(dropout_rate)

model = PhIREGANs(data_type='temp',
                  print_every = 1000,
                  save_every = 1,
                  N_epochs = 7,
                  epoch_shift = None,
                  model_name = model_name,
                  run_id= run_id)

model.train(r = [3],
            data_path = '/hpc/shared/uu_imau_ocean/kovacs/More_Time/TFRecords/T_mr_hr.tfrecord',
            model_path = '/hpc/shared/uu_imau_ocean/kovacs/More_Time/temperature/MR_HR/cnn00037/cnn',
            batch_size = 1,
            alpha_advers = alpha,
            dropout_rate = dropout_rate) 
