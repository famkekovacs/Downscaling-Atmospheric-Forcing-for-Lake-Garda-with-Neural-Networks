"""
Temperature Test Train data larger training size
Stengel + van Rijk combo
UMC Server
LR_MR or MR_HR
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PhIREGANs import PhIREGANs

#%% Test train data
  
#grid search initial model was epoch 7 LR MR
#standard: alpha_advers=0.001

alphas = 0.001
r = [3]
run_id = 'MR_HR/'
model_name = '/hpc/shared/uu_imau_ocean/kovacs/More_Time/temperature/'
data_path = '/hpc/shared/uu_imau_ocean/kovacs/More_Time/TFRecords/T_mr_hr.tfrecord'
a_d = 'alpha0.001_dropout0.8'
model_dir = '/'.join([model_name + run_id + a_d, 'gan-all00007/gan'])
data_out_path = '/hpc/shared/uu_imau_ocean/kovacs/More_Time/data_out/temperature/MR_HR/'
        
model = PhIREGANs(data_type='temp',
                  print_every = 1000,
                  save_every = 1,
                  N_epochs = 7,
                  epoch_shift = None,
                  model_name = model_name,
                  run_id= run_id,
                  data_out_path = data_out_path)

model.set_mu_sig(data_path='/hpc/shared/uu_imau_ocean/kovacs/More_Time/TFRecords/T_mr_hr.tfrecord')

model.test(r=r,
               data_path=data_path,
               model_path=model_dir,
               batch_size=1)
