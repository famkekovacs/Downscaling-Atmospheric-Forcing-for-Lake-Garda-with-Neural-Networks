"""
Temperature PRETRAIN larger training size
Stengel + van Rijk combo
UMC Server
LR_MR or MR_HR
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from PhIREGANs import PhIREGANs

#%% Pretraining

start_epoch_shift = 0


for i in range(39):
  # requested number of epochs, refresh model every time to increase runtime
  model_name = '/hpc/shared/uu_imau_ocean/kovacs/More_Time/temperature/'
  run_id = 'MR_HR' #change to MR when needed

  if start_epoch_shift==0 and i==0:
    model_path = None
  else:
    model_path= '/'.join([model_name + run_id, 'cnn{0:05d}/cnn'.format(start_epoch_shift + i)])
    print(model_path)

  if start_epoch_shift + i >= 100:
    learning_rate=1e-5
  else:
    learning_rate=1e-5

  model = PhIREGANs(data_type='temp',
                        N_epochs = 1,
                        save_every = 1,
                        learning_rate = learning_rate,
                        epoch_shift = start_epoch_shift + i,
                        print_every = 1000,
                        model_name = model_name,
                        run_id = run_id
                        )

  model_dir = model.pretrain(r=[3],
                      data_path='/hpc/shared/uu_imau_ocean/kovacs/More_Time/TFRecords/T_mr_hr.tfrecord',
                      model_path = model_path,
                      batch_size=1)
  # r=[2] for LR_MR and r=[3] for MR_HR
  # read and plot the loss function
  loss_df = pd.read_csv(model.model_name + '/loss_df.csv')

  fig = plt.figure()

  plt.plot(loss_df['Epoch'],loss_df['G loss'])
  plt.ylim(0,np.max(loss_df['G loss'])+0.01)
  plt.ylabel('Loss')
  plt.xlabel('Epoch')

  # vertical lines for when learning rate is changed
  #plt.vlines(27,0,loss_df['G loss'][26])

  plt.tight_layout()
  plt.show()
