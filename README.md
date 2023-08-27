# Downscaling Atmospheric Forcing for Lake Garda with Neural Networks

The machine learning code consists of the following files: PhiREGANs.py, sr_network.py and utils.py.
A file with a lot of handy commands can be found in helper_functions.py.

The files which are used on the HPC facility of the Utrecht Bioinformatics Community (UBC) consists of two different groupes: the .py files and .sh files.
The .sh files are submitting files to run the code, which are the .py files. These .py files contain the pretraining, training and testing of the neural network. These files are from the more training years case (MT for more time) and the future test case (FF for future forcing). For the standard test case (ST) the same files can be used, only different locations of the .tfrecord and later the models should be modified.

At last two different jupyter notebooks exists: data preparation and analysis files.
The data preparation files contain the datasets and the modification of them to use them in the neural network. Therefore the data files need to be set to a .tfrecord file. The analysis files contain the code to analyse the super-resolution and compare them to the original data and their counterparts.
