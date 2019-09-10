#!/bin/python

# Author: Jon Garry
# Date Created: 2018-06-18
# Last Modified: 2019-01-24
#
# Description:  Script for creating training data for use with classifiers.
#               Data is extracted from MNE .fif files for a given list of 
#               participant IDs. Each record represents a single task trial.
#               Data are filtered, resampled, and split up into
#               active and baseline time courses. 
#               After processing, all records are saved to an HDF5 file with 
#               keys containing participant ID, trial number, and 
#               active/baseline label.


import os
from tqdm import tqdm
import mne
import pandas as pd
import numpy as np
import h5py

from sklearn.preprocessing import scale
# Note: scale() only allows scaling over each axis separately, not the entire dataset
# Default:      axis=0 per column (opposite of other packages?)
#               axis=1 per row


data_path = '~/camcan/proc_data/'
data_path = os.path.expanduser(data_path)
fif_path = data_path + 'TaskSensorAnalysis_transdef/'

file_name = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'


# Get list of participants from csv
part_data = np.loadtxt(data_path + 'evoked_process_stats.csv', dtype = object, delimiter=',', skiprows=1)
num_epochs = np.array(part_data[:,5],float)
part_list = part_data[np.where(num_epochs>0),1][0]

"""
# OR manually select list of participants for testing/debugging
part_list = ['CC110033', 'CC110037', 'CC110045', 'CC110056', 'CC110069',
               'CC110087', 'CC110098', 'CC110101', 'CC110126', 'CC110174']
"""


for part in tqdm(part_list):
    
    #print("\nProcessing", part)

    file_path = fif_path + part + "/" + file_name

    # Process and extract epoch data using mne
    # -------------------------------------------------------------------------
   
    # Load epochs fif file
    epochs = mne.read_epochs(file_path, verbose='WARNING')

    # Get only the magnetometer channels
    epochs.pick_types(meg='mag')

    # Apply lowpass filter of 40 Hz and apply new baseline
    epochs.filter(None, 40)
    
    # Downsample
    epochs = epochs.copy().resample(250, npad='auto')
    
    # Apply baseline
    epochs.apply_baseline((None, None))

    # Define rest and task time ranges (in seconds)
    task_tmin = -0.5
    task_tmax = 0.5

    rest_tmin = -1.7
    rest_tmax = -0.7

    # Split epochs into rest and task 
    rest = epochs.copy()
    rest.crop(rest_tmin, rest_tmax-0.004)

    task = epochs.copy()
    task.crop(task_tmin, task_tmax-0.004)

    # Get data from epoch objects
    rest_data = rest.get_data()
    task_data = task.get_data()

    # Export data to file
    # -------------------------------------------------------------------------
    f = h5py.File("processed_data.hdf5", 'a')
   
    # Iterate over each epoch for each class and write to file
    for epoch in range(rest_data.shape[0]):
          
        rest_record = rest_data[epoch, :, :]
        task_record = task_data[epoch, :, :]

        # Centre record and scale to unit variance
        rest_record_scaled = (rest_record - np.mean(rest_record)) / np.std(rest_record)
        task_record_scaled = (task_record - np.mean(task_record)) / np.std(task_record)

        # Create dataset name
        rest_dset_name = part + "_" + str(epoch) + "_" + str(0)
        task_dset_name = part + "_" + str(epoch) + "_" + str(1)

        # Store measurement array as dataset
        f.create_dataset(rest_dset_name, data=rest_record_scaled)
        f.create_dataset(task_dset_name, data=task_record_scaled)

    f.close()
