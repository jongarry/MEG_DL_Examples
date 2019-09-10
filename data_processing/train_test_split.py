#!/bin/python

# Author: Jon Garry
# Date Created: 2018-06-18
# Last Modified: 2019-02-20
#
# Description:  Script for splitting up the set of processed records 
#               into training, validation, and testing subsets
#
#               Random sampling is performed on the basis of participant
#               to ensure no information leakage. 
#               i.e. records from one participant can only exist in one 
#               of the three subsets.

from tqdm import tqdm
import h5py
import numpy as np



def main():
    # Open connection to HDF5 files
    fname = 'processed_data.hdf5'
    fname2 = 'camcan_nn_data_scaled_per-record.hdf5'

    input_f = h5py.File(fname, 'r')
    output_f = h5py.File(fname2, 'a')

    # Get keys and dimensions of dataset
    keys = np.array(list(input_f.keys()), object)
    num_channels = input_f[keys[0]].shape[0]
    num_samples = input_f[keys[0]].shape[1]

    # Generate list of participant IDs and map ID back to key indices
    part_list = []
    part_key_map = {}

    for idx, key in enumerate(keys):
        part_id = key[:8]

        # map part_id to key indices
        part_key_map[idx] = part_id
        
        if part_id not in part_list:
            part_list.append(part_id)


    # Set up train / validate / test split over participants 
    # (0.8, 0.05, 0.15)
    num_parts = len(part_list)
    num_train_parts = int(np.round(num_parts * 0.8))
    num_val_parts = int(np.round(num_parts * 0.05))
    num_test_parts = num_parts - (num_train_parts + num_val_parts) 


    # Randomly shuffle part_list and extract part IDs
    np.random.shuffle(part_list)

    # Split participants into subsets
    train_parts = part_list[:num_train_parts]
    val_parts = part_list[num_train_parts:num_train_parts + num_val_parts]
    test_parts = part_list[num_train_parts + num_val_parts:]

    # Get indices of keys associated with each participant
    train_indices = get_keys_by_value(part_key_map, train_parts)
    val_indices = get_keys_by_value(part_key_map, val_parts)
    test_indices = get_keys_by_value(part_key_map, test_parts)

    train_keys = keys[train_indices]
    val_keys = keys[val_indices]
    test_keys = keys[test_indices]

    # Randomly shuffle keys
    np.random.shuffle(train_keys)
    np.random.shuffle(val_keys)
    np.random.shuffle(test_keys)

    num_train = len(train_keys)
    num_val = len(val_keys)
    num_test = len(test_keys)

    print("\nTotal number of records:", len(keys))
    print("Splits: %.3f training, %.3f validation, %.3f testing" % (len(train_keys)/len(keys), len(val_keys)/len(keys), len(test_keys)/len(keys)))

    # Create groups and placeholder datasets in new file
    output_f.create_group('train')
    output_f.create_group('validation')
    output_f.create_group('test')

    train_data = output_f['train'].create_dataset('data', (num_train, num_channels, num_samples, 1), float)
    train_labels = output_f['train'].create_dataset('labels', (num_train, 2), int)

    val_data = output_f['validation'].create_dataset('data', (num_val, num_channels, num_samples, 1), float)
    val_labels = output_f['validation'].create_dataset('labels', (num_val, 2), int)

    test_data = output_f['test'].create_dataset('data', (num_test, num_channels, num_samples, 1), float)
    test_labels = output_f['test'].create_dataset('labels', (num_test, 2), int)

    # Populate split datasets
    # Train
    print("\nProcessing training data...")
    idx = 0
    for k in tqdm(train_keys):
        train_data[idx,:,:,0] = input_f[k]

        label = int(k[-1])

        if label == 1:
            train_labels[idx,:] = np.array([1, 0])
        else:
            train_labels[idx,:] = np.array([0, 1])

        idx += 1

    # Validation
    print("\nProcessing validation data...")
    idx = 0
    for k in tqdm(val_keys):
        val_data[idx,:,:,0] = input_f[k]

        label = int(k[-1])

        if label == 1:
            val_labels[idx,:] = np.array([1, 0])
        else:
            val_labels[idx,:] = np.array([0, 1])
        
        idx += 1

    # Test
    print("\nProcessing test data...")
    idx = 0
    for k in tqdm(test_keys):
        test_data[idx,:,:,0] = input_f[k]

        label = int(k[-1])

        if label == 1:
            test_labels[idx,:] = np.array([1, 0])
        else:
            test_labels[idx,:] = np.array([0, 1])

        idx += 1

    input_f.close()
    output_f.close()

def get_keys_by_value(input_dict, values):
# Function that returns keys associated with passed values
# Used for getting list of key indices associated with participant IDs

    key_list = []
    
    for key, value in input_dict.items():
        
        if value in values:
            key_list.append(key)

    return key_list

if __name__ == "__main__":
    main()
