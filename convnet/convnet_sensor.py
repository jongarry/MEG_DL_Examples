#!/bin/python

# Author: Jon Garry
# Date Created: 2018-06-29
# Last Modified: 2019-02-25
#
# Description:  Keras CNN implementation for classifying between active 
#               and baseline classes in MEG data. 
#               The network has a simple architecture and consists of 
#               two convolutional layers followed by a densely connected
#               layer and a softmax classifier.
#               Script accepts a single argument for naming logs and saved models

# General utils
import sys
import datetime
import h5py
import numpy as np
from tqdm import tqdm

# Keras imports
import keras 
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, SpatialDropout2D, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

# Ignore tensorflow's noisy warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    # Script accepts a single argument for a log file name in order to track experiments
    if len(sys.argv) != 2:
        print("\nError: missing network descriptor\nUsage: python convnet_sensor.py 'descriptive_name'\n")
        return
    else:
        log_name = str(sys.argv[1])

    print("Log name:", log_name)
    
    
    # Set hyper-parameters, load data, and set up callbacks
    # -------------------------------------------------------------------------------
    
    epochs = 50
    batch_size = 25
    
    # Open data file and assign dataset       
    f = h5py.File("datasets/camcan_nn_data_scaled_per-record_no-leak.hdf5", 'r')
    
    x_train = f['train']['data']
    y_train = f['train']['labels']

    x_val = f['validation']['data']
    y_val = f['validation']['labels']

    # Set network input shape per sample dataset
    num_channels = x_train.shape[1]
    num_timesteps = x_train.shape[2]
    num_classes = y_train.shape[1]

    input_shape = [num_channels, num_timesteps, 1]
    
    total_batches = int(len(x_train) / batch_size)

    # Set up Tensorboard callback for model analysis and visualisation
    dname = "summary_" + log_name

    tensorboard = keras.callbacks.TensorBoard(log_dir='summaries/' + dname, 
                                            histogram_freq=0,
                                            write_graph=True, 
                                            #write_grads=True,
                                            batch_size=batch_size, 
                                            write_images=True)

    # Set up checkpoint callback for saving model weight values
    mname = "model-weights_" + log_name + '.hdf5'
   
    # Save only the weights with best validation accuracy
    checkpoint = keras.callbacks.ModelCheckpoint("models/" + mname, 
                                            monitor='val_acc', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            mode='max')

    callback_list = [tensorboard, checkpoint]


    # Construct model one layer at a time
    # -------------------------------------------------------------------------------
    
    model = Sequential()

    # ------------------------------------------
    # First convolutional layer
    model.add(Conv2D(8, 
                kernel_size=(8, 16),
                padding='same',
                use_bias=False,
                input_shape=input_shape))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.25))
    
    # ------------------------------------------
    # Second convolutional layer
    model.add(Conv2D(16, 
                kernel_size=(3, 3), 
                padding='same',
                use_bias=False))
    
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.25))

    # ------------------------------------------
    # Dense, fully-connected layer
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    
    # ------------------------------------------
    # Softmax classifier output
    model.add(Dense(num_classes, activation='softmax'))

    # Note on loss functions:
    # For multi-class:  keras.losses.categorical_crossentropy
    # For binary:       keras.losses.binary_crossentropy

    try:
        model = multi_gpu_model(model)
        print("\nTraining network using multiple GPUs...")
    except:
        print("\nTraining network using single GPU or CPU...")
        pass

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.summary()

    # Save architecture to file:
    mname_arch = "model-arch_" + log_name + ".json" 
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/"+ mname_arch, "w") as json_file:
        json_file.write(model_json)
        json_file.close()
    print("\nSaved model to disk\n")
    
    
    # Train the model
    # -------------------------------------------------------------------------------
    # The 'effortless' way of training and validating (assuming data can fit into memory):
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_val, y_val),
            callbacks=callback_list,
            shuffle='batch')
    
    '''
    # The custom batch way of training and validating:
    # (could be useful for implementing input normalisation)
    # Downside: would need to implement Tensorboard summaries by hand
    for epoch in range(epochs):
        for i in tqdm(range(total_batches)):

            batch_x, batch_y = get_batch(x_train, 
                                        y_train, 
                                        batch_size, 
                                        num_channels, 
                                        num_timesteps)

            train_loss, train_acc = model.train_on_batch(batch_x, batch_y)
            
        x_test, y_test = get_batch(x_val, 
                                y_val, 
                                len(y_val), 
                                num_channels, 
                                num_timesteps)

        test_loss, test_acc = model.test_on_batch(x_test, y_test)
            
        print("Epoch: %d, train loss: %.3f, train acc: %.3f, test loss: %.3f, test acc: %.3f" % 
            (epoch, train_loss, train_acc, test_loss, test_acc))
    '''


    '''
    # The semi-automated, generator method for training and validating:
    # Tensorboard callbacks do not work properly with using a generator for
    # validation data
    model.fit_generator(
        generate_arrays_from_file(x_train, y_train, batch_size, num_channels, num_timesteps),
        steps_per_epoch=total_batches, 
        epochs=epochs, 
        verbose=1, 
        validation_data=generate_arrays_from_file(x_val, y_val, 100, num_channels, num_timesteps),
        validation_steps=5, 
        use_multiprocessing=True, 
        shuffle=False,
        callbacks=[tb_callback])
    '''


    f.close()
    

def generate_arrays_from_file(data, labels, batch_size, num_channels, num_timesteps):
# Generator function for use with keras fit_generator() function
# yields the passed data in the shape the network input requires

    x_data = np.zeros([batch_size, num_channels, num_timesteps, 1],float)
    y_data = np.zeros([batch_size, 2], int)

    while True:
        for i in range(batch_size):
            x_data[i,:,:,0] = data[i,:,:]
            y_data[i,:] = labels[i,:]
        yield (x_data, y_data)

def get_batch(data, labels, batch_size, num_channels, num_timesteps):
# Function for randomly sampling from passed datasets
# Returns feature and label arrays of size batch_size

    x_data = np.zeros([batch_size, num_channels, num_timesteps, 1],float)
    y_data = np.zeros([batch_size, 2], int)

    np.random.seed(1337)

    # Populate sample sets
    for i in range(batch_size):
        idx = np.random.randint(0,batch_size)

        x_data[i,:,:,0] = data[idx,:,:]
        y_data[i,:] = labels[idx, :]

    return x_data, y_data



if __name__ == "__main__":
    main()
