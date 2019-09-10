#!/bin/python
# Author: Jon Garry
# 
# Description:  Script for testing visualisation and attribution methods. 
#               Techiques include feature map plotting, activation and saliency
#               using keras-vis, and a custom occlusion routine.
#
#               A VGG16 network is loaded with weights that were trained using the
#               Imagenet 1000 dataset. Sample images can be loaded to test 
#               classification and visualisation techniques.

from keras.applications import VGG16
from vis.utils import utils
from keras import activations
from keras import backend as K

from vis.visualization import visualize_activation
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
from vis.input_modifiers import Jitter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bars


# ignore Tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def iter_occlusion(image, size=4):
# function for performing occlusion mapping over an input image

    occlusion = np.full((size * 5, size * 5, 3), [0], np.float32)
    occlusion_centre = np.full((size, size, 3), [0], np.float32)
    occlusion_padding = size * 2

    image_padded = np.pad(image,
                    ((occlusion_padding, occlusion_padding),
                    (occlusion_padding, occlusion_padding),
                    (0, 0)),
                    'constant', 
                    constant_values = 0.0)

    for y in tqdm(range(occlusion_padding, image.shape[0] + occlusion_padding, size)):
        for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):

            tmp = image_padded.copy()

            tmp[y - occlusion_padding:y + occlusion_centre.shape[0] + occlusion_padding,
                x - occlusion_padding:x + occlusion_centre.shape[1] + occlusion_padding] = occlusion

            tmp[y:y + occlusion_centre.shape[0], 
                x:x + occlusion_centre.shape[1]] = occlusion_centre

            yield x - occlusion_padding, y - occlusion_padding, tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]


# Build VGG16 network with Imagenet weights
model = VGG16(weights='imagenet', include_top=True)

# Search layer index by name
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap out softmax with linear layer
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# Load an example image
data = utils.load_img('griffin_bed.jpg', target_size=(224, 224))

# Load class dictionary
with open('imagenet1000_clsidx_to_labels.txt','r') as inf:
    class_labels = eval(inf.read())

# reshape data for use with model and make prediction
inp = np.reshape(data, (1,224,224,3))

pred_idx = np.argmax(model.predict(inp))

print("\nPredicted class %i : %s" % (pred_idx, class_labels[pred_idx]))

class_idx = pred_idx
#class_idx = 283 # Persian


# Uncomment blocks for each visualisation/attribution technique
# ===============================================================


# Feature Maps
# --------------------------

# Plot some of the first layer feature maps
# Different layers can be examined by changing the index in model.layers[1].output
# List layers using model.summary()
get_conv_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                        [model.layers[1].output])

# Get layer output using input image 
feature_maps = get_conv_layer_output([inp,0])[0]

# Remove extra dimension for plotting
feature_maps = np.squeeze(feature_maps)

# Plot the first 5 feature maps 
for i in range(5):
    plt.figure()
    plt.imshow(feature_maps[...,i])

'''
# Activation Visualisation
# --------------------------

# Generates inputs that maximise output for a specific class
# can be seeded with input images using seed_input

# Note: max_iter=500 can take a long time to complete but can produce
#       informative looking maps

img = visualize_activation(model, 
                        layer_idx, 
                        filter_indices=class_idx,
                        #seed_input=data,
                        max_iter=500,
                        input_modifiers=[Jitter(16)],
                        verbose=True)

plt.figure()
plt.imshow(img)
'''


'''
# Saliency/Attention Maps
# ---------------------------------

# plot gradient-based saliency maps over specified class 

grads = visualize_saliency(model, 
                    layer_idx, 
                    filter_indices=class_idx, 
                    seed_input=data, 
                    backprop_modifier='guided')

fig = plt.figure()
fig.set_tight_layout(True)
plt.imshow(data)
plt.imshow(grads, cmap='jet', alpha=0.8)
'''


'''
# grad-CAM
grads = visualize_cam(model,
                    layer_idx,
                    filter_indices=class_idx,
                    seed_input=data,
                    backprop_modifier=None)
        

plt.figure()
plt.imshow(data)
plt.imshow(grads, cmap='jet', alpha=0.8)
'''


'''
# Image Occlusion
# -------------------

# NOTE: This process can take some time depending on the size of the 
#       occluder that is used. A smaller occluder means more iterations.

correct_class = class_idx

# input tensor for model.predict
inp = data.reshape(1, 224, 224, 3)

# image data for matplotlib's imshow
img = data.reshape(224, 224, 3)

# occlusion
img_size = img.shape[0]
#occlusion_size = 4
occlusion_size = 8

heatmap = np.zeros((img_size, img_size), np.float32)
class_pixels = np.zeros((img_size, img_size), np.int16)



for n, (x, y, img_float) in enumerate(iter_occlusion(data, size=occlusion_size)):

    X = img_float.reshape(1, 224, 224, 3)
    out = model.predict(X)

    heatmap[y:y + occlusion_size, x:x + occlusion_size] = out[0][correct_class]
    class_pixels[y:y + occlusion_size, x:x + occlusion_size] = np.argmax(out)

heatmap_scaled = (heatmap - np.amin(heatmap)) / (np.amax(heatmap) - np.amin(heatmap))

fig = plt.figure()
fig.set_tight_layout(True)
plt.imshow(img)
plt.imshow(heatmap_scaled, cmap='jet', alpha=0.8)
plt.colorbar()
'''

plt.show()
