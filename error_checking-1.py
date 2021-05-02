#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:10:43 2021

@author: rollympoyi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
from skimage.metrics import (adapted_rand_error,
                              variation_of_information)
from skimage.filters import sobel
from skimage.measure import label
from skimage.util import img_as_float
from skimage.feature import canny
from skimage.morphology import remove_small_objects
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  watershed,
                                  mark_boundaries)

filename = input("Input filename of image: ")
def error_check(filename):

    image = np.asarray(Image.open(filename))

    elevation_map = sobel(image)
    markers = np.zeros_like(image)  
    markers[image < 110] = 1 #lower marker for Chabbat
    markers[image > 100] = 2 #higher marker for Chabbat
    im_true = watershed(elevation_map, markers)
    im_true = ndi.label(ndi.binary_fill_holes(im_true - 1))[0]

    edges = sobel(image)
    im_test1 = watershed(edges, markers=468, compactness=0.001)


    #image = img_as_float(image)


    method_names = ['Compact watershed'] #segmentation method used in ur program
    short_method_names = ['Compact WS']

    precision_list = []
    recall_list = []
    split_list = []
    merge_list = []
    for name, im_test in zip(method_names, [im_test1]):
        error, precision, recall = adapted_rand_error(im_true, im_test)
        splits, merges = variation_of_information(im_true, im_test)
        split_list.append(splits)
        merge_list.append(merges)
        precision_list.append(precision)
        recall_list.append(recall)
        print(f"\n## Method: {name}")
        print(f"Adapted Rand error: {error}")
        print(f"Adapted Rand precision: {precision}")
        print(f"Adapted Rand recall: {recall}")
        print(f"False Splits: {splits}")
        print(f"False Merges: {merges}")

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    ax = axes.ravel()

    ax[0].scatter(merge_list, split_list)
    for i, txt in enumerate(short_method_names):
            ax[0].annotate(txt, (merge_list[i], split_list[i]),
                   verticalalignment='center')

    ax[0].set_xlabel('False Merges (bits)')
    ax[0].set_ylabel('False Splits (bits)')
    ax[0].set_title('Split Variation of Information')

    ax[1].scatter(precision_list, recall_list)
    for i, txt in enumerate(short_method_names):
        ax[1].annotate(txt, (precision_list[i], recall_list[i]),
                   verticalalignment='center')
    ax[1].set_xlabel('Precision')
    ax[1].set_ylabel('Recall')
    ax[1].set_title('Adapted Rand precision vs. recall')
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)

    # cannot do this part yet because of a Value Error: NumPy boolean array indexing assignment cannot assign 3 input values to the 953328 output values where the mask is true
    #ax[2].imshow(mark_boundaries(image, im_true)) 
    #ax[2].set_title('True Segmentation')
    #ax[2].set_axis_off()
    #ax[3].imshow(mark_boundaries(image, im_test1))
    #ax[3].set_title('Compact Watershed')
    #ax[3].set_axis_off()



    plt.show()
    
error_check(filename)


