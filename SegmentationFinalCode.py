# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:51:14 2021

@author: Victoria Lindsey
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.exposure import histogram
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed

print("What is the name of the desired file to be analyzed.")
print("Remember that the current directory is set to the location of the program. If file is in a different location, move the file or give the new directory to find the desired file.")
print("Press enter when finished.")
filename = input("File name:")
print("")

def segmentation (filename):
    Image_1 = np.asarray(Image.open(filename))
    tree = Image_1[:, :, 0]

    for i in range(0, 3):
        hist, hist_centers = histogram(Image_1[:,:,i])
        plt.plot(hist_centers, hist)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Image Values')
        plt.show()

    print("After looking at the histogram detemine the upper and the lower setting for the markers.")
    print("Press enter when finished")
    lower = int(input("Lower Bound:"))
    upper = int(input("Upper Bound:"))

    if lower >250 or lower < 0:
        print("Unusable input for Lower Bound would you like to retry?")
        response = input("Input Y for yes and N for no:")
        if response == "Y":
            lower = int(input("Lower Bound:")) 
    
    print("")       
 
    if upper > 250 or upper < 0:
        print("Unusable input for Upper Bound would you like to retry?")
        response = input("Input Y for yes and N for no:")
        if response == "Y":
            upper = int(input("Upper Bound:"))                      


    edge = canny(tree/600)
    fill_tree = ndi.binary_fill_holes(edge)
    plt.imshow(fill_tree)

    label_objects, nb_labels = ndi.label(fill_tree)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 5
    mask_sizes[0] = 0
    trees_cleaned = mask_sizes[label_objects] 
    print("The number of date palms is:" + str(len(sizes)))
    count = len(sizes)
   
    
    plt.imshow(trees_cleaned)

    elevation_map = sobel(Image_1)
    plt.imshow(elevation_map)
    markers = np.zeros_like(Image_1)
    markers[Image_1 < lower] = 1
    markers[Image_1 > upper] = 2

#Chabbat: upper = 180, lower = 110
#PalmTreesOasis1: upper = 80, lower = 100
#Shakmo: upper = 100, lower = 75

    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_trees, _ = ndi.label(segmentation)
    plt.imshow(segmentation[:,:,0])
    
    return(segmentation, labeled_trees, count)

segmentation(filename)