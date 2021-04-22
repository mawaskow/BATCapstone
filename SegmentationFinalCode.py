# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:51:14 2021

@author: Victoria Lindsey
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from PIL import Image
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

Image = np.asarray(Image.open(filename))
tree = Image[:, :, 0]

for i in range(0, 3):
    hist, hist_centers = histogram(Image[:,:,i])
    plt.plot(hist_centers, hist)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Image Values')
plt.show()

print("After looking at the histogram detemine the upper and the lower setting for the markers.")
print("Press enter when finished")
lower = int(input("Lower bound:"))
upper = int(input("Upper bound:"))


edge = canny(tree/600)
fill_tree = ndi.binary_fill_holes(edge)
plt.imshow(fill_tree)

label_objects, nb_labels = ndi.label(fill_tree)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 5
mask_sizes[0] = 0
trees_cleaned = mask_sizes[label_objects] 
plt.imshow(trees_cleaned)

elevation_map = sobel(Image)
plt.imshow(elevation_map)
markers = np.zeros_like(Image)
markers[Image < lower] = 1
markers[Image > upper] = 2

#Chabbat: upper = 180, lower = 110
#PalmTreesOasis1: upper = 80, lower = 100
#Shakmo: upper = 100, lower = 75

segmentation = watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_trees, _ = ndi.label(segmentation)
plt.imshow(segmentation[:,:,0])

