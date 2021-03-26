# import statements
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

from scipy import ndimage as ndi
from skimage import data
from skimage.exposure import histogram
from skimage.feature import canny
from skimage.filters import sobel
from skimage.segmentation import watershed

print("Reading image...")

# open image file
ImagenTotal = np.asarray(Image.open('../Tutorial/Tozeur/Chabbat.png'))
# extract data frame/matrix from 0th layer
trees = ImagenTotal[:,:,0]

print("Plotting histogram statistics for each band...")

for i in range(0,3):
  # gets an array of histogram values and the values at the center of each bin for each layer
  hist, hist_centers = histogram(ImagenTotal[:,:,i])
  plt.figure()
  plt.plot(hist_centers,hist)
  plt.title("Histogram " + str(i+1)+ " of 3")
  plt.xlabel("Centers of the Histogram Bins")
  plt.ylabel("Histogram Values")
  plt.savefig('./Outputs/GraphofHistogramValues_'+ str(i)+'.png', dpi=300, bbox_inches='tight')
  #plt.show()

print("Filling the holes in the trees...")

edges = canny(trees/255.)
fill_trees = ndi.binary_fill_holes(edges) # fills the holes in binary objects

print("Building the filled trees image...")

plt.figure()
plt.imshow(fill_trees)
plt.title("Filled Trees")
plt.savefig('./Outputs/filled_trees.png', dpi=300, bbox_inches='tight')
#plt.show()

print("Cleaning trees...")

label_objects, nb_labels = ndi.label(fill_trees) # labels features in an array
sizes = np.bincount(label_objects.ravel()) # .ravel() flattens array, .bincount() returns label array and number of labels
mask_sizes = sizes > 20
mask_sizes[0] = 0
trees_cleaned = mask_sizes[label_objects] #index array, take the array value as index, and return an array

print("Building cleaned trees image...")

plt.figure()
plt.imshow(trees_cleaned) #can be used for clean the template
plt.title("Cleaned Trees")
plt.savefig('./Outputs/cleaned_trees.png', dpi=300, bbox_inches='tight')
#plt.show()

print("Building elevation map...")

elevation_map = sobel(ImagenTotal)
# look into https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html to create markers

print("Building histogram of elevations...")

plt.figure()
plt.hist(ImagenTotal.flatten(), bins=80)
markers = np.zeros_like(ImagenTotal) # makes a matrix of same dimension but full of 0s
markers[ImagenTotal < 110] = 1
markers[ImagenTotal > 180] = 2
plt.title("Histogram")
plt.savefig('./Outputs/histogram.png', dpi=300, bbox_inches='tight')
#plt.show()

print("Segmenting image...")

segmentation = watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_trees, _ = ndi.label(segmentation)

print("Building segmented trees image...")

plt.figure()
plt.imshow(segmentation[:,:,0])
plt.title("Segmented Image")
plt.savefig('./Outputs/elevation.png', dpi=300, bbox_inches='tight')
plt.show()

#https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
#https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.watershed