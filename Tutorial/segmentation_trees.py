import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from PIL import Image
from skimage.exposure import histogram
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed

import ee
#ee.Authenticate()
#ee.Initialize()

# open image as numpy array
ImagenTotal = np.asarray(Image.open('Tozeur/Chabbat.png'))
# assign trees the 1st band
trees = ImagenTotal[:,:,0]
# make histograms all 3 bands
for i in range(0,3):
  hist, hist_centers = histogram(ImagenTotal[:,:,i])
  plt.plot(hist_centers,hist)
  plt.title("Band "+ str(i+1))
  plt.show()

edges = canny(trees/255.)
fill_trees = ndi.binary_fill_holes(edges)
plt.imshow(fill_trees)

label_objects, nb_labels = ndi.label(fill_trees)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
trees_cleaned = mask_sizes[label_objects] #index array, take the array value as index, and return an array
plt.imshow(trees_cleaned) #can be used for clean the template


elevation_map = sobel(ImagenTotal)
# look into https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html to create markers
plt.hist(ImagenTotal.flatten(), bins=80)
markers = np.zeros_like(ImagenTotal)
markers[ImagenTotal < 110] = 1
markers[ImagenTotal > 180] = 2

segmentation = watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_trees, _ = ndi.label(segmentation)
plt.imshow(segmentation[:,:,0])

#https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
#https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.watershed

