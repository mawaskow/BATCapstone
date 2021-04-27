'''
@mawaskow
Description:
This program recognizes crop row lines to clean the data of an image being used in
a machine learning algorithm.

References:
https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
'''

# import statements
import numpy as np

from PIL import Image, ImageEnhance

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

# Constructing test image

print("Enhancing image with contrast transform...")

img = Image.open('../Tutorial/Tozeur/Chabbat.png')
plt.figure()
plt.imshow(img)

enhancer = ImageEnhance.Contrast(img)
clvl = 4
factor = 259*(clvl+255)/(255*(259-clvl))
img_cont = enhancer.enhance(factor)

plt.figure()
plt.imshow(img_cont)

print("Loading image into array...")

ImagenTotal = np.asarray(img)
image = ImagenTotal[:,:,0]

print("Conducting basic Hough Transform...")

# Classic straight-line Hough transform
# Set a precision of 0.5 degree with tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
precis = 0.25
tested_angles = np.linspace(-np.pi*precis, np.pi*precis, 360, endpoint=False)
h, theta, d = hough_line(image)  # additional hough_line() argument: theta=tested_angles

print("Building basic Hough Transform output image...")

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

angle_step = 0.5 * np.diff(theta).mean()
d_step = 0.5 * np.diff(d).mean()
bounds = [np.rad2deg(theta[0] - angle_step),
          np.rad2deg(theta[-1] + angle_step),
          d[-1] + d_step, d[0] - d_step]
ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image, cmap=cm.gray)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    x = np.linspace(x0, x0+100, 101)
    m=np.tan(angle + np.pi/2)
    ax[2].plot(x, m*x+y0)

plt.tight_layout()
plt.savefig('./Outputs/basic_hough_precis'+str(precis)+'.png', dpi=300, bbox_inches='tight')
#plt.show()

#################################################################################

from skimage.transform import probabilistic_hough_line

print("Conducting Probabilistic Hough Transform...")

# Line finding using the Probabilistic Hough Transform
ImagenTotal = np.asarray(Image.open('../Tutorial/Tozeur/Chabbat.png'))
image = ImagenTotal[:,:,0]
edges = canny(image, 2, 1, 25)
# probabilistic_hough_line(image, threshold=10, line_length=50, line_gap=10, theta=None, seed=None)
# line_length: min accepted length of detected lines
# line_gap: max gap btwn px to still form line. Increase to merge broken lines more aggressively
# returns list containing lines identified as point start and end ((x0, y0), (x1, y1))

thres = 10
linlen = 100
lingap = 5

lines = probabilistic_hough_line(edges, threshold=thres, line_length=linlen, line_gap=lingap)

print("Building Probabilistic Hough Transform output image...")

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.savefig('./Outputs/prob_hough_thres'+str(thres)+'_len'+str(linlen)+'_gap'+str(lingap)+'.png', dpi=300, bbox_inches='tight')
plt.show()

print("Program ended.")