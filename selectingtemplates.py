'''
@mawaskow
Description:
This program allows a user to select a polygon from an image 
to be saved as its own image.
The completed program will be able to be used in conjunction
with other code in order to create template files for machine
learning feature identification algorithms.
'''

# import statements
from PIL import Image # PIL library named pillow

import numpy as np

from skimage import data
from skimage.feature import match_template

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets  import RectangleSelector

import tkinter
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename

'''
import mpld3
from mpld3 import plugins
'''

# How to create a template from clicking on a picture

'''
Open image
Display image
Prompt user to Increase resolution to desired dimension
Prompt user to click to draw rectangle
Display selected polygon
Display selected image
Save image
'''

#infile= open(askopenfilename(), "r")
#outfile= open(asksaveasfilename(), "w")

'''
fig, ax = plt.subplots(figsize=(10,5))
im = ax.imshow(RGBImage,extent=(0, 3100, 656,0),origin='upper', zorder=1, interpolation='none')
plugins.connect(fig, plugins.MousePosition(fontsize=14))
mpld3.enable_notebook()
mpld3.display()
'''

ImagenTotal = np.asarray(Image.open('../Tutorial/Tozeur/Chabbat.png'))
trees = ImagenTotal[:,:,0]

plt.figure()
plt.imshow(trees)
plt.show()

xdata = np.linspace(0,9*np.pi, num=301)
ydata = np.sin(xdata)

fig, ax = plt.subplots()
line, = ax.plot(xdata, ydata)


def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    ax.add_patch(rect)


rs = RectangleSelector(ax, line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)

plt.show()