'''
@mawaskow
Description:
This program allows a user to select a polygon from an image 
to be saved as its own image.
The completed program will be able to be used in conjunction
with other code in order to create template files for machine
learning feature identification algorithms.
https://stackoverflow.com/questions/8056458/display-image-with-a-zoom-1-with-matplotlib-imshow-how-to
https://matplotlib.org/stable/gallery/widgets/rectangle_selector.html
'''

# import statements
from PIL import Image # PIL library named pillow

import numpy as np

from skimage import data
from skimage.feature import match_template

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector

import tkinter
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename

def gettestimg(show = False):
    '''
    Returns the test image to be analyzed.
    '''
    img = Image.open('../Tutorial/Tozeur/Chabbat.png')
    ImagenTotal = np.asarray(img)
    trees = ImagenTotal[:,:,0]
    if show:
        plt.figure()
        plt.imshow(trees)
        plt.show()
    return img

def get_img(show = False):
    '''
    Prompts user to select the file they want to open.
    Returns the image to be analyzed.
    '''
    infile= askopenfilename()
    print(infile)
    img = Image.open(infile)
    im_tot = np.asarray(img)
    if show:
        plt.figure()
        plt.imshow(im_tot[:,:,0])
        plt.title("Image to be Analyzed")
        plt.show()
    return img

###
def zoom(img, factor = 1):
    '''
    margin = 0.05 # (5% of the width/height of the figure...)
    '''
    margin = 0.05/factor
    dpi = 80
    xpixels, ypixels = img.size
    figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    ax.imshow(img, interpolation='none')
    plt.show()
###

def line_select_callback(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    #tmp.append([x1, y1, x2, y2])

def toggle_selector(event):
    print(' Key pressed.')
    if event.key == 't':
        if toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        else:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

def select_temp(img):
    fig, ax = plt.subplots(figsize = (10,7))
    N = 100000  # If N is large one can see improvement by using blitting.
    x = np.linspace(0, 10, N)
    ax.imshow(img)  # plot something
    ax.set_title(
        "Click and drag to draw a rectangle.\n"
        "Press 't' to toggle the selector on and off.\n"
        "Select & deselect magnifying glass/'zoom to rectangle' \nbutton below for closer selection.")
    # drawtype is 'box' or 'line' or 'none'
    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                        drawtype='box', useblit=True,
                                        button=[1, 3],  # disable middle button
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.show()

def main():
    see_results = False
    image = gettestimg(see_results)
    #image = get_img(see_results)
    fct = 500
    #fct = float(input("Select zoom level (>0) [>1 for zoom in, <1 for zoom out]: "))
    select_temp(image)

if __name__ == "__main__":
    main()