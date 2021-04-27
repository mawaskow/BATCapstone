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


rs = RectangleSelector(ax, line_select_callback,drawtype='box', useblit=False, button=[1], 
    minspanx=5, minspany=5, spancoords='pixels', interactive=True)

plt.show()

import ctypes
from graphics import *

def getScsz(text):
    '''
    This function takes a text argument which becomes the window title
    as well as initializes window w and values for x and y of full screen
    '''
    scsz=[]
    root = tkinter.Tk()
    scsz.append(root.winfo_screenwidth())
    scsz.append(root.winfo_screenheight())
    #root.destroy() ### generates error
    w = GraphWin(text, scsz[0], scsz[1])
    return w, scsz[0], scsz[1]

def okBox(w,x,y, msg, txtcolor, boxcolor):
    '''
    This function draws a box about 2/3 of the way down the screen which
    undraws when the inside of the box is clicked. Directions (msg parameter)
    are written in the box, and the text and box color can also be changed
    '''
    # initializing box width (rw) and height (rh)
    rw, rh = x/2, 60
    x1 = x/2 - rw/2
    y1 = y/2.5 - rh/2
    x2, y2 = x/2 + rw/2, y/2.5 + rh/2
    # draw the rectangle
    btmR = Rectangle(Point(x1, y1), Point(x2,y2)).draw(w)
    btmR.setFill(boxcolor)
    # draw/write the text
    clickMouse = Text(Point(x/2, y/2.5), msg).draw(w)
    clickMouse.setFace("arial"), clickMouse.setSize(13), clickMouse.setStyle("bold")
    clickMouse.setTextColor(txtcolor)
    # initialize while loop to determine when the okbox has been clicked
    while True:
        p = w.getMouse()
        if p.getX() >= x1 and p.getX() <= x2:
            if p.getY() >= y1 and p.getY() <= y2:
                break
    clickMouse.undraw()
    btmR.undraw()
    return

def main():
    w, x, y = getScsz("Template Selection Window")
    okBox(w,x,y, "Please Click Here to Continue...", "white", "black")

if __name__ == "__main__":
    main()