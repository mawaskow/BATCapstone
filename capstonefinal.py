'''
Authors: Victoria Lindsey, Rolly Mpoyi, Ales Waskow
Class: BAT 498 Capstone
Date: 1 May 2021
Description:
This program takes an image file and templates and runs an algorithm
to identify date palm trees in the image.
'''

##### import statements
import os

import numpy as np
import math

from PIL import Image, ImageEnhance

from scipy import ndimage as ndi
from skimage import data
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny, match_template
from skimage.draw import line
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.filters import sobel
from skimage.measure import label
from skimage.util import img_as_float
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, morphological_geodesic_active_contour, inverse_gaussian_gradient, watershed, mark_boundaries
from skimage.exposure import histogram

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector

import tkinter
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename

import imutils
import glob
import cv2

##### 

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

def line_select_callback(eclick, erelease):
    '''
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    https://matplotlib.org/stable/gallery/widgets/rectangle_selector.html
    '''
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    #tmp.append([x1, y1, x2, y2])

def toggle_selector(event):
    '''
    https://matplotlib.org/stable/gallery/widgets/rectangle_selector.html
    '''
    print(' Key pressed.')
    if event.key == 't':
        if toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        else:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

def select_temp(img):
    '''
    https://matplotlib.org/stable/gallery/widgets/rectangle_selector.html
    '''
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

def adj_contrast(img, clvl, show = False):
    '''
    Author: Waskow
    Purpose: Adjusts contrast to enhance & clean image
    Inputs: 
        img = image (opened with PIL library Image.open() method)
        clvl = [integer]
        show = [boolean] whether or not to display the old & new images
    Returns: contrast-adjusted image
    '''
    enhancer = ImageEnhance.Contrast(img)
    factor = 259*(clvl+255)/(255*(259-clvl))
    img_cont = enhancer.enhance(factor)
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].title.set_text('Original Image')
        ax[0].imshow(img)
        ax[1].title.set_text('Contrast Enhancement')
        ax[1].imshow(img_cont)
    return img_cont

def segmentation(filename):
    '''
    Author: Lindsey
    Purpose: Completes segmentation method
    Inputs: filename
    Returns: 
    '''
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
    
    return(segmentation, labeled_trees)

def categorization(filename, threshold = 0.7):
    '''
    Author: 
    Purpose: 
    Inputs:
    Returns:
    '''
    img_rgb = cv2.imread(filename)  # original image
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) # grey-scaling original image

    template_small = cv2.imread(category_small,0) #reading template
    w, h = template_small.shape[::-1] 

    res_small = cv2.matchTemplate(img_gray,template_small,cv2.TM_CCOEFF_NORMED) 
    count_small = 0  # creating counter for the matched trees
    loc = np.where( res_small >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)# creating rectangle of region of template
        count_small+=1
    cv2.imwrite('res_small.png',img_rgb) # creating output file of counted trees being part of that category

    template_medium = cv2.imread(category_medium,0) #reading template
    w, h = template_medium.shape[::-1] 

    res_medium = cv2.matchTemplate(img_gray,template_medium,cv2.TM_CCOEFF_NORMED) 

    count_mid = 0
    loc = np.where( res_medium >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        count_mid+=1
    
    cv2.imwrite('res_medium.png',img_rgb)

    template_large = cv2.imread(category_large,0) #reading template
    w, h = template_large.shape[::-1] 

    res_large = cv2.matchTemplate(img_gray,template_large,cv2.TM_CCOEFF_NORMED) 
    count_large = 0
    loc = np.where( res_large >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        count_large+=1
    
    cv2.imwrite('res_large.png',img_rgb)

    Total = count_small + count_mid + count_large # counting total number of trees categorized

    return [count_small, count_mid, count_large, Total]

def compare(temp_list, seg_list):
    '''
    Author: 
    Purpose: 
    Inputs:
    Returns:
    '''
    #template <- #input the results from Ales program
    #segmentation <- #input the results from Victoria program
    template = temp_list[-1]
    segmentation = seg_list[-1]
    print("The Template Matching program counted ", template, " trees in the image.")
    print("The Segmentation program counted ", segmentation, " trees in the image.")
    print("")

    print("Is the number of trees known?")
    known = input("Input Y for yes and N for no:")
    print("")

    if known == "Y":
        difference_S = sqrt((known - segmentation)^2) 
        difference_T = sqrt((known - template)^2)
        if difference_T == 0:
            print("Template Matching has the correct result.")
        if difference_S == 0:
            print("Segmantation has the correct result.")
        if difference_T > difference_S:
            print("Segmenation has the more accurate count.")
            difference = known - segmantation
            print("Segmentation was", difference, "trees off to actual amount.")
        if difference_T < difference_S:
            print("Template Matching has the more accurate count.")
            difference = known - template
            print("Template Matching was", difference, "trees off to actual amount.")

    if known == "N"
        print("What is the estimate trees in the image?")
        print("Remember to press enter after number is inputed.")
        estimate = int(input(" estimate:"))
        est_dif_S = sqrt((estimate - segmentation)^2)
        est_dif_T = sqrt((estimate - templat)^2)
        if est_dif_S == 0:
            print("Segmentation has the same result as your estimate.")
        if est_dif_T == 0:
            print("Template Matching has the same result as your estimate.")
        if est_dif_T > est_dif_S:
            print("Segmentation has the closest result to your estimate.")
            est_dif = estimate - segmentation
            print("Segmantation was ", est_dif, " trees off of estimate amount")
        if est_dif_T < est_dif_S:
            print("Template Matching has the closest result to your estimate.")
            est_dif = estimate - template
            print("Templat Matching was ", est_dif, " trees off of estimate amount")

def error_check(filename):
    '''
    Author: 
    Purpose: 
    Inputs:
    Returns:
    '''
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

def main():
    '''
    Author: Waskow
    Purpose: 
    Inputs:
    Returns:
    '''
    path = os.getcwd()
    print(path)
    img = Image.open('../Tutorial/Tozeur/Chabbat.png')
    print("Enhancing image...")
    contrast = 60
    adj_img = adj_contrast(img, contrast, show = True)
    print("categorizing")
    categorization(filename)
    # need to adapt to selected templates/ undo hardcoding
    print("Compiling results...")
    see_results = False
    image = gettestimg(see_results)
    #image = get_img(see_results)
    select_temp(image)
    compare(temp_list, seg_list)
    error_check(filename)
    plt.show()
    print("Program ended.")

if __name__ == "__main__":
    main()