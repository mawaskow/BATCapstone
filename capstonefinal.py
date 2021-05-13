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

##### function definitions

def gettestimg(show = False):
    '''
    Author: Waskow
    Purpose: Gets the hardcoded Chabbat image
    Parameters:
        show = [boolean] whether or not to display the images
    Returns: 
        Reference image (PIL Image.open() type)
        filename
    '''
    filename = '../Tutorial/Tozeur/Chabbat.png'
    img = Image.open(filename)
    ImagenTotal = np.asarray(img)
    trees = ImagenTotal[:,:,0]
    if show:
        plt.figure()
        plt.imshow(trees)
        plt.show()
    return img, filename

def get_img(show = False):
    '''
    Author: Waskow
    Purpose: Gets user-selected image (both reference and templates)
    Parameters: 
        show = [boolean] whether or not to display the old & new images
    Returns: 
        Selected image (PIL Image.open() type)
        infile = filename
    '''
    infile= askopenfilename()
    img = Image.open(infile)
    im_tot = np.asarray(img)
    if show:
        plt.figure()
        plt.imshow(im_tot[:,:,0])
        plt.title("Image to be Analyzed")
        plt.show()
    return img, infile

def line_select_callback(eclick, erelease):
    '''
    Author: Matplotlib reference
        https://matplotlib.org/stable/gallery/widgets/rectangle_selector.html
    Purpose: Callback for line selection.
    Parameters: 
        eclick = press event
        erelease = release event
    Returns: None
    '''
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")

def toggle_selector(event):
    '''
    Author: Matplotlib reference
        https://matplotlib.org/stable/gallery/widgets/rectangle_selector.html
    Purpose: Allows selector to be activated/deactivated
    Parameters: 
        event = key-press/click event
    Returns: None
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
    Author: Waskow
    Purpose: Allows selector to be activated/deactivated
    Parameters: 
        img = image (opened with PIL library Image.open() method)
    Returns: None
    Notes: RectangleSelector call from https://matplotlib.org/stable/gallery/widgets/rectangle_selector.html
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

def enable_multselect(img):
    '''
    Author: Waskow
    This function enables a user to select a number of templates from a display.
    '''
    print("\nA window will appear displaying your reference image. Zoom in by selecting the magnifying glass icon at the bottom of the window.")
    print("After zooming in by selecting a rectangle to zoom to, remember to click the magnifying glass icon again to enable the rectangle selector.")
    print("\nOnce you have found the area you want to save as a template, as you select it a red rectangle will appear.")
    print("You can only select one template at a time, so exit the window after selecting the rectangle so you can save the template.")
    input("\nEnd of instructions. Press enter to continue. ")
    resp = " "
    tmpimglst= []
    tmpfilelst= []
    while resp != "N":
        resp = input("\nWould you like to select a new template? (Y/N): ")
        if resp == "Y":
            select_temp(img)
            dimstr = input("\nUsing Ctrl+C to copy and Ctrl+V to paste, paste the above callback here [the form (###,###) --> (###,###)]: ")
            rectdim = dimstr.split(" ")
            x1 = int(round(float(rectdim[0][1:-1])))
            y1 = int(round(float(rectdim[1][0:-1])))
            x2 = int(round(float(rectdim[3][1:-1])))
            y2 = int(round(float(rectdim[4][0:-1])))
            im_tot = np.asarray(img)
            template = im_tot[y1:y2, x1:x2]
            im = Image.fromarray(template)
            tmpimglst.append(im)
            filenm = asksaveasfilename()
            im.save(filenm)
            tmpfilelst.append(filenm)
        elif resp != "N" and resp != "Y":
            print("Input not recognized.")
    return tmpimglst, tmpfilelst

def adj_contrast(img, clvl, show = False):
    '''
    Author: Waskow
    Purpose: Adjusts contrast to enhance & clean image
    Parameters: 
        img = image (opened with PIL library Image.open() method)
        clvl = [integer]
        show = [boolean] whether or not to display the old & new images
    Returns: contrast-adjusted image (image type)
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

def segmentation(img):
    '''
    Author: Lindsey
    Purpose: Completes segmentation method
    Parameters:
        img = image (opened with PIL library Image.open() method)
    Returns: 
        segmentation = 
        labeled_trees = 
    '''
    Image_1 = np.asarray(img)
    tree = Image_1[:, :, 0]
    for i in range(0, 3):
        hist, hist_centers = histogram(Image_1[:,:,i])
        plt.figure(figsize = (10,10))
        plt.plot(hist_centers, hist)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Image Values')
        plt.show()
    print("\nAfter looking at the histogram detemine the upper and the lower setting for the markers.")
    print("Press enter when finished")
    lower = int(input("\nLower Bound: "))
    upper = int(input("\nUpper Bound: "))
    if lower >250 or lower < 0:
        response = input("\nUnusable input for Lower Bound. \nWould you like to retry? (Y/N): ")
        if response == "Y":
            lower = int(input("\nLower Bound: ")) 
    if upper > 250 or upper < 0:
        response = input("\nUnusable input for Upper Bound. \nWould you like to retry? (Y/N): ")
        if response == "Y":
            upper = int(input("\nUpper Bound: "))                      
    
    edge = canny(tree/600)
    fill_tree = ndi.binary_fill_holes(edge)
    label_objects, nb_labels = ndi.label(fill_tree)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 5
    mask_sizes[0] = 0
    trees_cleaned = mask_sizes[label_objects]

    count = len(sizes)
    print("The number of date palms is:" + str(count))
    
    plt.figure(figsize = (10,10))
    plt.imshow(fill_tree)
    plt.title("Fill tree")
    plt.show()

    plt.figure(figsize = (10,10))
    plt.imshow(trees_cleaned)
    plt.title("Trees Cleaned")
    plt.show()

    elevation_map = sobel(Image_1)
    plt.figure(figsize = (10,10))
    plt.imshow(elevation_map)
    plt.show()

    #Chabbat: upper = 180, lower = 110
    #PalmTreesOasis1: upper = 80, lower = 100
    #Shakmo: upper = 100, lower = 75

    markers = np.zeros_like(Image_1)
    markers[Image_1 < lower] = 1
    markers[Image_1 > upper] = 2
    
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_trees, _ = ndi.label(segmentation)

    plt.figure(figsize = (10,10))
    plt.imshow(segmentation[:,:,0])
    plt.show()

    return segmentation, labeled_trees, count

def categorization(filename, templatenames, threshold = 0.7):
    '''
    Author: Mpoyi
    Purpose: 
    Parameters:
        filename
        threshold = [float] template matching threshold between 0 and 1
    Returns:
    Notes:
    Waskow adaptations to code only modify input structure slightly, so only 3 templates readable from lists so far.
    '''
    ntemp = len(templatenames)
    # original image
    img_rgb = cv2.imread(filename)
    #plt.imshow(img_rgb)
    # grey-scaling original image
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    counts = []
    for i in range(ntemp):
        template = cv2.imread(templatenames[i], 0)
        #plt.imshow(template)
        w, h = template.shape[::-1]
        #w = len(template[:,0])
        #h = len(template[0,:])
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        count = 0
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            count+=1
        cv2.imwrite('res_'+str(i)+'.png',img_rgb)
        counts.append(count)
    # counting total number of trees categorized
    Total = 0
    for i in counts:
        Total += i
    counts.append(Total)
    print(counts)
    return counts

def compare(temp_list, seg_list):
    '''
    Author: Lindsey
    Purpose: 
    Parameters:
        temp_list = the results from template-matching
        seg_list = the results from segmentation
    Returns:
    '''
    template = temp_list[-1]
    segmentation = seg_list[0]
    print("\nThe Template Matching program counted", template, "trees in the image.")
    print("The Segmentation program counted", segmentation, "trees in the image.")

    print("\nIs the number of trees known?")
    known = input("Input Y for yes and N for no: ")

    if known == "Y":
        difference_S = math.sqrt((known - segmentation)^2) 
        difference_T = math.sqrt((known - template)^2)
        if difference_T == 0:
            print("\nTemplate Matching has the correct result.")
        if difference_S == 0:
            print("\nSegmantation has the correct result.")
        if difference_T > difference_S:
            print("\nSegmenation has the more accurate count.")
            difference = known - segmentation
            print("\nSegmentation was", difference, "trees off to actual amount.")
        if difference_T < difference_S:
            print("\nTemplate Matching has the more accurate count.")
            difference = known - template
            print("\nTemplate Matching was", difference, "trees off to actual amount.")

    if known == "N":
        print("\nWhat is the estimate trees in the image?")
        print("Press enter after number is inputed.")
        estimate = int(input("\nEstimate: "))
        est_dif_S = math.sqrt((estimate - segmentation)^2)
        est_dif_T = math.sqrt((estimate - template)^2)
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

def error_check(filename, lower, upper):
    '''
    Author: Mpoyi
    Purpose: 
    Parameters:
        filename = [string]
        lower = [integer] lower marker
        upper = higher marker
    Returns:
    '''
    image = np.asarray(Image.open(filename))

    elevation_map = sobel(image)
    markers = np.zeros_like(image)
    # lower marker for Chabbat
    markers[image < lower] = 1
    # higher marker for Chabbat
    markers[image > upper] = 2
    im_true = watershed(elevation_map, markers)
    im_true = ndi.label(ndi.binary_fill_holes(im_true - 1))[0]

    edges = sobel(image)
    im_test1 = watershed(edges, markers=468, compactness=0.001)

    # segmentation method used in ur program
    method_names = ['Compact watershed']
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

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
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

    plt.show()

def main():
    '''
    Author: Waskow
    Purpose: To run the algorithm
    Parameters: None
    Returns: None
    '''
    input("\nPlease select reference image in the next prompt.\nPress enter to continue. ")
    img, path = get_img(show = False)

    resp = input("\nDo you have template images for matching? (Y/N): ")
    if resp == "Y"or resp == "y":
        tmpimglst = []
        tmpfilelst = []
        cont = " "
        while cont != "N":
            cont = input("\nWould you like to upload a template image? (Y/N): ")
            if cont == "Y":
                pretemp, prefile = get_img(show = False)
                tmpimglst.append(pretemp)
                tmpfilelst.append(prefile)
            elif cont != "N" and cont != "Y":
                print("Input not recognized.")
    elif resp == "N" or resp == "n":
        # this section/function will need more development
        # in fxn, need to save images as pngs, then return the filenames so it will match up with Rolly's program
        tmpimglst, tmpfilelst = enable_multselect(img)
    #print(tmpfilelst)
    contrast = int(input("\nEnter the desired contrast adjustment level (suggestion = 60): "))
    print("Applying contrast adjustment...")
    adj_img = adj_contrast(img, contrast, show = True)
    adj_fname = path[0:-4]+"_adj.png"
    adj_img.save(adj_fname)
    tmpimglst_cont = []
    tmpfilelst_cont = []
    for i in range(len(tmpimglst)):
        tempimge = adj_contrast(tmpimglst[i], contrast, show = False)
        tmpimglst_cont.append(tempimge)
        tempadjn = tmpfilelst[i][0:-4]+"_adj.png"
        tmpfilelst_cont.append(tempadjn)
        tempimge.save(tempadjn)
    #print(tmpfilelst_cont)
    print("Conducting categorization...")
    countslst = categorization(adj_fname, tmpfilelst_cont, threshold = 0.7)
    var1, var2, finct = segmentation(adj_img)
    #print("Compiling results...")
    compare(countslst, [finct])
    error_check(adj_fname, 110, 100)
    plt.show()
    print("Program ended.")

if __name__ == "__main__":
    main()
