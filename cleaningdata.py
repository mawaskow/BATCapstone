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
from skimage.feature import canny, match_template
from skimage.draw import line
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

# function definitions

def adj_contrast(img, clvl, show = False):
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

def hough_straight(img, precis, show= False):
    ImagenTotal = np.asarray(img)
    image = ImagenTotal[:,:,0]
    tested_angles = np.linspace(-np.pi*precis, np.pi*precis, 360, endpoint=False)
    h, theta, d = hough_line(image)  # additional hough_line() argument: theta=tested_angles
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        ax = axes.ravel()
        ax[0].imshow(image, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()
        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step), np.rad2deg(theta[-1] + angle_step), d[-1] + d_step, d[0] - d_step]
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

from skimage.transform import probabilistic_hough_line

def hough_prob(img, thres, linlen, lingap, show = False):
    '''
    probabilistic_hough_line(image, threshold=10, line_length=50, line_gap=10, theta=None, seed=None)
    line_length: min accepted length of detected lines
    line_gap: max gap btwn px to still form line. Increase to merge broken lines more aggressively
    returns list containing lines identified as point start and end ((x0, y0), (x1, y1))
    '''
    ImagenTotal = np.asarray(img)
    image = ImagenTotal[:,:,0]
    edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=thres, line_length=linlen, line_gap=lingap)
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
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

def listapuntos(result):
    '''
    For dem_improv fxn
    '''
    xlist = []
    ylist = []
    for punto in range(np.shape(result)[1]):
        xlist.append(result[1][punto])
        ylist.append(result[0][punto])
    return xlist, ylist

def dem_improv(img, cimg, tol, clvl, show = False):
    '''
    This function is meant to demonstrate the improvement of the algorithm.
    '''
    ImagenTotal = np.asarray(img)
    ImagenContTotal = np.asarray(cimg)
    # load test templates as array
    imgtempsmall = Image.open('../Tutorial/Tozeur/Template1.png')
    imgtempmed = Image.open('../Tutorial/Tozeur/Template2.png')
    imgtemplrg = Image.open('../Tutorial/Tozeur/Template3.png')
    imgtempxtra = Image.open('../Tutorial/Tozeur/Template4.png')
    #
    ImagenTemplateSmall = np.asarray(imgtempsmall)
    ImagenTemplateMedium = np.asarray(imgtempmed)
    ImagenTemplateLarge = np.asarray(imgtemplrg)
    ImagenTemplateExtra = np.asarray(imgtempxtra)
    # convert to single band
    imagen = ImagenTotal[:,:,1]
    SmallTrees =ImagenTemplateSmall[:,:,2]
    MedTrees = ImagenTemplateMedium[:,:,2]
    LrgTrees = ImagenTemplateLarge[:,:,2]
    ExLrgTrees = ImagenTemplateExtra[:,:,2]
    # run matching
    resultsmall = match_template(imagen, SmallTrees)
    resultsmallquery = np.where(resultsmall>tol)
    resultmedium = match_template(imagen, MedTrees)
    resultmediumquery = np.where(resultmedium>tol)
    resultlarge = match_template(imagen, LrgTrees)
    resultlargequery = np.where(resultlarge>tol)
    resultextra = match_template(imagen, ExLrgTrees)
    resultextraquery = np.where(resultextra>tol)
    # Adjust contrast of template images in addition to original image
    ImagenTemplateSmall_c = np.asarray(adj_contrast(imgtempsmall, clvl))
    ImagenTemplateMedium_c = np.asarray(adj_contrast(imgtempmed, clvl))
    ImagenTemplateLarge_c = np.asarray(adj_contrast(imgtemplrg, clvl))
    ImagenTemplateExtra_c = np.asarray(adj_contrast(imgtempxtra, clvl))
    #
    imagen_c = ImagenContTotal[:,:,1]
    SmallTrees_c =ImagenTemplateSmall_c[:,:,2]
    MedTrees_c = ImagenTemplateMedium_c[:,:,2]
    LrgTrees_c = ImagenTemplateLarge_c[:,:,2]
    ExLrgTrees_c = ImagenTemplateExtra_c[:,:,2]
    # run matching
    resultsmall_c = match_template(imagen_c, SmallTrees_c)
    resultsmallquery_c = np.where(resultsmall_c>tol)
    resultmedium_c = match_template(imagen_c, MedTrees_c)
    resultmediumquery_c = np.where(resultmedium_c>tol)
    resultlarge_c = match_template(imagen_c, LrgTrees_c)
    resultlargequery_c = np.where(resultlarge_c>tol)
    resultextra_c = match_template(imagen_c, ExLrgTrees_c)
    resultextraquery_c = np.where(resultextra_c>tol)
    #show the interpreted results 
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        ax[0].plot(listapuntos(resultsmallquery)[0], listapuntos(resultsmallquery)[1], 'o', 
                markeredgecolor='g', markerfacecolor='none', markersize=5, label="Palm Tree - Type 1")
        ax[0].plot(listapuntos(resultmediumquery)[0], listapuntos(resultmediumquery)[1], 'o', 
                markeredgecolor='r', markerfacecolor='none', markersize=5, label="Palm Tree - Type 2")
        ax[0].plot(listapuntos(resultlargequery)[0], listapuntos(resultlargequery)[1], 'o', 
                markeredgecolor='b', markerfacecolor='none', markersize=5, label="Palm Tree - Type 3")
        ax[0].plot(listapuntos(resultextraquery)[0], listapuntos(resultextraquery)[1], 'o', 
                markeredgecolor='y', markerfacecolor='none', markersize=5, label="Palm Tree - Type 4")
        ax[0].imshow(ImagenTotal[10:-10,10:-10,:])
        ax[1].plot(listapuntos(resultsmallquery_c)[0], listapuntos(resultsmallquery_c)[1], 'o', 
                markeredgecolor='g', markerfacecolor='none', markersize=5, label="Palm Tree - Type 1")
        ax[1].plot(listapuntos(resultmediumquery_c)[0], listapuntos(resultmediumquery_c)[1], 'o', 
                markeredgecolor='r', markerfacecolor='none', markersize=5, label="Palm Tree - Type 2")
        ax[1].plot(listapuntos(resultlargequery_c)[0], listapuntos(resultlargequery_c)[1], 'o', 
                markeredgecolor='b', markerfacecolor='none', markersize=5, label="Palm Tree - Type 3")
        ax[1].plot(listapuntos(resultextraquery_c)[0], listapuntos(resultextraquery_c)[1], 'o', 
                markeredgecolor='y', markerfacecolor='none', markersize=5, label="Palm Tree - Type 4")
        ax[1].imshow(ImagenContTotal[10:-10,10:-10,:])
        plt.legend(loc='upper center', bbox_to_anchor=(1, 0.5))
        plt.savefig('./Outputs/cleaned_tree_class_tol'+ str(tol)+ '.png', dpi=300, bbox_inches='tight')
    # classification library
    result_types = [resultsmallquery, resultmediumquery, resultlargequery, resultextraquery]
    result_types_c = [resultsmallquery_c, resultmediumquery_c, resultlargequery_c, resultextraquery_c]
    key_nms = ["small", "medium", "large", "extra"]
    count_dct = {}
    for i in range(len(result_types)):
        x,y = listapuntos(result_types[i])
        count_dct[key_nms[i]] = len(x)
    total = 0
    for i in count_dct.keys():
        total += count_dct[i]
    count_dct["total"] = total
    print("Original Conditions:", count_dct)
    count_dct_c = {}
    for i in range(len(result_types_c)):
        x,y = listapuntos(result_types_c[i])
        count_dct_c[key_nms[i]] = len(x)
    total = 0
    for i in count_dct_c.keys():
        total += count_dct_c[i]
    count_dct_c["total"] = total
    print("Contrast Adjustment:", count_dct_c)

def main():
    img = Image.open('./Tutorial/Tozeur/Chabbat.png')
    print("Enhancing image with contrast transform...")
    contrast = 60
    adj_img = adj_contrast(img, contrast, show = True)
    '''
    print("Conducting basic Hough Transform...")
    hough_straight(adj_img, precis = 0.25, show = True)
    print("Conducting Probabilistic Hough Transform...")
    hough_prob(adj_img, thres=10, linlen=100, lingap=5, show =True)
    '''
    print("Compiling results...")
    dem_improv(img, adj_img, tol = 0.7, clvl = contrast, show = True)
    plt.show()
    print("Program ended.")

if __name__ == "__main__":
    main()
