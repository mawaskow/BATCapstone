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

def main():
    img = Image.open('../Tutorial/Tozeur/Chabbat.png')
    print("Enhancing image with contrast transform...")
    adj_img = adj_contrast(img, 100, True)
    print("Conducting basic Hough Transform...")
    hough_straight(adj_img, 0.25, True)
    print("Conducting Probabilistic Hough Transform...")
    hough_prob(adj_img, 10, 100, 5, True)
    plt.show()
    print("Program ended.")

if __name__ == "__main__":
    main()