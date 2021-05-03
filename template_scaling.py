'''
@mawaskow
Description:
This program allows a template to be scaled as it tries to match
with image features in an algorithm.

References:
https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
'''
import tkinter
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename

import numpy as np
import imutils
import glob
import cv2

# construct the argument parser and parse the arguments
#print("Parsing arguments...")
#ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", required=True, help='../Tutorial/Tozeur/Template1.png')
#ap.add_argument("-i", "--images", required=True, help='../Tutorial/Tozeur/Chabbat.png')
#args = vars(ap.parse_args())

def get_img_cv2(show = False):
    '''
    Prompts user to select the file they want to open.
    Returns the image to be analyzed.
    '''
    infile= askopenfilename()
    img = cv2.imread(infile)
    im_tot = np.asarray(img)
    if show:
        plt.figure()
        plt.imshow(im_tot[:,:,0])
        plt.title("Image to be Analyzed")
        plt.show()
    return img

# load the image image, convert it to grayscale, and detect edges
# loop over the images to find the template in

def scale_match(image, template, vis = False):
	#template = cv2.Canny(template, 50, 200)
	(tH, tW) = template.shape[:2]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None
	# loop over the scales of the image
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])
		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break
		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		# check to see if the iteration should be visualized
		if vis:
			# draw a bounding box around the detected region
			clone = np.dstack([edged, edged, edged])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualize", clone)
			cv2.waitKey(0)
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

def main():
	print("Loading image...")
	template = get_img_cv2()
	print("Converting image to grayscale...")
	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	print("Detecting edges...")
	image = get_img_cv2()
	print("Searching for template matches...")
	scale_match(image, template, True)
	print("Program ended.")

if __name__ == "__main__":
	main()

'''
ImagenTotal = np.asarray(Image.open('../Tutorial/Tozeur/Chabbat.png'))
trees = ImagenTotal[:,:,0]

ImagenTemplateSmall = np.asarray(Image.open('../Tutorial/Tozeur/Template1.png'))
ImagenTemplateMedium = np.asarray(Image.open('../Tutorial/Tozeur/Template2.png'))
ImagenTemplateLarge = np.asarray(Image.open('../Tutorial/Tozeur/Template3.png'))
ImagenTemplateExtra = np.asarray(Image.open('../Tutorial/Tozeur/Template4.png'))
'''

'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
img2 = img.copy()
template = cv.imread('template.jpg',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
'''