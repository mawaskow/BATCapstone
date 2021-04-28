#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rollympoyi
"""

import os

path = os.getcwd()

print(path)
# check your current working directory to be sure the necessary files are there
# /Users/rollympoyi/Downloads/BATCapstone-main

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('Tutorial/Tozeur/Chabbat.png')  # original region image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) # grey-scaling original image
template_small = cv2.imread('Tutorial/Tozeur/Template1.png',0) # Choosing the template which corresponds to choosing the category of trees we are interested in 
w, h = template_small.shape[::-1] 

res_small = cv2.matchTemplate(img_gray,template_small,cv2.TM_CCOEFF_NORMED) 
threshold = 0.70  # creating a threshold
count_small = 0   # counting number of matches
loc = np.where( res_small >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)# creating rectangle for the region of template match
    count_small+=1
cv2.imwrite('res_small.png',img_rgb) # create file with the highlighted trees corresponding to the category 
print("The total number of trees categorized as small is", count_small) #Automatic count of the numbers of trees of that category in the region


img_rgb = cv2.imread('Tutorial/Tozeur/Chabbat.png') 
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template_medium = cv2.imread('Tutorial/Tozeur/Template3.png',0) 
w, h = template_medium.shape[::-1] 

res_medium = cv2.matchTemplate(img_gray,template_medium,cv2.TM_CCOEFF_NORMED) 
threshold = 0.70 
count_mid = 0
loc = np.where( res_medium >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    count_mid+=1
    
cv2.imwrite('res_medium.png',img_rgb)
print("The total number of trees categorized as average is", count_mid)




mg_rgb = cv2.imread('Tutorial/Tozeur/Chabbat.png') 
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template_large = cv2.imread('Tutorial/Tozeur/Template4.png',0) 
w, h = template_large.shape[::-1] 

res_large = cv2.matchTemplate(img_gray,template_large,cv2.TM_CCOEFF_NORMED) 
threshold = 0.70  # creating a threshold 
count_large = 0
loc = np.where( res_large >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    count_large+=1
    
cv2.imwrite('res_large.png',img_rgb)
print("The total number of trees categorized as large is", count_large)

Total = count_small + count_mid + count_large
print("The total number of trees categorized are", Total) #Total number of trees recognized by the program to be part of the categories selected


