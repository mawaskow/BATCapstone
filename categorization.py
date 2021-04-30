#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rollympoyi
"""

import os

path = os.getcwd()

print(path)
# check your current working directory to ensure the necessary files are there


import cv2
import numpy as np

print("Enter the filename of the original image")
filename = input("File name: ")
print("Enter the filename of template for the small tree category")
category_small = input("File name: ")
print("Enter the filename of template for the average tree category")
category_medium = input("File name: ")
print("Enter the filename of template for the large tree category")
category_large = input("File name: ")

def categorization(filename):

    img_rgb = cv2.imread(filename)  # original image
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) # grey-scaling original image


    template_small = cv2.imread(category_small,0) #reading template
    w, h = template_small.shape[::-1] 

    res_small = cv2.matchTemplate(img_gray,template_small,cv2.TM_CCOEFF_NORMED) 
    threshold = 0.70  # creating a threshold
    count_small = 0  # creating counter for the matched trees
    loc = np.where( res_small >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)# creating rectangle of region of template
        count_small+=1
    cv2.imwrite('res_small.png',img_rgb) # creating output file of counted trees being part of that category
    print("The total number of trees categorized as small is", count_small)


    
    template_medium = cv2.imread(category_medium,0) #reading template
    w, h = template_medium.shape[::-1] 

    res_medium = cv2.matchTemplate(img_gray,template_medium,cv2.TM_CCOEFF_NORMED) 
    threshold = 0.70  # creating a threshold 
    count_mid = 0
    loc = np.where( res_medium >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        count_mid+=1
    
    cv2.imwrite('res_medium.png',img_rgb)
    print("The total number of trees categorized as average is", count_mid)



    template_large = cv2.imread(category_large,0) #reading template
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

    Total = count_small + count_mid + count_large # counting total number of trees categorized
    print("The total number of trees categorized are", Total)

categorization(filename)
