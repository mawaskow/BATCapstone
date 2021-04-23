# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:40:06 2021

@author: Victoria Lidnsey 
"""
import math

template <- #input the results from Ales program
segmentation <- #input the results from Victoria program

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
        print("Segmantation was ", difference, " trees off to actual amount.")
    if difference_T < difference_S:
        print("Template Matching has the more accurate count.")
        difference = known - template
        print("Template Matching was ", difference, " trees off to actual amount.")

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

#I cant think of anything else we can do besides this... 



