#!/usr/bin/env python
from pprint import pprint
import cv2
import numpy as np
from pythonRLSA import rlsa
import math
import pytesseract
from PIL import Image

image = cv2.imread('./dds-89395-page-8.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
(thresh, im_bw) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # converting to binary image

cv2.imshow('binary_gray', im_bw)
cv2.imwrite('binary_gray.png', im_bw)

# image_rlsa = rlsa.rlsa(im_bw, True, False, 10) # horizontal
# image_rlsa = rlsa.rlsa(im_bw, False, True, 10) # vertical
# image_rlsa = rlsa.rlsa(im_bw, True, True, 8) # both
# cv2.imshow('binary_gray_horizontal', image_rlsa)
# cv2.imwrite('binary_gray_horizontal.png', image_rlsa)

mask = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
mask_content = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
(contours, _) = cv2.findContours(~im_bw, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
heights = [cv2.boundingRect(contour)[3] for contour in contours] # collecting heights of each contour
widths = [cv2.boundingRect(contour)[2] for contour in contours] # collecting heights of each contour
avgheight = sum(heights)/len(heights) # average height
avgwidth = sum(widths)/len(widths) # average width

# finding the larger text
for idx, contour in enumerate(contours):
    [x,y,w,h] = cv2.boundingRect(contour)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
    if h > 2*avgheight:
        cv2.drawContours(mask, [contour], -1, 0, -1)
    else:
        cv2.drawContours(mask_content, [contour], -1, 0, -1) # everything else not heading-like

cv2.imshow('contour', image) # on original image
cv2.imwrite('contours.png', image)

cv2.imshow('mask', mask_content) # on mask
cv2.imwrite('mask_content.png', mask_content)

x, y = mask.shape # image dimensions

value = max(math.ceil(x/100),math.ceil(y/100))+10
value = 10
mask = rlsa.rlsa(mask, True, False, value) #rlsa application

cv2.imshow('mask-fin', mask)
cv2.imwrite('mask-fin.png', mask)

(contours, _) = cv2.findContours(~mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

mask2 = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    if w > 0.60*image.shape[1]:# width heuristic applied
        title = image[y: y+h, x: x+w]
        mask2[y: y+h, x: x+w] = title # copied title contour onto the blank image
        image[y: y+h, x: x+w] = 255 # nullified the title contour on original image

# title = pytesseract.image_to_string(Image.fromarray(mask2))
# content = pytesseract.image_to_string(Image.fromarray(image))
# print('title - {0}, content - {1}'.format(title, content))

# cv2.imshow('title', mask2)
# cv2.imwrite('title.png', mask2)
# cv2.imshow('content', image)
# cv2.imwrite('content.png', image)