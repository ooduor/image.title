#!/usr/bin/env python
from pprint import pprint
import cv2
import numpy as np
from pythonRLSA import rlsa
from scipy import stats
import math
import pytesseract
from PIL import Image
from utils import top_chunk

image = cv2.imread('./dds-89395-page-8.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
(thresh, im_bw) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # converting to binary image

cv2.imwrite('binary_gray.png', im_bw)

mask = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
mask_content = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
(contours, _) = cv2.findContours(~im_bw, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
heights = [cv2.boundingRect(contour)[3] for contour in contours] # collecting heights of each contour
avgheight = sum(heights)/len(heights) # average height

# finding the larger text
for idx, contour in enumerate(contours):
    [x,y,w,h] = cv2.boundingRect(contour)
    # cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
    if h > 2*avgheight:
        cv2.drawContours(mask, [contour], -1, 0, -1) # heading like contours
    else:
        cv2.drawContours(mask_content, [contour], -1, 0, -1) # everything else not heading-like

cv2.imshow('contour', image) # on original image
cv2.imwrite('contours.png', image)

cv2.imshow('title', mask) # on original image
cv2.imwrite('title.png', mask)


cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)