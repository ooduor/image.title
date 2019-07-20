#!/usr/bin/env python
from typing import List, Dict
import cv2
import numpy as np
import os
import math
import glob
import pytesseract
from PIL import Image
import sys
import requests
import re
from pythonRLSA import rlsa

from utils import determine_precedence

minLineLength = 100
maxLineGap = 50

def lines_extraction(gray: List[int]) -> List[int]:
    """
    this function extracts the lines from the binary image. Cleaning process.
    """
    edges = cv2.Canny(gray, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
    return lines

# image = cv2.imread('./dds-89395-page-8.png') #reading the image (dev copy)
# image = cv2.imread('./dds-89407-page-8.png') #reading the image
image = cv2.imread('./dds-89412-page-8.png') #reading the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
(thresh, im_bw) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # converting to binary image
im_bw = ~im_bw

mask_titles = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
mask_contents = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image for content
lines = lines_extraction(gray) # line extraction

try:
    for line in lines:
        """
        drawing extracted lines on mask
        """
        x1, y1, x2, y2 = line[0]
        cv2.line(mask_titles, (x1, y1), (x2, y2), (0, 255, 0), 3)
except TypeError:
    pass
(contours, _) = cv2.findContours(im_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]
avgArea = sum(areas)/len(areas)
for c in contours:
    if cv2.contourArea(c)>60*avgArea:
        cv2.drawContours(mask_titles, [c], -1, 0, -1)

im_bw = cv2.bitwise_and(im_bw, im_bw, mask=mask_titles) # nullifying the mask over binary (toss images)

mask_titles = np.ones(image.shape[:2], dtype="uint8") * 255 # create the blank image
(contours, _) = cv2.findContours(im_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
heights = [cv2.boundingRect(contour)[3] for contour in contours]
avgheight = sum(heights)/len(heights)

# finding the larger text
for c in contours:
    [x,y,w,h] = cv2.boundingRect(c)
    if h > 2*avgheight:
        cv2.drawContours(mask_titles, [c], -1, 0, -1)
    else:
        cv2.drawContours(mask_contents, [c], -1, 0, -1)

cv2.imshow('mask_titles', mask_titles)
cv2.imwrite('mask_titles.png', mask_titles)
cv2.imshow('mask_contents', mask_contents)
cv2.imwrite('mask_contents.png', mask_contents)

x, y = mask_titles.shape # image dimensions

value = max(math.ceil(x/100),math.ceil(y/100))+20
rlsa_titles_mask = rlsa.rlsa(mask_titles, True, False, value) #rlsa application
rlsa_titles_mask_ = rlsa.rlsa(mask_titles, True, False, value) #rlsa application
cv2.imshow('rlsa_title_mask', rlsa_titles_mask)
cv2.imwrite('rlsa_title_mask.png', rlsa_titles_mask)

value = max(math.ceil(x/100),math.ceil(y/100))+20
rlsa_contents_mask = rlsa.rlsa(mask_contents, False, True, value) #rlsa application
cv2.imshow('rlsa_contents_mask', rlsa_contents_mask)
cv2.imwrite('rlsa_contents_mask.png', rlsa_contents_mask)

# Total of regions
total_columns = int(image.shape[1]/378)

(contours, _) = cv2.findContours(~rlsa_titles_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda contour:determine_precedence(contour, total_columns))
title_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
# for idx, contour in enumerate(contours):
for idx in range(len(contours)):
    [x, y, w, h] = cv2.boundingRect(contours[idx])
    # apply some heuristic to different other stranger things masquerading as titles
    if w*h > 1500: # remove tiny contours the dirtify the image
        # cv2.drawContours(title_mask, [c], -1, 0, -1)
        # cv2.rectangle(title_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
        title = image[y: y+h, x: x+w]
        title_mask[y: y+h, x: x+w] = title # copied title contour onto the blank image
        image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
        # cv2.putText(title_mask, "#{},x{},y{}".format(idx, x, y), cv2.boundingRect(contours[idx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

(contours, _) = cv2.findContours(~rlsa_contents_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
content_contours = sorted(contours, key=lambda contour:determine_precedence(contour, total_columns))
contents_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
# for idx, contour in enumerate(contours):
for idx in range(len(content_contours)):
    [x, y, w, h] = cv2.boundingRect(contours[idx])
    # apply some heuristic to different other stranger things masquerading as titles
    if w*h > 1500: # remove tiny contours the dirtify the image
        # cv2.drawContours(contents_mask, [c], -1, 0, -1)
        # cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
        contents = image[y: y+h, x: x+w]
        contents_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
        image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
        # cv2.putText(contents_mask, "#{},x{},y{}".format(idx, x, y), cv2.boundingRect(contours[idx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

# the final act!!!
(contours, _) = cv2.findContours(~rlsa_titles_mask_,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda contour:determine_precedence(contour, total_columns))

article_complete = False
title_lines_count = 0
article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for one article
# for idx, contour in enumerate(contours):
for idx, (prev, curr) in enumerate(zip(contours[::],contours[1::])):
    # https://www.quora.com/How-do-I-iterate-through-a-list-in-python-while-comparing-the-values-at-adjacent-indices/answer/Jignasha-Patel-14
    if article_complete:
        article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for one article
    [px, py, pw, ph] = cv2.boundingRect(prev)
    [cx, cy, cw, ch] = cv2.boundingRect(curr)
    # apply some heuristic to differentiate other stranger things masquerading as titles
    if pw*ph > 1500: # remove tiny contours the dirtify the image
        if (cy-py) > 0: # EITHER SMALL OR BIG GAP DETECTED
            if (cy-py) > (ch+ph)*2: # WE HAVE A BIG GAP
                title_lines_count += 1
                # bigger gap, slot any articles in the column and close this and one article done
                # cv2.drawContours(article_mask, [c], -1, 0, -1)
                # cv2.rectangle(article_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                # loop through contents and insert any valid ones in this gap
                content_exists = False
                for idxx in range(len(content_contours)):
                    [x, y, w, h] = cv2.boundingRect(content_contours[idxx])
                    # apply some heuristic to different other stranger things masquerading as titles
                    if w*h > 1500: # remove tiny contours the dirtify the image
                        dist = cv2.pointPolygonTest(content_contours[idxx],(x,y),False)
                        if dist and (cy+ch) < (y+h) and (cx+cw) < (x+w):
                            # cv2.drawContours(contents_mask, [c], -1, 0, -1)
                            # cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                            contents = contents_mask[y: y+h, x: x+w]
                            article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                            image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                            # cv2.putText(contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]
                            content_exists = False
                            break

                # add this title into blank layer
                article_title_p = title_mask[py: py+ph, px: px+pw]
                article_mask[py: py+ph, px: px+pw] = article_title_p # copied title contour onto the blank image
                image[py: py+ph, px: px+pw] = 255 # nullified the title contour on original image

                article_title_c = title_mask[cy: cy+ch, cx: cx+cw]
                article_mask[cy: cy+ch, cx: cx+cw] = article_title_c # copied title contour onto the blank image
                image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

                article_complete = True
                # cv2.putText(article_mask, "#{},x{},x{}".format(idx, x, y), cv2.boundingRect(contours[idx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]
                title_lines_count = 0
            else: # WE HAVE A SMALL GAP
                title_lines_count += 2
                content_exists = False
                for idxx in range(len(content_contours)):
                    [x, y, w, h] = cv2.boundingRect(content_contours[idxx])
                    # apply some heuristic to different other stranger things masquerading as titles
                    if w*h > 1500: # remove tiny contours the dirtify the image
                        dist = cv2.pointPolygonTest(content_contours[idxx],(x,y),False)
                        if dist and (cy+ch) < (y+h) and (cx+cw) < (x+w):
                            # cv2.drawContours(contents_mask, [c], -1, 0, -1)
                            # cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                            contents = contents_mask[y: y+h, x: x+w]
                            article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                            image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                            # cv2.putText(contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]
                            content_exists = True
                            break

                # add this title into blank layer
                article_title_p = title_mask[py: py+ph, px: px+pw]
                article_mask[py: py+ph, px: px+pw] = article_title_p # copied title contour onto the blank image
                image[py: py+ph, px: px+pw] = 255 # nullified the title contour on original image

                article_title_c = title_mask[cy: cy+ch, cx: cx+cw]
                article_mask[cy: cy+ch, cx: cx+cw] = article_title_c # copied title contour onto the blank image
                image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

                # check if any content contours were found (like in case of last article in the column...)
                if content_exists:
                    cv2.imwrite('article_{}.png'.format(idx), article_mask)

                article_complete = False
                print(title_lines_count)
        else: # INVALID GAP ALERT!!
            title_lines_count = 0
            if (cy-py) < (ch+ph)*2: # CONFIRMED INVALD
                # create new blank and skip this loop altogether
                article_complete = True
            else:
                article_complete = True







# title = pytesseract.image_to_string(Image.fromarray(title_mask))
# content = pytesseract.image_to_string(Image.fromarray(image))

# print('title - {0}, content - {1}'.format(title, content))

cv2.imshow('title', title_mask)
cv2.imwrite('title.png', title_mask)
cv2.imshow('contents', contents_mask)
cv2.imwrite('contents.png', contents_mask)
# cv2.imshow('content', image)
# cv2.imshow('content.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()