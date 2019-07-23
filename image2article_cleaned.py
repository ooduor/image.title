#!/usr/bin/env python
from typing import List, Dict
import cv2
import numpy as np
import os
import math
import glob
import pytesseract
from scipy import stats
from PIL import Image
import sys
import requests
import re
from pythonRLSA import rlsa

from utils import determine_precedence, lines_extraction

image = cv2.imread('./dds-89395-page-8test.png') #reading the image (dev copy)
# image = cv2.imread('./dds-89395-page-8.png') #reading the image (dev copy)
# image = cv2.imread('./dds-89407-page-8.png') #reading the image
# image = cv2.imread('./dds-89412-page-8.png') #reading the image
# image = cv2.imread('./dds-89417-page-8.png') #reading the image
# image = cv2.imread('./dds-89445-page-8.png') #reading the image
# image = cv2.imread('./dds-89475-page-8.png') #reading the image
# image = cv2.imread('./dds-89491-page-8.png') #reading the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
(thresh, im_bw) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # converting to binary image
im_bw = ~im_bw

# Total of regions
total_columns = int(image.shape[1]/378)
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

#cv2.imshow('mask_titles', mask_titles)
cv2.imwrite('mask_titles.png', mask_titles)
#cv2.imshow('mask_contents', mask_contents)
cv2.imwrite('mask_contents.png', mask_contents)

x, y = mask_titles.shape # image dimensions

value = max(math.ceil(x/100),math.ceil(y/100))+20
rlsa_titles_mask = rlsa.rlsa(mask_titles, True, False, value) #rlsa application
rlsa_titles_mask_for_final = rlsa.rlsa(mask_titles, True, False, value) #rlsa application
cv2.imwrite('rlsa_title_mask.png', rlsa_titles_mask)

value = max(math.ceil(x/100),math.ceil(y/100))+20
rlsa_contents_mask = rlsa.rlsa(mask_contents, False, True, value) #rlsa application
rlsa_contents_mask_for_avg_width = rlsa.rlsa(mask_contents, False, True, value) #rlsa application
cv2.imwrite('rlsa_contents_mask.png', rlsa_contents_mask)
cv2.imwrite('rlsa_contents_mask_for_avg_width.png', rlsa_contents_mask_for_avg_width)

# CALC AVG WIDTHS?!
(for_avgs_contours, _) = cv2.findContours(~rlsa_contents_mask_for_avg_width,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for_avgs_contours_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
contents_sum_list = []
contents_x_list = [] # to get the left-most content box
contents_length = 0
for idx, contour in enumerate(for_avgs_contours):
    [x, y, w, h] = cv2.boundingRect(contour)
    # apply some heuristic to different other stranger things masquerading as titles
    if w*h > 1500: # remove tiny contours the dirtify the image
        cv2.drawContours(for_avgs_contours_mask, [contour], -1, 0, -1)
        cv2.rectangle(for_avgs_contours_mask, (x,y), (x+w,y+h), (255, 0, 0), 5)
        cv2.putText(for_avgs_contours_mask, "#{},x{},y{},w{}".format(idx, x, y, w), cv2.boundingRect(contour)[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [0, 0, 255], 2) # [B, G, R]
        contents_sum_list.append(w)
        contents_x_list.append(x)
        contents_length += 1

trimmed_mean = int(stats.trim_mean(contents_sum_list, 0.1)) # trimmed mean
leftmost_x = min(contents_x_list)
print("Sum of list element is : ", contents_sum_list)
print("Average of list element is : ", trimmed_mean )
print("Leftmost content box is : ", min(contents_x_list) )
cv2.imwrite('for_avgs_contours_mask.png', for_avgs_contours_mask) # debug remove
threshold = 1500 # remove tiny contours that dirtify the image

(contours, _) = cv2.findContours(~rlsa_titles_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# apply some heuristic to differentiate other stranger things masquerading as titles
nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x))
title_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
# for idx, contour in enumerate(contours):
for idx in range(len(contours)):
    [x, y, w, h] = cv2.boundingRect(contours[idx])
    # cv2.drawContours(title_mask, [c], -1, 0, -1)
    # cv2.rectangle(title_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
    title = image[y: y+h, x: x+w]
    title_mask[y: y+h, x: x+w] = title # copied title contour onto the blank image
    image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
    # cv2.putText(title_mask, "#{},x{},y{},h{}".format(idx, x, y, h), cv2.boundingRect(contours[idx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 0, 0], 2) # [B, G, R]

(contours, _) = cv2.findContours(~rlsa_contents_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# apply some heuristic to different other stranger things masquerading as titles
nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

content_contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x))
contents_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
# for idx, contour in enumerate(contours):
for idx in range(len(content_contours)):
    [x, y, w, h] = cv2.boundingRect(content_contours[idx])
    # cv2.drawContours(contents_mask, [c], -1, 0, -1)
    # cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
    contents = image[y: y+h, x: x+w]
    contents_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
    image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
    # cv2.putText(contents_mask, "#{},x{},y{}".format(idx, x, y), cv2.boundingRect(contours[idx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

# the final act!!!
(contours, _) = cv2.findContours(~rlsa_titles_mask_for_final,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# apply some heuristic to different other stranger things masquerading as titles
nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x))

article_complete = False
title_lines_count = 0
article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for one article
# for idx, contour in enumerate(contours):
for idx, (_curr, _next) in enumerate(zip(contours[::],contours[1::])):
    # https://www.quora.com/How-do-I-iterate-through-a-list-in-python-while-comparing-the-values-at-adjacent-indices/answer/Jignasha-Patel-14
    if article_complete:
        article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for antother separate article
    [cx, cy, cw, ch] = cv2.boundingRect(_curr)
    [nx, ny, nw, nh] = cv2.boundingRect(_next)
    if (ny-cy) > (nh+ch)*2: # next is greater than current...
        print('Big Gap! {}'.format(idx))

        # loop through contents and insert any valid ones in this gap
        for idxx in range(len(content_contours)):
            [x, y, w, h] = cv2.boundingRect(content_contours[idxx])
            # search_area_rect = cv2.rectangle(contents_mask,(cx,cy),(x+w,y+h),(0,0,255),thickness=3,shift=0)
            dist = cv2.pointPolygonTest(content_contours[idxx],(x,y), False)
            # https://stackoverflow.com/a/50670359/754432
            if cy < y and cx-10 < x and x < (cx+w): # less than because it appears above
                # check but not greater than the next title!!
                if y > ny: # or next is another column
                    break
                # cv2.drawContours(contents_mask, [c], -1, 0, -1)
                # cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                contents = contents_mask[y: y+h, x: x+w]
                article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                # cv2.putText(contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

        article_title_p = title_mask[cy: cy+ch, cx: cx+cw]
        article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
        image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

        cv2.imwrite('article_{}big.png'.format(idx), article_mask)
        article_complete = True
    elif (ny-cy) < (nh+ch)*2 and (ny-cy) > 0 and nx <= cx+10: # next if not greater... but just small|
        print('Small Gap! {}'.format(idx))

        # handle special cases like end of the page
        if len(contours) == (idx+2): # we are on last article, it's always greater by 2 instead of one. Nkt!
            # loop through contents and insert any valid ones in this gap
            for idxx in range(len(content_contours)):
                [x, y, w, h] = cv2.boundingRect(content_contours[idxx])
                if cy-ch < y and (cy+ch) < (y+h) and (cx+cw) < (x+w): # more than because it appears above but not too above
                    # cv2.drawContours(contents_mask, [c], -1, 0, -1)
                    # cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                    contents = contents_mask[y: y+h, x: x+w]
                    article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                    image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                    # cv2.putText(contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

                    # check that it does not encounter new title in next column
                    if y > ny:
                        break

            article_title_p = title_mask[cy: cy+ch, cx: cx+cw]
            article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
            image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

            article_title_p = title_mask[ny: ny+nh, nx: nx+nw]
            article_mask[ny: ny+nh, nx: nx+nw] = article_title_p # copied title contour onto the blank image
            image[ny: ny+nh, nx: nx+nw] = 255 # nullified the title contour on original image

            cv2.imwrite('article_{}.png'.format(idx), article_mask)

        article_title_p = title_mask[cy: cy+ch, cx: cx+cw]
        article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
        image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

        article_complete = False

    elif (ny-cy) < (nh+ch)*2 and (ny-cy) < 0: # next is not greater... must be invalid
        print('Invalid Gap! {}'.format(idx))
        # loop through contents and insert any valid ones in this gap
        for idxx in range(len(content_contours)):
            [x, y, w, h] = cv2.boundingRect(content_contours[idxx])
            if cy-ch < y and cx-10 < x: # more than because it appears above but not too above
                # cv2.drawContours(contents_mask, [c], -1, 0, -1)
                # cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                contents = contents_mask[y: y+h, x: x+w]
                article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                # cv2.putText(contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

                # check that it does not encounter new title in next column
                if y > ny:
                    break

        article_title_p = title_mask[cy: cy+ch, cx: cx+cw]
        article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
        image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

        cv2.imwrite('article_{}invalid.png'.format(idx), article_mask)
        article_complete = True

    else: # must be first one with next invalid...
        print('Invalid First Gap! {}'.format(idx))
        # loop through contents and insert any valid ones in this gap
        for idxx in range(len(content_contours)):
            [x, y, w, h] = cv2.boundingRect(content_contours[idxx])
            if cy-ch < y and cx-10 < x: # more than because it appears above but not too above
                # cv2.drawContours(contents_mask, [c], -1, 0, -1)
                # cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                contents = contents_mask[y: y+h, x: x+w]
                article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                # cv2.putText(contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

                # check that it does not encounter new title in next column
                if y > ny or nx-10 > x: # its lower in the page || the next title even with 10px offset is still larger... then we are tresspassing
                    break

        article_title_p = title_mask[cy: cy+ch, cx: cx+cw]
        article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
        image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

        cv2.imwrite('article_{}invalidfirst.png'.format(idx), article_mask)
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