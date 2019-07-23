import cv2
from typing import List, Dict
import numpy as np

minLineLength = 100
maxLineGap = 50

def lines_extraction(gray: List[int]) -> List[int]:
    """
    this function extracts the lines from the binary image. Cleaning process.
    """
    edges = cv2.Canny(gray, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
    return lines

def top_chunk(widths, chunks=2):
    """
    Extract the top quadrant of a sorted list
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    """
    avg = len(widths) / float(chunks)
    out = []
    last = 0.0

    while last < len(widths):
        out.append(widths[int(last):int(last + avg)])
        last += avg

    return out[chunks-1]

def determine_precedence(contour, cols, avgwidth, leftmost_x):
    """
    Sort contours by distance from...
    https://stackoverflow.com/questions/39403183/python-opencv-sorting-contours
    """
    tolerance_factor = 10
    [x,y,w,h] = cv2.boundingRect(contour)
    i = 1
    col_loc = None
    while i <= cols:
        # for the first loop only, offset with beginning of first title
        if i == 1:
            avgwidth = avgwidth + leftmost_x

        if x <= avgwidth:
            col_loc = ((x // tolerance_factor) * tolerance_factor) * i + y
        i = i + 1
        avgwidth = avgwidth*i

    if not col_loc: # if wasn't within any of the columns put it in the last one atleast
        col_loc = ((x // tolerance_factor) * tolerance_factor) * cols + y

    return col_loc

def greater(a, b):
    """
    Contours sorter!
    https://stackoverflow.com/questions/27152698/sorting-contours-left-to-right-in-python-opencv/27156873#27156873
    """
    momA = cv2.moments(a)
    (xa,ya) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

    momB = cv2.moments(b)
    (xb,yb) = int(momB['m10']/momB['m00']), int(momB['m01']/momB['m00'])
    if xa>xb:
        return 1

    if xa == xb:
        return 0
    else:
        return -1