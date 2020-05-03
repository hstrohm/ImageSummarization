import torch
from torchvision import models, transforms
import torchvision
from torch.utils.data import DataLoader

from pycocotools.coco import COCO

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib, os, json
import numpy as np

import cv2
import binascii
import scipy
import scipy.misc
import scipy.cluster
import webcolors

# not currently used
def show_mask(output, i, size=(10,10)):
    plt.figure(figsize=size)
    plt.imshow(output['masks'][i].cpu().squeeze())
    plt.axis('off')
    plt.show()
    
def imageSegmentationForeground(img, show=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale image
    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU)
    # Build a kernal for morphing the image
    kernel = np.ones((3, 3), np.uint8)
    # Apply the kernal to the threhold_image
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                                kernel, iterations = 10)

    # Find the background of the image
    # bg = cv2.dilate(closing, kernel, iterations = 1)

    # Find the foreground
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret, fg = cv2.threshold(dist_transform, 0.02
                            * dist_transform.max(), 255, 0)

    # Build a new image based off the detected foreground
    new_img = np.zeros_like(img)
    for i in range(0, len(fg)):
        for j in range(0, len(fg[i])):
            if fg[i][j] > 0:
                new_img[i][j] = img[i][j]

    if show:
        # Display the foreground image
        fig = plt.figure(figsize=(10,10))
        plt.imshow(fg, cmap='gray')
        plt.axis('off')

        # Display the colored version
        fig = plt.figure(figsize=(10,10))
        plt.imshow(new_img)
        plt.axis('off')
    
    return new_img



def grabCutForeground(img, show=False):
    # Plot the Before
    if show: 
        fig = plt.figure(figsize=(15,15))
        plt.imshow(img)

    # From: https://stackoverflow.com/questions/47503177/use-grabcut-algorithm-to-separate-the-saliency-areas
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Find the threshold value
    th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # Use the threshold to generate contours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea)
    cnt = cnts[-1]
    # Build the bounding box
    rect = x,y,w,h = cv2.boundingRect(cnt)
    # Generate a mask
    mask = np.ones_like(gray, np.uint8)*cv2.GC_PR_BGD
    cv2.drawContours(mask, [cnt], -1, cv2.GC_FGD, -1)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # Cut out the image
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask2  =  np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img  = img*mask2[:,:,np.newaxis]
    
    if show: 
        fig2 = plt.figure(figsize=(15,15))
        plt.imshow(img)
        plt.show()
    return img



def grabCut2(img, show=False):
    mask = np.zeros(img.shape[:2], np.uint8)   # img.shape[:2] = (400, 600)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # Need to figure out how to determine this "region of interest"
    rect = (300,120,470,350)

    # this modifies mask
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # adding additional dimension for rgb to the mask, by default it gets 1
    # multiply it with input image to get the segmented image
    img_cut = img*mask2[:,:,np.newaxis]

    if show: 
        fig = plt.figure(figsize=(10,10))
        plt.imshow(img)
        plt.axis('off')
        fig = plt.figure(figsize=(10,10))
        plt.imshow(img_cut)
        plt.axis('off')
    
    return img_cut

def getRemainingBoxesAndLabels(img, boxes, labels, masks, scores, score_threshold):
    remainingBoxes = []
    remainingLabels = []
    remainingScores = []
    remainingMasks = []
    
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        score = scores[i]
        mask = masks[i]
        
        # detect if region is completely black
        j_min = int(box[1])
        j_max = int(box[3])
        k_min = int(box[0])
        k_max = int(box[2])
        for j in range(j_min, j_max):
            for k in range(k_min, k_max):
                # had an issue where sometime j,k were Tensors(?). think i fixed it, but this is just in case.
                if type(j) != type(0) or type(k) != type(0):
                    continue
                # if it's not completely black, keep it
                if gray[j][k] != 0:
                    if box not in remainingBoxes and score >= score_threshold:
                        remainingBoxes.append(box)
                        remainingLabels.append(label)
                        remainingScores.append(score)
                        remainingMasks.append(mask)
                    j = box[3]
                    k = box[2]
            
    return (remainingBoxes, remainingLabels, remainingMasks, remainingScores)
        
def pointWithin(point, boxCorners):
    # point = (col, row) = (x, y)
    x = point[0]
    y = point[1]
    #boxCorners = [topLeft, topRight, bottomLeft, bottomRight]
    if x >= boxCorners[0][0] and x <= boxCorners[2][0] and y >= boxCorners[0][1] and y <= boxCorners[1][1]:
        return True
    return False

# Returns a list of phrases that follow the format: boxA _phrase_ boxB
def getPhraseList(boxA, boxB):
    # Corners of box A
    bATL = (boxA[0], boxA[1])
    bATR = (boxA[2], boxA[1])
    bABL = (boxA[0], boxA[3])
    bABR = (boxA[2], boxA[3])
    boxACorners = [bATL, bATR, bABL, bABR]
    
    # Corners of box B
    # (x, y) <-> (col, row)
    bBTL = (boxB[0], boxB[1])
    bBTR = (boxB[2], boxB[1])
    bBBL = (boxB[0], boxB[3])
    bBBR = (boxB[2], boxB[3])
    boxBCorners = [bBTL, bBTR, bBBL, bBBR]
    
    # Col, Row
    # (box[0], box[1]) = top left
    # (box[2], box[3]) = bottom right
    
    # boxA within boxB
    if pointWithin(bATL, boxBCorners) and pointWithin(bATR, boxBCorners) and pointWithin(bABL, boxBCorners) and pointWithin(bABR, boxBCorners):
            return ['is within', 'is in', 'is on', 'is behind', 'is in front of']
        
    # boxA surrounds boxB
    if pointWithin(bBTL, boxACorners) and pointWithin(bBTR, boxACorners) and pointWithin(bBBL, boxACorners) and pointWithin(bBBR, boxACorners):
            return ['surrounds', 'encompasses', 'is behind', 'is in front of']
    
    # cross overlap (like +)
    # Testing (A=-,B=|) or (A=|,B=-)
    topLeftCorner = (bATL[0] <= bBTL[0] and bATL[1] >= bBTL[1]) or (bBTL[0] <= bATL[0] and bBTL[1] >= bATL[1])
    topRightCorner = (bATR[0] >= bBTR[0] and bATR[1] >= bBTR[1]) or (bBTR[0] >= bATR[0] and bBTR[1] >= bATR[1])
    bottomLeftCorner = (bABL[0] <= bBBL[0] and bABL[1] <= bBBL[1]) or (bBBL[0] <= bABL[0] and bBBL[1] <= bABL[1])
    bottomRightCorner = (bABR[0] >= bBBR[0] and bABR[1] <= bBBR[1]) or (bBBR[0] >= bABR[0] and bBBR[1] <= bABR[1])
    if topLeftCorner and topRightCorner and bottomLeftCorner and bottomRightCorner:
        return ['overlaps', 'is behind', 'is in front of']
    
    # left
    # check if cols of left side of boxA are lower than cols of left side of boxB
    leftSide = (bATL[0] < bBTL[0] and bABL[0] < bBBL[0])
    # same for right side, but this means it could still overlap boxB
    rightSide = (bATR[0] <= bBTR[0] and bABR[0] <= bBBR[0])
    # test for non-overlapping left
    rightSideDirectLeft = (bATR[0] <= bBTL[0] and bABR[0] <= bBBL[0])
    if leftSide and rightSideDirectLeft:
        return ['is beside', 'is to the left of', 'is adjacent to', 'is holding']
    if leftSide and rightSide:
        return ['is beside', 'is to the left of', 'is adjacent to', 'is holding', 'overlaps']
    
    # right
    # check if cols of left side of boxA are greater than cols of left side of boxB
    leftSide = (bATL[0] >= bBTL[0] and bABL[0] >= bBBL[0])
    # test for non-overlapping right
    leftSideDirectRight = (bATL[0] >= bBTR[0] and bABL[0] >= bBBR[0])
    # same for right side, but this means it could still overlap boxB
    rightSide = (bATR[0] > bBTR[0] and bABR[0] > bBBR[0])
    if leftSideDirectRight and rightSide:
        return ['is beside', 'is to the right of', 'is adjacent to', 'is holding']
    if leftSide and rightSide:
        return ['is beside', 'is to the right of', 'is adjacent to', 'is holding', 'overlaps']
    
    # above
    topSide = (bATL[1] < bBTL[1] and bATR[1] < bBTR[1])
    bottomSide = (bABL[1] <= bBBL[1] and bABR[1] <= bBBR[1])
    bottomSideDirectAbove = (bABL[1] <= bBTL[1] and bABR[1] <= bBTR[1])
    if topSide and bottomSideDirectAbove:
        return ['is above', 'is on top of']
    if topSide and bottomSide:
        return ['is above', 'is on top of', 'overlaps']
    
    # below
    topSide = (bATL[1] >= bBTL[1] and bATR[1] >= bBTR[1])
    topSideDirectBelow = (bATL[1] >= bBBL[1] and bATR[1] >= bBBR[1])
    bottomSide = (bABL[1] > bBBL[1] and bABR[1] > bBBR[1])
    if topSideDirectBelow and bottomSide:
        return ['is below']
    if topSide and bottomSide:
        return ['is below', 'overlaps']
    
    # If this gets returned...then I didn't think about the possibilities enough.
    return None
    
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name

    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def getObjectColor(img, box, show=False):
    # crop to region of interest
    sub_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
    if show: plt.imshow(sub_img)
    # Run kmeans with 5 clusters to get color
    ar = np.asarray(sub_img)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    codes, dist = scipy.cluster.vq.kmeans(ar, 5)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)
    counts, bins = scipy.histogram(vecs, len(codes))
    index_max = scipy.argmax(counts)
    peak = codes[index_max]
    # Get the hex version of the color
    color = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    requested_colour = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    # Determine the name of the color
    actual_name, closest_name = get_colour_name(requested_colour)
    if actual_name is None:
        return closest_name
    return actual_name
    
