from cv2 import cv2
import numpy as np
import collection as col
from matplotlib import pyplot as plt
import imagemanipulation as im

def getKeyPoints(image):
    """[takes in an image and detects the key points using cv2's ORB classifier]
    
    Arguments:
        image {[Object]} -- [the input image]
    
    Returns:
        [tuple] -- [tuple of the key points list and descriptor list]
    """
    orb = cv2.ORB_create(nfeatures=50, WTA_K=4)
    kp = orb.detect(image,None)
    kp, des = orb.compute(image, kp)
    return kp, des    

def compareKeyPoints(comparisons, image):
    """[Compares an input image to the comparisons list which contains all the predefined key points for the symbols]
    
    Arguments:
        comparisons {[List]} -- [the list of predetermined key points and descriptors for the symbols]
        image {[Objct]} -- [the input image]
    
    Returns:
        [String] -- [the classified symbol label]
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    shape = image.shape
    cropped = im.cropImage(image, [int(shape[1]*0.20),int(shape[0]*0.05),int(shape[1]*0.80),int(shape[0]*0.46)])
    imageFeatures = getKeyPoints(cropped)
    if imageFeatures[0]:
        bestMatch = [],[]
        bestValue = np.inf
        for comparitor in comparisons:
            matches = bf.match(comparitor[2],imageFeatures[1])
            sort_match = sorted(matches, key=lambda x:x.distance)
            firstFive = sort_match[:5]
            average = np.mean([x.distance for x in firstFive])
            if average < bestValue:
                bestMatch = (comparitor,matches)
                bestValue = average
        returnVal = bestMatch[0][3].split(".")[0]
    else:
        returnVal = "(none)"
    return returnVal