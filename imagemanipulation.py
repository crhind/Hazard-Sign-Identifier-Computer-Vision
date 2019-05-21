from cv2 import cv2
import sys
import numpy
from matplotlib import pyplot
import numpy as np
import math
import collection as col
import imagepreprocessing as ip

#Set of kernels that were used on and off throughout development. 
kernels = { "prewitx" : np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
            "prewity" : np.array([[-1,-1,-1],[0,0,0],[1,1,1]]),
            "sobelx" : np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
            "sobely" : np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
            "rect3" : np.array([[0,1,0],[1,1,1],[0,1,0]]),
            "rect" : np.array([[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0]])/5,
            "line1": np.array([[1,1,0],[0,1,0],[0,1,1]]),
            "line2": np.array([[0,1,1],[0,1,0],[1,1,0]]),
            "lineA": np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]]),
            "lineB": np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])}


def alterColourSpace(image, flag):
    """[Small function to change the colourspace of the image. This was for the simplicity of calling the
        opencv function over and over again during experimenting and forgetting the syntax]
    
    Arguments:
        image  -- [The image you want the colourspace changed]
        flag  -- [the opencv flag to change between colours]
    
    Returns:
        [image] -- [image in the changed colour spaces]
    """
    return cv2.cvtColor(image, flag)

def blur(image, kernel = None):
    """[Gaussian blur taken form opencv.]
    
    Arguments:
        image  -- [image to be blurred]
    
    Keyword Arguments:
        kernel {[numpy array]} -- [the chosen kernel size to use in the blur] (default: {None})
    
    Returns:
        [image] -- [blurred image]
    """
    if kernel == None:
        kernel = 5
    return cv2.GaussianBlur(image,(kernel,kernel),0)

def perspectiveManipulation(image, imagePoints):
    """[Takes the given image and given points and tranforms the pixels contained in the points to a 
        new image of chosen size. warpMat gets the tranform matrix to be used by the function warpPerspective.]
    
    Arguments:
        image -- [the image to be transformed, it will be the image that contains all of the hazard labels.]
        imagePoints {[numpy array]} -- [the set of points that contain the hazard label to be transformed.]
    
    Returns:
        [image] -- [the new image that contains the standardised hazard label]
    """
    endPoints = np.float32([[250,0],[0,250],[250,500],[500,250]])
    warpMat = cv2.getPerspectiveTransform(imagePoints, endPoints)
    return cv2.warpPerspective(image, warpMat, (500,500))

def canny(image):
    """[Standard canny edge detection functionality. uses the opencv method Canny]
    
    Arguments:
        image -- [the input image to be transformed]
    
    Returns:
        [image] -- [the canny transformed image.]
    """
    edges = cv2.Canny(image, 150, 190, 5)
    return edges

def edgeFinding(image):
    """[The inputted image goes through multiple tranforms to find the contours and edge points of every hazard sign in the 
        image, these edge points are stored in a list and returned to the calling function for further maipulation. This function
        does not change the actual image in anyway]
    
    Arguments:
        image -- [the image containing the hazard labels]
    
    Returns:
        [array] -- [the set of points that correlate to the corners of each hazard label in the input image.]
    """
    print(image.shape)
    finalPoints =[]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = adaptiveThreshold(gray)

    gaus = cv2.GaussianBlur(thresh, (5,5), 0)
    
    bilat = bilatFilter(gaus)

    cannied = canny(bilat)    

    _, contours, heir = cv2.findContours(cannied,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    holder = []
    for contour in contours:
        epsilon = 0.1*cv2.arcLength(contour,True)
        poly = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(poly)
        if len(poly) == 4 and area > 50000:
            holder.append(poly)
    for contour in holder:
        for point in contour:
            cv2.circle(cannied, (point[0][0],point[0][1]), 3, [255,0,0], 20)
        points = np.float32([[contour[0][0][0], contour[0][0][1]], [contour[1][0][0],contour[1][0][1]], [contour[2][0][0],contour[2][0][1]], [contour[3][0][0],contour[3][0][1]]])
        finalPoints.append(points)
    return finalPoints  

# Retrieves the euclidian distance between two rectangle centres
def getCentralDistance(centerA, centerB):
    return (math.fabs(centerA[0] - centerB[0]), math.fabs(centerA[1] - centerB[1]))

# Converts the given image to YUV colourspace performs histgram equalization on the Y channel in an attempt to remove shadow.
def histEqualization(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])    
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# uses the predefined dictionary of kernels to filter the image.
def linearFilter(image, name):
    return (cv2.filter2D(image, -1, kernels[name]))

# Erodes the input image using the opencv's erode method.
def erode(image):
    return cv2.erode(image, np.ones((5,5), np.uint8))

#Performs a convoluton on the input image using a line detecting kernels. Similar to a sobel transfomation.
def lineDetect(image):
    X = linearFilter(image, "lineA")
    Y = linearFilter(image, "lineB")
    result = (np.hypot(X,Y)).astype(np.uint8)
    return result

# Dilates the image using opencv's dialte method.
def dilate(image):
    return cv2.dilate(image, np.ones((3,3),np.uint8))

# Median blur's the input image
def median(image):
    return cv2.medianBlur(image, 7)

# Uses the close flag on the morphologyEx method in opencv.
def closing(image):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

# Uses the open flag on the morphologyEx method in opencv 
def opening(image):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

# Filters the image using opencv's bilateralFilter method in opencv.
def bilatFilter(image):
    return cv2.bilateralFilter(image, 11, 110, 110)

# Perfomrs a threshold on the inut image using opencvs' threshold method
def threshold(image):
    _ ,thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

# Perfomrs an inverse threshold on the inut image using opencvs' threshold method with the inverse threshold flag.
def thresholdInv(image):
    _ ,thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresh

# Perfomrs a threshold on the inut image using opencvs' adaptiveThreshold method
def adaptiveThreshold(image):
    return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,43,1)

# Returns the cropped image from the points given in parameter input points.
def cropImage(image, points):
    return image[points[1]:points[3], points[0]:points[2]]
