from PIL import Image
import pytesseract as pyt
import imagemanipulation as im
from cv2 import cv2
import imagepreprocessing as ip
import numpy as np
import collection as col
from functools import reduce
import symboldetection as sym
import difflib
import math

#List of all words that are found in the hazard labels. used for sanity check on OCR results
corpus = [  "explosive", 
            "oxygen",
            "blasting agents", 
            "flammable gas",
            "flammable liquid",
            "non-flammable gas", 
            "inhalation hazard", 
            "dangerous when wet", 
            "organic peroxide", 
            "radioactive iii", 
            "radioactive ii",
            "corrosive", 
            "spontaneously combustible", 
            "oxidizer", 
            "combustible",
            "flammable",
            "gasoline", 
            "poison", 
            "explosives",
            "toxic",
            "fuel oil"]

#Dictionary of all colours that can be found in the hazard labels, used as a lookup table for what the colours classes are from the KNN
colours = { "1" : "blue",
            "2" : "red",
            "3" : "yellow",
            "4" : "orange",
            "5" : "white",
            "6" : "green",
            "7" : "black"}

# Dictionary of all digits expected in the hazard labels along with their common mistakes. The approach of having text similarity for the digit strings would not have worked due to their short length.
digits = {  "3" : ["3", "33", "33/", "3/", "53", "33/1", "3/."],
            "6" : ["64", "6/"],
            "4" : ["4", "4/", "34"],
            "2" : ["2","24", "2/"],
            "8" : ["8", "81", "8/", "831"],
            "1" : ["1", "1/", "9"],
            "7" : ["7", "7/", "7//.", "7//"],
            "5.1" : ["5.1"],
            "5.2" : ["5.2"]}


def getText(image): 
    """[Takes in a standardised form of the hazard label and using the pytesseract library performs OCR on a cropped and preprocessed version of the input image.]
    
    Arguments:
        image {[image]} -- [The standardised hazard label image.]
    
    Returns:
        [List] -- [A list of all words found in the label]
    """

    shape = image.shape
    cropped = im.cropImage(image, [int(shape[1]*0.13),int(shape[0]*0.42),int(shape[1]*0.87),int(shape[0]*0.63)])
    gray = im.alterColourSpace(cropped, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,101,1)
    text = pyt.image_to_string(gray, config='-l eng --oem 3 --psm 6 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    text = checkWordSimilarity(text.lower())

    return text
    # --psm 10 -c tessedit_char_whitelist=0123456789
    # -l eng --oem 1 --psm 3

def checkWordSimilarity(inWord):
    """[Checks the similiarity of an input word to the corpus of words found in all of the hazard labels and returns the highest percentage match. There is a threshold for cutoff though, if the text cannot reach a 50% match with any work it will be cut off. ]
    
    Arguments:
        inWord {[String]} -- [The word to matched to the corpus]
    
    Returns:
        [List] -- [the list of words that were found to have over a 50% match.]
    """
    trueWord = []
    words = inWord.split("\n")
    for word in words:
        if word in corpus or word.isdigit():
            trueWord.append(word.upper())
        else:
            sims = []
            max = 0,0
            for i in range(len(corpus)):
                similarity = difflib.SequenceMatcher(None,corpus[i],word).ratio()*100
                sims.append(similarity)
                if similarity > max[0]:
                    max = (similarity, i)
            if max[0] > 50:
                hold = corpus[max[1]].upper()
                trueWord.append(hold)
    trueWord.sort(key=lambda x : len(x), reverse=True)
    if not trueWord:
        trueWord.append("(none)")
    return trueWord[0]

def checkDigitSimilarity(inDigit):
    """[Checks the similarity of the class digits by cycling through the digit set and attempting to find a match.]
    
    Arguments:
        inDigit {[int]} -- [The diigit tto be compared]
    
    Returns:
        [int] -- [the digit that was fouund in the digits set.]
    """
    trueDigit = "(none)"
    for key, value in digits.items():
        if inDigit in value:
            trueDigit = key
    return trueDigit

def getColour(image,knn, top=False):  
    """[Takes in an image and attemps to classify its top or bottom colour by using the pretrained KNN model. it then returns the colour back to the calling function]
    
    Arguments:
        image {[Object]} -- [The input image to be colour classified]
        knn {[Object]} -- [the pretrained KNN model used to classify colour]
    
    Keyword Arguments:
        top {bool} -- [Flag for chekcing top colour or bottom colour] (default: {False})
    
    Returns:
        [String] -- [the colour clasified by the KNN]
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = rgb.shape
    if top:
        points = [int(shape[1]*0.75),int(shape[0]*0.365),int(shape[1]*0.755),int(shape[0]*0.37)]
    else:
        points = [int(shape[1]*0.77),int(shape[0]*0.62),int(shape[1]*0.775),int(shape[0]*0.625)]
    cropped = im.cropImage(rgb, points)
    average = np.array([cv2.mean(cropped)[0:3]]).astype(np.float32)
    ret, results, neighbours ,dist = knn.findNearest(average,5) 

    return colours[str(int(results[0][0]))]
    

def getClass(image):
    """[Using the pytesseract OCR it retrieves the class list from the input image.]
    
    Arguments:
        image {[Object]} -- [the input image]
    
    Returns:
        [int] -- [the found class digit.]
    """
    shape = image.shape
    cropped = im.cropImage(image, [int(shape[1]*0.36),int(shape[0]*0.72),int(shape[1]*0.64),int(shape[0]*0.88)])
    gray = im.alterColourSpace(cropped, cv2.COLOR_BGR2GRAY)
    text = pyt.image_to_string(cropped, config = ('--psm 12 --oem 3 -c tessedit_char_whitelist=123456789./'))
    
    hazClass = checkDigitSimilarity(text)
    return hazClass

 


