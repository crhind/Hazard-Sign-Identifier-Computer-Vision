import imagemanipulation as im
import imagepreprocessing as ip
import collection as col
import textprocessing as txt
import symboldetection as sym
import numpy as np
import os
from cv2 import cv2, ml
from collection import outputs
import time

imageDir = "./"
trainData = "./kNNData"

def main():
    start_time = time.time()
    comparisons =[]
    print("starting")
    # Begins loading all training data for KNN ad retrieves all image file paths from chosen directory
    trainData, labels = col.readTrainData()
    knn = ml.KNearest_create()
    knn.train(np.array(trainData),ml.ROW_SAMPLE,np.array(labels))
    col.readTrainData()
    images = col.getImageFiles(imageDir)
    comparisonImages = col.getImageFiles("./ORB_images")

    # Checks to make sure that we have image paths
    if(images):
        # Loads all of the image files
        images = col.getImages(images, cv2.IMREAD_COLOR)
        comparisonImages = col.getImages(comparisonImages)
        for image in comparisonImages:
            kp, des  = sym.getKeyPoints(image[0])
            compared = (image[0], kp, des, image[1])
            comparisons.append(compared)
        # Used for testing purposes and has only been leeft in to demonstrate that testing occured.
        i = 0
        topScore = 0
        bottomScore = 0
        symbolScore = 0
        textScore = 0
        hazScore = 0
        toSort = []
        # Now we cycle through the images to find all the points that corespond to hazard label corners.
        for image in images:
            img = image[0]
            realligned = im.edgeFinding(img)
            if not realligned:
                print("Something seriously wrong happend, no diamond founds.")

            # We now cycle through all of the found diamonds (Corners of the hazamat labels) and transform them to a standardised form.
            # Once in a standard form we apply our feature detection methods to retrieve all hazrd label attributes.
            for diamond in realligned:
                newImage = im.perspectiveManipulation(img, diamond)
                text = txt.getText(newImage)
                symbol = sym.compareKeyPoints(comparisons, newImage)
                hazClass = txt.getClass(newImage)
                top = txt.getColour(newImage,knn, top=True)
                bottom = txt.getColour(newImage, knn)
                toSort.append((top,bottom,hazClass,text,symbol))
                col.display_image(newImage)
                # Display the hazard label attributes.

                
                # This is the increment for the automated testing score.
                if top is outputs[i][0]:
                    topScore +=1
                if bottom is outputs[i][1]:
                    bottomScore += 1
                if len(symbol) == len(outputs[i][2]):
                    symbolScore += 1
                if " ".join(text) is outputs[i][3]:
                    textScore += 1
                if hazClass is str(outputs[i][4]) or hazClass is outputs[i][4]:
                    hazScore += 1
                i += 1
            toSort.sort(key=lambda x: (x[0], x[1], x[2]))
            col.display(toSort, image[1])
            toSort = []

        # printing out the automated test score. NOTE this was only left in as proof of testing.
        # elapsed_time = time.time() - start_time
        # print("TOP SCORE : " + str(topScore) + " = " + str(topScore/74))
        # print("BOTTOM SCORE : " + str(bottomScore) + " = " + str(bottomScore/74))
        # print("SYMBOL SCORE : " + str(symbolScore) + " = "  + str(symbolScore/74))
        # print("TEXT SCORE : " + str(textScore) + " = " + str(textScore/74))
        # print("CLASS SCORE : " + str(hazScore) + " = "  + str(hazScore/74))
        # print("TIME : " + str(elapsed_time))

    else:
        print("No images in the directory")

if __name__ == "__main__":
    main()