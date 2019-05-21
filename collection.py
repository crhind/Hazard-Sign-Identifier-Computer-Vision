import sys
from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np
import collection as col
import os
import csv

# Ouputs is a list of all of the correct outputs from all of the diamonds I am currently able to find.
# it is used for a quick automated way of finding differences from input image to output classificcation.
# It is hardcoded in so does definitely depend on what order your images get found.
# NOTE it has only been left in as proof of testing.
outputs = [ ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["RED","RED","FLAME","COMBUSTIBLE",3],
            ["RED","RED","FLAME","GASOLINE",3],
            ["RED","RED","FLAME","COMBUSTIBLE",3],
            ["RED","RED","FLAME","GASOLINE",3],
            ["YELLOW","YELLOW","OXIDIZER","OXYGEN",2],
            ["RED","RED","FLAME","FLAMMABLE",3],
            ["ORANGE","ORANGE","ONE POINT FIVE","BLASTING AGENTS",1],
            ["YELLOW","WHITE","RADIOACTIVE","RADIOACTIVE II",7],
            ["YELLOW","WHITE","RADIOACTIVE","RADIOACTIVE III",7],
            ["ORANGE","ORANGE","ONE POINT SIX","EXPLOSIVES",1],
            ["WHITE","WHITE","CROSS BONES IN BLACK BOX","INHALATION HAZARD",2],
            ["RED","RED","FLAME","FUEL OIL",3],
            ["GREEN","GREEN","GAS BOTTLE","NON-FLAMMABLE GAS",2],
            ["WHITE","WHITE","CROSS BONES IN BLACK BOX","INHALATION HAZARD",6],
            ["WHITE","WHITE","CROSS BONES IN BLACK BOX","TOXIC",6],
            ["WHITE","BLACK","CORROSIVE","CORROSIVE",8],
            ["WHITE","BLACK","CORROSIVE","CORROSIVE",8],
            ["RED","WHITE","FLAME","ORGANIC PEROXIDE",5.2],
            ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["YELLOW","YELLOW","OXIDIZER","OXIDIZER",5.1],
            ["GREEN","GREEN","GAS BOTTLE","NON-FLAMMABLE GAS",2],
            ["RED","RED","FLAME","FLAMMABLE GAS",2],
            ["WHITE","WHITE","CROSS BONES","POISON",6],
            ["WHITE","WHITE","CROSS BONES","POISON",6],
            ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["RED","RED","FLAME","GASOLINE",3],
            ["RED","RED","FLAME","COMBUSTIBLE",3],
            ["WHITE","WHITE","CROSS BONES IN BLACK BOX","INHALATION HAZARD",2],
            ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["WHITE","BLACK","CORROSIVE","NONE",8],
            ["RED","RED","FLAME","FLAMMABLE LIQUID",3],
            ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["YELLOW","YELLOW","OXIDIZER","OXIDIZER",5.1],
            ["WHITE","RED","FLAME","SPONTANEOUSLY COMBUSTIBLE",4],
            ["RED","RED","FLAME","FLAMMABLE GAS",2],
            ["WHITE","RED","FLAME","SPONTANEOUSLY COMBUSTIBLE",4],
            ["RED","RED","FLAME","FLAMMABLE GAS",2],
            ["WHITE","RED","FLAME","SPONTANEOUSLY COMBUSTIBLE",4],
            ["RED","RED","FLAME","FLAMMABLE GAS",2],
            ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["ORANGE","ORANGE","EXPLOSION","NONE",1],
            ["RED","RED","FLAME","FLAMMABLE GAS",2],
            ["RED","RED","FLAME","FLAMMABLE GAS",2],
            ["BLUE","BLUE","FLAME","DANGEROUS WHEN WET",4],
            ["ORANGE","ORANGE","EXPLOSION","NONE",1],
            ["RED","RED","FLAME","FLAMMABLE GAS",2],
            ["RED","RED","FLAME","FLAMMABLE GAS",2],
            ["YELLOW","YELLOW","OXIDISER","NONE",2],
            ["YELLOW","YELLOW","OXIDISER","NONE","NONE"],
            ["YELLOW","YELLOW","OXIDISER","NONE",5.1],
            ["WHITE","WHITE","CROSS BONES IN BLACK BOX","INHALATION HAZARD","NONE"],
            ["YELLOW","YELLOW","OXIDISER","NONE",2],
            ["WHITE","WHITE","CROSS BONES IN BLACK BOX","INHALATION HAZARD","NONE"],
            ["WHITE","WHITE","CROSS BONES IN BLACK BOX","NONE",2],
            ["BLUE","BLUE","FLAME","NONE","NONE"],
            ["RED","RED","FLAME","NONE","NONE"],
            ["ORANGE","ORANGE","ONE POINT FIVE","NONE",1],
            ["WHITE","WHITE","RADIOACTIVE","NONE",7],
            ["RED","RED","FLAME","NONE","NONE"],
            ["BLUE","BLUE","FLAME","NONE","NONE"],
            ["WHITE","WHITE","CROSS BONES IN BLACK BOX","NONE",2],
            ["YELLOW","WHITE","RADIOACTIVE","NONE",7],
            ["ORANGE","ORANGE","ONE POINT FIVE","NONE",1],
            ["WHITE","RED","FLAME","NONE","NONE"],
            ["WHITE","WHITE","CROSS BONES IN BLACK BOX","INHALATION HAZARD","NONE"],
            ["WHITE","WHITE","RADIOACTIVE","NONE",7],
            ["WHITE","RED","SPONTANEOUSLY COMBUSTIBLE","NONE",4],
            ["WHITE","BLACK","CORROSIVE","NONE",8]]

def display_matplot(images, title = None, gray=None):
    """[Standard display fuction used throughout testing to see the output of thhe various transforms.
        Displays multilpe plots at once for comparison, always in a square format.]
    Arguments:
        images {[Array]} -- [the array that contains all of the images you wish to display]
    Keyword Arguments:
        title {[String]} -- [A title to display on the plot to keep track of which image is bing shown.] (default: {None})
        gray {[Opencv const]} -- [The colour space you wish to display the image in.] (default: {None})
    Returns:
        [matplotlib plot] -- [The created plot]
    """
    n = np.ceil(np.sqrt(len(images)))
    index = 1
    plt.set_cmap('gray')
    plt.title(title)

    for image in images:
        plt.subplot(n, n, index)
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        index += 1
        
    plt.waitforbuttonpress(0)
    plt.close()
    return plt

def display_image(image, title=None):
    """[Standard display fuction used throughout testing to see the output of thhe various transforms.
        Displays only a single image on the plot.]
    
    Arguments:
        image-- [The input image you wish to display.]
    
    Keyword Arguments:
        title {[String]} -- [A title to display on the plot to keep track of which image is bing shown.] (default: {None})
    """

    if len(image.shape) ==3 :
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.set_cmap('gray')
    plt.imshow(image)
    plt.waitforbuttonpress(0)
    plt.close()

def getImages(imageFiles, flag=None):
    """[loads the images from the image files, by default it is in grayscale, but the flag parameter allows
        a choice of colourmaps]
    Arguments:
        imageFiles {[array]} -- [contains the image files]
    Keyword Arguments:
        flag {[opencv const]} -- [chosen colourmap to load the image in.] (default: {None})
    Returns:
        [array] -- [list of loaded images.]
    """
    images = []
    if flag is None:
        flag = cv2.IMREAD_GRAYSCALE
    for file in imageFiles:
        images.append((cv2.imread(file[0], flag),file[1]))
    return images

def getImageFiles(directory, subset=None):
    """[Gets all image files from a chosen directory with the .jpg or .ng file extension, has the option of getting
        a subsection of image files that match the subset parameter]
    
    Arguments:
        directory {[String]} -- [The directory that contains the image files.]
    
    Keyword Arguments:
        subset {[String]} -- [subset string used to get a subset of images.] (default: {None})
    
    Returns:
        [array] -- [The array of image files found in the directory]
    """
    files =  os.listdir(directory)
    images = []
    for file in files:
        file = file.lower()
        if ".jpg" in file or ".png" in file:
            fileName = file
            if subset:
                if subset in file:
                    images.append((directory  + "/" + file, file))
            else:
                images.append((directory  + "/" + file, file))
    return images

def readTrainData():
    """[Reads in the training colour data for the KNN classifier into a numpy array as per the format required 
        by the KNN classifier]
    
    Returns:
        [tuple] -- [tuple containing all the read in RGB colours and labels.]
    """
    values = []
    hazClass = []
    with open('knnData.data') as file:
        lines = csv.reader(file, delimiter=' ')
        for line in lines:
            col = np.array(line[0:3]).astype(np.float32)
            values.append(col)
            clas = np.array(line[3:4]).astype(np.float32)
            hazClass.append(clas)
        return values, hazClass
    
def display(inOrder, imgName):
    """[Displays all of the information for each hazard label in the order specified by the assignment specification sheet]
    
    Arguments:
        inOrder {[Array]} -- [the ordered array contain the hazard information]
        imgName {[String]} -- [the name of the original image file]
    """

    print(imgName)
    for entry in inOrder:
        print(" top : " + entry[0])
        print(" bottom : " + entry[1])
        print(" class = " + entry[2])
        print(" text = " + entry[3])
        print(" symbol :" + entry[4] + "\n")
    print("\n")




