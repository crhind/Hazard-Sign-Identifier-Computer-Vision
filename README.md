# Computer Vision: Using low level techniques for identifying various hazard signs in static images.

### Note: This started out as a project for uni but got carried on after completion date due to OCD drive to not have a semi-complete project.

## Problem Specification
+ Each hazard sign can take up 20% to 80% of an image.
+ Clearly each image can contain between 1 and 5 hazard signs.
+ A hazard sign can have up to 90 degrees of rotation.
+ The images vary in difficulty:
  - They start off with a single standard hazard sign with no rotation, a simple background and good lighting.
  - Then they progress to multiple hazard signs with varying rotations, cast on complex backgrounds (geometric patterns) with shadows cast and image capture angles of approx 20 degrees from vertical.

## Problem Solution
+ Starting with a standard RGB image
+ Sequentially threshold and filter the image to remove background surface textures (noise).
+ Canny transform the image to get a binary image.
+ Use a polygon approximator to identify polygons in the binary image, then validate rectangles using geometry, then hazard signs using logic.
+ once a diamond (Hazard sign) was identified it came down to identifying the various elements internally:
  - Top half colour, Bottom half colour, and words on the sign, what was the hazmat symbol, what is the hazard identifier. 
+ To do this I used a combination of techniques (SO MUCH TRIAL AND ERROR):
  - ORB pattern matching (Oriented FAST and rotated BRIEF, google it) to identify the hazard symbol in the top half of the image. 
  - Tesseract OCR was used to identify both the hazard number and the text (if any)
  - Colour identification was a simple KNN algorithm to identify a pre-set number of colours. NOTE: I first transformed the colour space to YUV instead of RGB as it had a much better chance of dealing with shadows.

## Limitations.
+ As with many low level approaches to computer vision it is far from perfect with a successfull identification rate of the most complex images of approx 70%. 
+ The system struggles very poorly with shadows that are cast directly across an edge of the hazard sign.
+ Finally if the complex background is made up of diamonds (i.e. pavers) the system struggles do discern between the background diamonds and the hazard diamond