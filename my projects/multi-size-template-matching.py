'''
    This project extends the utility of template matching to multi-size object matching (larger than the input template),
    which otherwise matches only the objects of same dimensions
    
    USAGE:
    ======
    python multi-size-template.py --template coke.jpg --images coke
    
    in order to visualize the process of template matching, that is, how the input image is continuously resized to match the object with the template, modify the run command as the following:
    
    python multi-size-template.py --template coke.jpg --images coke --visualize 1
'''

# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")      #input the template image in the placeholder "template"
ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")      #input the folder containing the images to objects to detect in the placeholder "image "
ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")         #optional parameter to visualize the process of template matching
args = vars(ap.parse_args())

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)           #images are converted to grayscale in order to ignore the color deviations and focus only on the pattern (in order to avoid noise in the image)
template = cv2.Canny(template, 50, 200)                                           #edges on the template image captured using Canny edge detector (https://bit.ly/2HPn5DR)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

# loop over the images to find the template in
for imagePath in glob.glob(args["images"] + "/*.jpg"):                    #glob is used to extract the path of images, for reading multiple images in a folder. (https://docs.python.org/3/library/glob.html)
    image = cv2.imread(imagePath)                                                       #the paths extracted using glob is used to read individual images in the folder
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                    # similar to the operations done on template image, one must convert the input images to grayscale as well in order to avoid noise.
    found = None                                                                                       # 'found' is a bookkeeping variable to keep track of the matched region
    
    
    # here we are looping over various scales of the image in order to find the exact bounding box of the template
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        
        # if the resized image is smaller than the template, then break from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
    
        edged = cv2.Canny(resized, 50, 200)                                                       #detecting edges in resized versions of the input images using Canny edge detector
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)      #template matching function (https://bit.ly/2FQ7nGs) is finally used here with the method CV_TM_CCOEFF which is used to output the global maximums using the minMaxLoc ( https://bit.ly/2JVvcAS ) function below.
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        
        #This loop will be called only when the command line argument 'visualize' is explicitly set as 'true'
        if args.get("visualize", False):
            
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)
    
    
        
        # if we have found a new maximum correlation value, then update the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)


# Finally we unpack the bookkeeping varaible and compute the (x, y) coordinates of the bounding box based on the resized ratio

    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 200, 100), 8)        # drawing a bounding box around the detected result and display the image
    cv2.imshow("Image", image)
    cv2.waitKey(1000)                            #waiting for 1 second for each output corresponding to each input image.
    cv2.destroyWindow("Image")

cv2.waitKey(0)
