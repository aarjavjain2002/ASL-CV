import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

#* 0 is the default camera -> set to iPhone camera on Mac
#- We can use 1 for laptop camera
cam = cv2.VideoCapture(1)

#* Hand detector object
detector = HandDetector(maxHands=1)

#* Offset value for the cropped image
offset = 20

#* Fixed image size
fixed_img_dimension = 500

#* Defining the location to save the training data
folder = "Data/Y"

#* Counter to keep track of the number of images
counter = 0

#* The following loop runs the camera (displays on screen)
while True:
    #* All the important stuff happens in this loop
    #- Reading the image
    success, img = cam.read() 

    #- This identifies the hands in the image and draws the landmarks on it
    #- It saves the hands it recogizes in an array called hands
    hands, img = detector.findHands(img) 
    
    #-If there is a hand, we take the first hand from the hands array (maxHands = 1 so there is only one hand)
    if hands:
        hand = hands[0]
        #- Bounding box of the hand
        x,y,w,h = hand['bbox']

        #* This is the cropped image of the hand
        #- Height = y to y+h[eight], Width = x to x+w[idth] (ALONG WITH SOME OFFSET)
        img_crop = img[y-offset : y+h+offset, x-offset : x+w+offset]

        #-*Here we must define a fixed background white image
        #- 500 x 500 x 3 matrix of ones + specifying the data type (unsigned integer 8 bit)
        #- The value inside the matrix is RGB (r,g,b)
        #- Since matrix of 1s, therefore color = black. We multiply with 255 to make the square white.
        img_white = np.ones((fixed_img_dimension, fixed_img_dimension, 3), np.uint8)*255

        #* Scaling the cropped image to match the white image size (to ensure uniformity of all cropped images)
        #- Aspect Ratio = height / width
        aspect_ratio = h/w

        #* Height > Width
        if aspect_ratio > 1:
            k = fixed_img_dimension/h #- Scale factor for height
            new_width = math.floor(k * w) #- Multiply width with same scale factor

            #* Resizing the cropped image
            resized_img = cv2.resize(img_crop, (new_width, fixed_img_dimension))

            #* Putting the *RESIZED* cropped hand image into the CENTER of the white image
            #- Gap between the left edge of the white image and the left edge of the resized image
            width_gap = (fixed_img_dimension - new_width) // 2 #- // since left and right gap

            #- img_white[height, width] -> height and  width where we want to put the image
            img_white[: , width_gap : width_gap + new_width] = resized_img #- Height is the same, Width = gap from left TO gap from left + new width

            #! We are not using the following code because it puts the image in the top left corner
            #! This is for reference only
            #- img_crop.shape[0] -> height of the cropped image
            #- img_crop.shape[1] -> width of the cropped image
            # img_white[0:resized_img.shape[0], 0:resized_img.shape[1]] = resized_img
        
        #* Width > Height
        if aspect_ratio <= 1:
            k = fixed_img_dimension/w #- Scale factor for width
            new_height = math.floor(k * h) #- Multiply height with same scale factor

            #* Resizing the cropped image
            resized_img = cv2.resize(img_crop, (fixed_img_dimension, new_height))

            #* Putting the *RESIZED* cropped hand image into the CENTER of the white image
            #- Gap between the bottom edge of the white image and the bottom edge edge of the resized image
            height_gap = (fixed_img_dimension - new_height) // 2 #- // since top and bottom gap

            #- img_white[height, width] -> height and  width where we want to put the image
            img_white[height_gap: height_gap + new_height , :] = resized_img #- Height is the same, Width = gap from left TO gap from left + new width
        
        #- Showing the cropped image on screen
        cv2.imshow("Image Crop", img_crop)
        cv2.imshow("Image White", img_white)

    #* Showing the image (webcam really) on screen
    cv2.imshow("Image", img) 

    #- 1 millisecond delay between frames
    key = cv2.waitKey(1) 

    #* Saving using the 's' keys
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", img_white) #- We are storing the image with a unique name (time.time())
        print(counter)
        if counter == 150:
            exit()