# --------------------------------------------------------- #
# Project 5. Exercise 3                                     #
#  Image detection using Background subtraction             #
#  Creating bounding box around moving objects              #
#  Calculating the coordinates of the                       #
#  centroid of the object                                   #
#                                                           #
# Michail Apostolidis                                       #
# --------------------------------------------------------- #

# import necessary libraries
import cv2 as cv
import numpy as np

# Start of exercise 3:
if __name__ == '__main__':
    # Read video:
    # Create VideoCapture object
    # set video path
    videoPath = './Background_subtraction/ped_walking.mp4'
    cap = cv.VideoCapture(videoPath)

    # Create the background subtractor object
    back_sub = cv.createBackgroundSubtractorMOG2(detectShadows=False)  # false for no gray shadows

    # Create kernel for morphological operation
    # define the kernel with size 3x3
    kernel = np.ones((3, 3), np.uint8)
    i = 0  # var for saving frames

    # While loop for video
    while cap.isOpened():
        i = i + 1
        # Reading the video stream
        ret, frame = cap.read()

        # Use every frame to calculate the foreground mask and update
        # the background
        # apply each mask for background subtraction
        fg_mask = back_sub.apply(frame)

        # Adding blur for better results:
        # Close dark gaps in foreground object using closing
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)

        # Remove noise with a Gaussian filter
        fg_mask = cv.GaussianBlur(fg_mask, (5, 5), 0)

        # Applying threshold for false positives
        _, fg_mask = cv.threshold(fg_mask, 127, 255, cv.THRESH_BINARY)

        # Find the index of the largest contour and draw bounding box
        fg_mask_bounding_box = fg_mask
        contours, hierarchy = cv.findContours(fg_mask_bounding_box, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv.contourArea(c) for c in contours]

        # if no contours where found
        if len(areas) < 1:

            # Display the output frame
            cv.imshow('frame', frame)

            # If "q" is pressed on the keyboard,
            if cv.waitKey(30) & 0xFF == ord('q'):
                break  # exit this loop

            # continue from top the while loop
            continue

        else:
            # Find the largest moving object in the image
            max_index = np.argmax(areas)

        # 3. Creating a bounding box around the moving object.
        # Draw the bounding box
        cnt = contours[max_index]
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Draw circle in the center of the bounding box
        x2 = x + int(w / 2)
        y2 = y + int(h / 2)
        cv.circle(frame, (x2, y2), 4, (0, 255, 0), -1)

        # show the centroid coordinates
        # on the center of the box of the frame
        text = "x: " + str(x2) + ", y: " + str(y2)
        cv.putText(frame, text, (x2 - 10, y2 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the output frame
        cv.imshow('frame', frame)

        # If "q" is pressed on the keyboard,
        if cv.waitKey(15) & 0xFF == ord('q'):
            break  # exit this loop
        else:
            cv.imwrite("./Background_subtraction/results/BS_det_" + str(i) + ".png", frame) # saving each frame

    cap.release()
    cv.destroyAllWindows()
    # End of exercise 3
