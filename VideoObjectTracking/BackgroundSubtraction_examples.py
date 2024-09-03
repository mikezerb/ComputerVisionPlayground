# importing libraries
import cv2 as cv

if __name__ == '__main__':
    # creating object
    fgbg1 = cv.bgsegm.createBackgroundSubtractorMOG()
    fgbg2 = cv.createBackgroundSubtractorMOG2()
    fgbg3 = cv.bgsegm.createBackgroundSubtractorGMG()
    fgbg4 = cv.createBackgroundSubtractorKNN(detectShadows=False)   # for no gray shadows

    # Read video:
    # loading a video to use object detection
    cap = cv.VideoCapture('./Background_subtraction/ped_walking.avi')
    while 1:
        # read frames
        ret, img = cap.read()

        # additional gaussian blur for better results
        img = cv.GaussianBlur(img, (5, 5), 0)
        # apply each mask for background subtraction
        fgmask1 = fgbg1.apply(img)
        fgmask2 = fgbg2.apply(img)
        fgmask3 = fgbg4.apply(img)

        cv.imshow('Original', img)
        cv.imshow('MOG', fgmask1)
        cv.imshow('MOG2', fgmask2)
        cv.imshow('KNN', fgmask3)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    # End of exercise 2