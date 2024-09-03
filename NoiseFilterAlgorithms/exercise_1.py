# --------------------------------------------------------- #
# Project 1. Exercise 1                                     #
# Quantify the effects that different filters have          #
# when applied to an image.                                 #
#                                                           #
# Michail Apostolidis                                       #
# --------------------------------------------------------- #

import csv
import glob  # for choosing random file
import random
from math import log10, sqrt

# import necessary packages
import cv2 as cv  # opencv
import numpy as np  # for arrays
import skimage.metrics
from matplotlib import pyplot as plt


def mean_square_error(img1, img2):
    """
        'Mean Squared Error' of two images.

            Calculates the the sum of the squared
            difference between the two images.
            Returns the MSE number, the lower the error,
            the more "similar" the two images are.

            :param img1: First Image to compare.
            :param img2: Second Image to compare.
            :return: MSE.
    """
    img1_fl = img1.astype("float")
    img2_fl = img2.astype("float")
    dif = img1_fl - img2_fl
    result = np.sum(dif ** 2)
    result = result / float(img1.shape[0] * img2.shape[1])

    return result  # return the MSE


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:   # MSE is zero means no noise is present in the signal.
        return 100
    max_pixel = 255.0
    result = 20 * log10(max_pixel / sqrt(mse))
    return result


if __name__ == '__main__':
    # 1. Read a random image from folder "../images_to_use"
    images = glob.glob("./images_to_use/*.jpg")  # select path with file type .jpg
    random_img = random.choice(images)           # pick random image from path

    img = cv.imread(random_img)                       # read the random image
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    # convert image from BGR to GRAYSCALE

    # display the original random image
    cv.namedWindow("Random Painting", cv.WINDOW_AUTOSIZE)
    cv.imshow("Random Painting", img)
    cv.waitKey(0)
    cv.destroyWindow("Random Painting")     # closing the window

    # display the grayscale random image
    cv.namedWindow("Random Painting (grayscale)", cv.WINDOW_AUTOSIZE)
    cv.imshow("Random Painting (grayscale)", gray_img)
    cv.waitKey(0)

    # 2. Applying filters to the random image (averaging, gaussian, median, and bilateral)
    kernel = (5, 5)
    # a. Averaging filter:
    averaging = cv.blur(gray_img, kernel)
    # b. Gaussian filter:
    gaussian = cv.GaussianBlur(gray_img, kernel, 0)
    # c. Median filter:
    median = cv.medianBlur(gray_img, 5)
    # d. Bilateral filter:
    bilateral = cv.bilateralFilter(gray_img, 9, 75, 75)

    # Display results:
    # setting values to rows and column variables for plot
    rows = 2
    columns = 2
    plt.figure("Filter Results")
    plt.subplot(rows, columns, 1)   # add averaging image at the 1st position
    plt.axis("off")                 # remove numbered axes
    plt.imshow(averaging, cmap="gray")
    plt.title("Averaging Filter")
    plt.subplot(rows, columns, 2)   # add bilateral image at the 2nd position
    plt.axis("off")                 # remove numbered axes
    plt.imshow(bilateral, cmap="gray")
    plt.title("Bilateral Filter")
    plt.subplot(rows, columns, 3)   # add gaussian image at the 3rd position
    plt.axis("off")                 # remove numbered axes
    plt.imshow(gaussian, cmap="gray")
    plt.title("Gaussian Filter")
    plt.subplot(rows, columns, 4)   # add median image at the 4th position
    plt.axis("off")                 # remove numbered axes
    plt.imshow(median, cmap="gray")
    plt.title("Median Filter")
    plt.show()

    # Saving the filtered images
    cv.imwrite("./images_saved_ex_1/rand_image.jpg", img)              # original image
    cv.imwrite("./images_saved_ex_1/rand_gray_image.jpg", gray_img)    # original grayscale image
    cv.imwrite("./images_saved_ex_1/rand_averaging.jpg", averaging)    # averaging filter image
    cv.imwrite("./images_saved_ex_1/rand_gaussian.jpg", gaussian)      # gaussian filter image
    cv.imwrite("./images_saved_ex_1/rand_bilateral.jpg", bilateral)    # bilateral filter image
    cv.imwrite("./images_saved_ex_1/rand_median.jpg", median)          # median filter image

    # 3 + 4. Comparing the similarity of the new image with the original and writing the scores to csv file
    with open('similarity_metrics_ex_1.csv', mode='w', newline='') as compare_file:
        compare_writer = csv.writer(compare_file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        compare_writer.writerow(["Averaging", str(mean_square_error(gray_img, averaging)), str(psnr(gray_img, averaging)),
                                 str(skimage.metrics.structural_similarity(gray_img, averaging))])
        compare_writer.writerow(["Gaussian", str(mean_square_error(gray_img, gaussian)), str(psnr(gray_img, gaussian)),
                                 str(skimage.metrics.structural_similarity(gray_img, gaussian))])
        compare_writer.writerow(["Bilateral", str(mean_square_error(gray_img, bilateral)), str(psnr(gray_img, bilateral)),
                                 str(skimage.metrics.structural_similarity(gray_img, bilateral))])
        compare_writer.writerow(["Median", str(mean_square_error(gray_img, median)), str(psnr(gray_img, median)),
                                 str(skimage.metrics.structural_similarity(gray_img, median))])

