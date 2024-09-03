# --------------------------------------------------------- #
# Project 1. Exercise 2                                     #
# Quantify the effect that different filters have           #
# when applied to an image.                                 #
#                                                           #
# Michail Apostolidis                                       #
# --------------------------------------------------------- #

# import necessary packages
import csv
import glob  # for choosing random file
import os   # for file name
import random
import cv2 as cv  # opencv
import numpy as np  # for arrays
from skimage.metrics import structural_similarity as ssim   # for metric 1
from skimage.metrics import mean_squared_error as mse   # for metric 2


def noise(ns_type, img):
    """
    This is a function to add different types of noise to an image.

        When the `ns_type` is "saltandpepper" returns the image with
        10% of salt and pepper noise. When `ns_type` is "poisson"
        it returns the image with poisson nose and when `ns_type` is
        "gauss" returns the gaussian nose image.

        :param ns_type: Type of noise.
        :param img: Image to add noise.
        :return: Noisy Image.
    """
    if ns_type == "saltandpepper":
        noiseout = np.zeros(img.shape, np.uint8)
        prob = 0.05
        threshold = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdm = random.random()
                if rdm < prob:
                    noiseout[i][j] = 0
                elif rdm > threshold:
                    noiseout[i][j] = 255
                else:
                    noiseout[i][j] = img[i][j]
        return noiseout
    elif ns_type == "poisson":
        poissonout = np.zeros(img.shape, np.uint8)
        peak = 0.15
        row = img.shape[0]
        col = img.shape[1]
        poisson = np.random.poisson(img / 255.0 * peak) / peak * 255
        poisson = poisson.reshape(row, col).astype('uint8')
        poissonout = img + poisson
        return poissonout
    elif ns_type == "gauss":
        mean = 0
        dev = 1
        # Generate Gaussian noise
        gauss = np.random.normal(mean, dev, img.size)
        gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv.add(img, gauss)
        return img_gauss


# Filter function:
def filters(img, imgID, imgNS):
    """
    This is a function to add different filters to an image.

        When the `fl_type` is "bilateral" it applies the bilateral
        filter. When `fl_type` is "median" it applies the median
        filter, when `fl_type` is "averaging" it applies the averaging
        filter and when `fl_type` is "gaussian" it applies the gaussian
        filter. Prints the results and writes the similarity scores.

        :param img: Image to filter.
        :param imgID: Image ID for the csv file.
        :param imgNS: Image Nose type for the csv file.

    """
    # Averaging filter:
    kernel = np.ones((5, 5), np.float32) / 25  # averaging kernel for 5 x 5 window patch
    averaging = cv.filter2D(img, -1, kernel)
    print_image(averaging, "averaging filter")

    # Median filter:
    median = cv.medianBlur(img, 5)
    print_image(median, "median filter")

    # Gaussian filter:
    kernel = (5, 5)
    gaussian = cv.GaussianBlur(img, kernel, 0)
    print_image(gaussian, "gaussian filter")

    # Bilateral filter:
    bilateral = cv.bilateralFilter(img, 9, 100, 5)
    print_image(bilateral, "bilateral filter")

    # 4 + 5 Comparing the similarity of the new image with the original
    ssim_score_avg = ssim(img, averaging)
    mse_score_avg = mse(img, averaging)
    ssim_score_median = ssim(img, median)
    mse_score_median = mse(img, median)
    ssim_score_gaussian = ssim(img, gaussian)
    mse_score_gaussian = mse(img, gaussian)
    ssim_score_bilateral = ssim(img, bilateral)
    mse_score_bilateral = mse(img, bilateral)

    file = os.path.basename(imgID)  # getting the file name for writing csv and saving images
    filenameID = os.path.splitext(file)[0]  # getting the name without the .jpg

    with open('similarity_metrics_ex_2.csv', mode='a', newline='') as compare_file:
        compare_writer = csv.writer(compare_file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        compare_writer.writerow([filenameID, imgNS + " noise", "Averaging", round(ssim_score_avg, 8), round(mse_score_avg, 4)])
        compare_writer.writerow([filenameID, imgNS + " noise", "Gaussian", round(ssim_score_gaussian, 8), round(mse_score_gaussian, 4)])
        compare_writer.writerow([filenameID, imgNS + " noise", "Median", round(ssim_score_median, 8), round(mse_score_median, 4)])
        compare_writer.writerow([filenameID, imgNS + " noise", "Bilateral", round(ssim_score_bilateral, 8), round(mse_score_bilateral, 4)])

    filename = "images_saved_ex_2/" + filenameID    # filename with path to save images

    # Saving images:
    cv.imwrite(filename + "noisy_image.jpg", img)
    print(filename + "noisy_image.jpg" + "saved!")
    cv.imwrite(filename + "_avg.jpg", averaging)
    print(filename + "_avg.jpg" + " saved!")
    cv.imwrite(filename + "_gauss.jpg", gaussian)
    print(filename + "_gauss.jpg" + " saved!")
    cv.imwrite(filename + "_median.jpg", median)
    print(filename + "_median.jpg" + " saved!")
    cv.imwrite( filename + "_bilateral.jpg", bilateral)
    print(filename + "_bilateral.jpg" + " saved!")


def print_image(image, type):
    cv.namedWindow("Random " + type, cv.WINDOW_AUTOSIZE)
    cv.imshow("Random " + type, image)
    cv.waitKey(0)
    cv.destroyWindow("Random " + type)


def get_rand_noise():
    """
        Function that returns random type of noise
    """
    noise_list = ["saltandpepper", "gauss", "poisson"]
    return random.choice(noise_list)


if __name__ == '__main__':
    for image in glob.glob("images_to_use/*.jpg"):
        # 1. Read all the images of the images_to_use folder
        originalImg = cv.imread(image)
        gray_img = cv.cvtColor(originalImg, cv.COLOR_BGR2GRAY)  # convert image from BGR to GRAYSCALE
        # 2. Apply random noise (salt and pepper, poisson and gaussian)
        randNoiseType = get_rand_noise()    # find random string of noise
        randNoisyImg = noise(randNoiseType, gray_img)   # applying random noise to each image
        print_image(randNoisyImg, randNoiseType)    # printing the image
        # 3. Applying filters to each image (averaging, gaussian, median, and bilateral)
        filters(randNoisyImg, image, randNoiseType)
