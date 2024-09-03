# --------------------------------------------------------- #
# Project 2. Exercise 1                                     #
#  Template matching under noise                            #
#                                                           #
# Michail Apostolidis                                       #
# --------------------------------------------------------- #

# import necessary packages
import cv2 as cv
import numpy as np


def noise(image, noise_perc):
    """
        Function for adding noise to a given image
        with a percentage of the effect.
        Using random.uniform(0, 1) to return floating point number in the range [0.0, 1.0].
        :param image: Image to apply the noise.
        :param noise_perc: Percentage of the noise (from 0% to 20%).
        :return: noisy image.
   """
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i][j] = image[i][j] + noise_perc * np.random.uniform(0, 1) * image[i][j] - noise_perc \
                           * np.random.uniform(0, 1) * image[i][j]
    return output


def print_image(window_name, image):
    """
        Function for printing image to window.

        :param window_name: Description of the window name.
        :param image: Image to show
    """
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, image)
    cv.resizeWindow(window_name, 800, 537)
    cv.waitKey(0)
    cv.destroyWindow(window_name)


def template_matching(original_image, image, template):
    """
        Function for running template matching of images

        :param original_image: Original image to add the rectangles.
        :param image: Image to apply template matching.
        :param template: Template image for matching.
    """
    # Apply template Matching
    result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.80

    # copy of original image to show different matches for each noise
    img_rec = original_image.copy()

    # mark the corresponding location(s)
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rec, pt, (pt[0] + w, pt[1] + h), (255, 0, 255), 2)

    return img_rec  # return the image with the rectangles


# - START OF EXERCISE -
if __name__ == '__main__':
    # A. adding noise to the original image:
    # load original image from folder ex1_images
    filename = './ex1_images/birds-flock.png'
    image = cv.imread(filename)

    # converting bgr to grayscale
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # template matching image
    tmpl_filename = './ex1_images/birds-template.jpg'
    template = cv.imread(tmpl_filename, 0)
    w, h = template.shape[::-1]     # getting the width and height

    # array of the noises percentage
    noise_perc = [0, 0.05, 0.10, 0.15, 0.20]

    # for every noise percentage:
    for per in noise_perc:
        # applying noise to the grayscale image:
        image_gray_noisy = noise(image_gray, per)
        # printing noisy grayscale image for every percentage
        print_image("Noise perc: " + str(int(per * 100)) + "%", image_gray_noisy)
        # saving the image to folder ex1_results
        cv.imwrite('./ex1_images/ex1_results/noise_res/gray_ns_' + str(int(per * 100))+ '%.png', image_gray_noisy)

        # copy of original image to show different matches for each noise
        img_det_noise = image.copy()
        # Apply template Matching
        image_detect = template_matching(img_det_noise, image_gray_noisy, template)

        # display the image with the detected boxes
        print_image("Detected boxes noise " + str(int(per * 100)) + "%", image_detect)

        # saving the image to folder ex1_results
        cv.imwrite('./ex1_images/ex1_results/noise_res/detected_ns_' + str(int(per * 100)) + '%.png', image_detect)

        # B. Applying Gaussian Filter to the noisy image:
        # applying gaussian blur on the noisy image with kernel size(5,5)
        # with sigmaX = 0.9
        gauss = cv.GaussianBlur(image_gray_noisy, (5, 5), 0.9)

        # copy of original image to show different matches for each noise
        img_det_gauss = image.copy()

        # printing the gaussian image
        print_image("Gaussian Blur " + str(int(per*100)) + "%", gauss)

        # saving the gaussian image to ex1_results/gaussian_res/ directory
        cv.imwrite("./ex1_images/ex1_results/gaussian_res/gaussian_blur_" + str(int(per * 100)) + "%.png", gauss)

        # applying template matching to the gaussian filtered image
        image_detect_gauss = template_matching(img_det_gauss, gauss, template)

        # display the image with the detected boxes
        print_image("Detected boxes with gaussian " + str(int(per * 100)) + "%", image_detect_gauss)
        # saving the image with the detected matches
        cv.imwrite('./ex1_images/ex1_results/gaussian_res/detected_gauss_' + str(int(per * 100)) + '%.png', image_detect_gauss)
        # - END OF EXERCISE -

