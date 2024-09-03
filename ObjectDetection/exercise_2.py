# --------------------------------------------------------- #
# Project 2. Exercise 2                                     #
#  Object detection using color histograms over             #
#  overlapping patches using custom thresholds              #
#                                                           #
# Michail Apostolidis                                       #
# --------------------------------------------------------- #

# import necessary packages
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim  # for second metric


def print_image(window_name, image):
    """
        Function for printing image to window.

        :param window_name: Description of the window name.
        :param image: Image to show
    """
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, image)
    cv.resizeWindow(window_name, 830, 973)
    cv.waitKey(0)
    cv.destroyWindow(window_name)


def calc_histogram(image_patch, template):
    """
        Function for calculating the histogram of each image patch
        and comparing it to the template histogram.

        :param image_patch: Image patch to calculate the histogram.
        :param template: Template image.
        :return: Similarity score.
   """
    # getting the histograms for the image patch and the template image
    tmp_histogram = cv.calcHist([template], [0], None, [256], [0, 256])
    img_patch_histogram = cv.calcHist([image_patch], [0], None, [256], [0, 256])
    # calculating the similarity score of the histograms of each image patch
    score = cv.compareHist(tmp_histogram, img_patch_histogram, 0)
    print(score)
    return score


def calc_ssim(image_patch, template):
    sim, diff = ssim(image_patch, template, full=True, multichannel=True)
    print(sim)
    return sim


def scanning_img_patches(image, template, sim_type):
    # original image dimensions:
    height = image.shape[0]
    width = image.shape[1]

    # template image dimensions:
    tmp_height = template.shape[0]
    tmp_width = template.shape[1]

    loc = []
    similarities = np.zeros((height - tmp_height, width - tmp_width))
    mask = np.ones((tmp_height, tmp_width, 3))
    # Getting the threshold:
    threshold = input("Please give threshold [0-1]: ")
    threshold = float(threshold)
    for i in range(0, height - tmp_height):
        for j in range(0, width - tmp_width):
            print("i: ", i)
            print("j: ", j)
            image_patch = image[i: i + tmp_height, j: j + tmp_width]
            # image_patch = image[i - tmp_height//2: i + tmp_height//2, j - tmp_width//2: j + tmp_width//2]
            # calculate the similarity score for each image patch
            if sim_type == "hist":
                similarities[i, j] = calc_histogram(image_patch, template)
            else:
                similarities[i, j] = calc_ssim(image_patch, template)
            if similarities[i, j] > threshold:
                offset = np.array((i, j))  # spreading the boundaries with the mask
                image[offset[0]:offset[0] + mask.shape[0], offset[1]:offset[1] + mask.shape[1]] = mask
                (max_Y, max_X) = (i, j)
                loc.append((max_X, max_Y))
    return loc


def apply_rectangles(loc, image_detected, height, width):
    for i in range(0, len(loc)):
        box = loc[i]
        cv.rectangle(image_detected, box, (box[0] + width, box[1] + height), (0, 0, 255), 3)
    print_image("Detected Image", image_detected)
    return image_detected


if __name__ == '__main__':
    # 1. Reading the images (original and template):
    # load original image from folder ex2_images
    filename = './ex2_images/pacman.png'
    image = cv.imread(filename)
    # template matching image
    tmpl_filename = './ex2_images/ghost.png'
    template = cv.imread(tmpl_filename)
    h, w = template.shape[:2]
    # display the original image
    print_image("Original Image", image)
    # display the template image
    print_image("template Image", template)

    # copies of the original image
    image_ssim = image.copy()
    image_hist = image.copy()
    # Calculating with histogram similarities
    image_ptc_hist = scanning_img_patches(image, template, "ssim")

    # getting the boxes of the scan to apply the rectangle box.
    # displaying the image with the areas where the object was located
    image_det_hist = apply_rectangles(image_ptc_hist, image_ssim, h, w)
    # saving the image to the results folder
    cv.imwrite("./ex2_images/ex2_results/detected_hist.png", image_det_hist)
    # Calculating with second method (ssim)
    image_ptc_ssim = scanning_img_patches(image, template, "ssim")
    # adding the bounding boxes and displaying the resulting image
    image_det_ssim = apply_rectangles(image_ptc_ssim, image_hist, h, w)
    # saving the image to the results folder
    cv.imwrite("./ex2_images/ex2_results/detected_ssim.png", image_det_ssim)
