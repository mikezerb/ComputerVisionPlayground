# --------------------------------------------------------- #
# Project 2. Exercise 3                                     #
#  Object detection using color histograms over             #
#  overlapping patches using custom thresholds              #
#                                                           #
# Michail Apostolidis                                       #
# --------------------------------------------------------- #

# import necessary packages
import cv2 as cv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, MiniBatchKMeans
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def noise(image, noise_perc):
    """
        Function for adding noise to a given image
        with a percentage of the effect.
        Using random.random() to return floating point number in the range (0.0, 1.0].
        :param image: Image to apply the noise.
        :param noise_perc: Percentage of the noise (from 0% to 20%).
        :return: noisy image.
   """
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i][j] = image[i][j] + noise_perc * np.random.random() * image[i][j] - noise_perc \
                           * np.random.random() * image[i][j]
    return output


def print_image(window_name, image):
    """
        Function for printing image to window.

        :param window_name: Description of the window name.
        :param image: Image to show
    """
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, image)
    cv.resizeWindow(window_name, 480, 348)
    cv.waitKey(0)
    cv.destroyWindow(window_name)


def get_f1_score (binary, image):
    precisionscore = precision_score(binary, image)
    recallscore = recall_score(binary, image)
    f1 = 2 * ((precisionscore * recallscore)/(precisionscore+recallscore))
    return f1


def segment_image(image, noise, binaryImg):
    # getting the shape of the image
    originShape = image.shape
    # getting the channel size
    chan = len(originShape)
    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities (or the 3 channels that I currently have)
    flatImg = np.reshape(image, [-1, chan])

    # 1. MeanShift algorithm
    # Estimate bandwidth for meanshift algorithm
    print('Implementing MeanShift algorithm to the image')
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # Performing Mean shift on flatImg
    ms.fit(flatImg)  # (r, g, b) vectors corresponding to the different clusters after Mean shift
    labels = ms.labels_

    # Remaining colors after Mean shift
    cluster_centers = ms.cluster_centers_

    # Finding and displaying the number of clusters
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("The number of estimated clusters is %d." % n_clusters_)

    # Displaying segmented image
    segmentedImg = np.reshape(labels, originShape[:2])
    segmentedImg = label2rgb(segmentedImg, bg_label=0) * 255  # need this to work with OpenCV cv.imshow

    # display segmented image
    print_image("MeanShift Segmentation of image", segmentedImg)
    # saving segmented image
    cv.imwrite("./ex3_images/ex3_results/meanshift_%s.png" % noise, segmentedImg)

    # convert image to binary to compare the results
    ret, binary_res = cv.threshold(segmentedImg, 128, 255, cv.THRESH_BINARY)
    print_image("Binary Result", binary_res)

    # 5. Displaying the Mean Shift segmentation result with plot
    plt.figure("Plot of Mean Shift Segmentation of the image")
    plt.clf()

    colours = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colours):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(flatImg[my_members, 0], flatImg[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters K-Means for')
    plt.show()

    # Evaluating the segmentation with F1 score from sklearn.metrics
    #f1 = get_f1_score(binary_res, binaryImg)
    #print('F1 score: %f' % f1)

    # K-means algorithm
    print('Using kmeans algorithm!')
    km = MiniBatchKMeans(n_clusters=n_clusters_)
    km.fit(flatImg)
    labels = km.labels_

    # Displaying segmented image
    segmentedImg = np.reshape(labels, originShape[:2])
    segmentedImg = label2rgb(segmentedImg) * 255  # need this to work with cv2. imshow

    print_image("K-Means Segments", segmentedImg.astype(np.uint8))
    cv.imwrite("./ex3_images/ex3_results/kmeans_%s.png" % noise, segmentedImg)  # saving segmented image

    # Evaluating the segmentation with F1 score from sklearn.metrics
    # f1 = get_f1_score(binary_res, binaryImg)
    # print('F1 score: %f' % f1)


if __name__ == '__main__':
    # 1. Choosing an image with an airplane in the sky
    # loading plane image in BGR
    originalImg = cv.imread('./ex3_images/aircraft-airplane.png')
    # display the original image
    print_image("Original Image", originalImg)

    # convert to RGB first
    originImgRGB = cv.cvtColor(originalImg, cv.COLOR_BGR2RGB)
    originImgGRAY = cv.cvtColor(originalImg, cv.COLOR_BGR2GRAY)
    # binary image of the plane
    binaryImg = cv.imread("./ex3_images/aircraft-airplane-binary.png")

    # display the binary image
    print_image("Binary Image", binaryImg)

    noise_perc = [0, 0.05, 0.10, 0.15, 0.20]
    for per in noise_perc:
        # Applying the noise percentage
        noisyImg = noise(originImgRGB, per)
        # display the noisy image
        print_image("Noisy Image with " + str(per) + "%", noisyImg)

        # saving the image to folder ex3_results
        cv.imwrite('./ex3_images/ex3_results/noisy_' + str(per) + '_perc.png', noisyImg)

        # Segmenting the noisy rgb image
        segment_image(noisyImg, str(int(per * 100)), binaryImg)