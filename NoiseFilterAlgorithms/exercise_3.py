# --------------------------------------------------------- #
# Project 1. Exercise 3                                     #
# Evaluate the similarity between numbers provided in the   #
# MNIST dataset using the frequency field.                  #
#                                                           #
# Michail Apostolidis                                       #
# --------------------------------------------------------- #

# import necessary packages
import cv2 as cv  # opencv
import numpy as np  # for arrays
import skimage.metrics
from matplotlib import pyplot as plt
from keras.datasets import mnist


# 4. Capture the images in the frequency field using the Fourier transform.
def fourier_trans(img, num, j):
    # add fourier transform
    f = np.fft.fft2(img)  # calculating fourier
    fshift = np.fft.fftshift(f)  # remove zero frequency to the center

    # calculate magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    magnitude_spectrum_img = np.round(magnitude_spectrum).astype('uint8')

    plt.imshow(magnitude_spectrum_img, cmap='gray')
    plt.title('Mag Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv.namedWindow("Mag Spectrum", cv.WINDOW_NORMAL)
    cv.imshow("Mag Spectrum", magnitude_spectrum_img)
    cv.waitKey(0)
    cv.destroyWindow("Mag Spectrum")

    # writing the fourier image:
    cv.imwrite("./images_saved_ex_3/fft_magnum_num_" + str(num) + "_" + str(j) + ".jpg", magnitude_spectrum_img)

    return fshift


# 5. Display four pairing pairs (spatial – frequency, one for each number)
def spatial_to_freq(img, fshift):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0

    # now you go back to the original image
    f_ishift = np.fft.ifftshift(fshift)  # get DC back to original space
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.round(np.real(img_back)).astype('uint8')  # we need this for the opencv library

    cv.namedWindow("Frequency to Spatial", cv.WINDOW_NORMAL)  # this allows for resizing using mouse
    cv.imshow("Frequency to Spatial", img_back)
    cv.resizeWindow("Frequency to Spatial", 480, 360)

    # 6. Getting the ssim to array

    return img_back


# 8. Applying laplacian transformation to the freq image
def laplacian(image):
    # removing noise with guassian filter
    img = cv.GaussianBlur(image, (3, 3), 0)

    # convolute with proper kernels
    laplacian = cv.Laplacian(img, cv.CV_64F)

    return laplacian


# 1. load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

numbers_array = [3, 5, 8, 9]  # array of numbers 3, 5, 8, 9

for i in numbers_array:  # for each number
    # filters for each number:
    train_filter = np.where(trainY == i)
    test_filter = np.where(testY == i)

    Xtrain, Ytrain = trainX[train_filter], trainY[train_filter]
    Xtest, Ytest = testX[test_filter], testY[test_filter]

    random_number = np.random.randint(0, 5)  # number for spatial to freq
    # 2. Keep 5 random images of the numbers
    # finding 5 random images of each number in MNIST dataset
    for j in range(5):
        rand_num = np.random.randint(0, 5200)  # to select random num for each number
        rand_img = Xtrain[rand_num]  # picking the image of the number

        # 3. Displaying the images as plot
        plt.imshow(rand_img, cmap='gray')
        plt.show()

        # saving each image number to images_saved_ex_3 directory:
        cv.imwrite("./images_saved_ex_3/numb_" + str(i) + "_" + str(j) + ".jpg", rand_img)

        # 4. Applying Fourier to each number image
        fshift = fourier_trans(rand_img, i, j)

        # 5. For one of each numbers apply the freq to spatial
        if j == random_number:  # runs once every repetition (for only one image)
            spat_freq_img = spatial_to_freq(rand_img, fshift)
            laplacian_img = laplacian(spat_freq_img)
            plt.imshow(laplacian_img, cmap='gray')
            plt.show()
            cv.imwrite("./images_saved_ex_3/laplacian_numb_" + str(i) + ".jpg", laplacian_img)
