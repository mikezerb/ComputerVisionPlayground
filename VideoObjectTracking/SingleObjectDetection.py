# importing libraries
import cv2
import matplotlib
import numpy as np
import imutils
import sklearn.metrics
import matplotlib.pyplot


def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


if __name__ == '__main__':
    # Initializing the HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Read video:
    # loading a video to use object detection
    cap = cv2.VideoCapture('./Single_object/ped_walking.mp4')
    i = 0  # var for saving video name
    # While loop for video
    while cap.isOpened():
        i = i + 1  # for saving images
        # Reading the video stream
        ret, image = cap.read()
        if ret:
            image = imutils.resize(image,
                                   width=min(400, image.shape[1]))

            # Detecting all the regions
            # in the Image that has a
            # pedestrians inside it
            (regions, _) = hog.detectMultiScale(image,
                                                winStride=(2, 2),
                                                padding=(2, 2),
                                                scale=1.01)

            # Drawing the regions in the
            # Image
            for (x, y, w, h) in regions:
                cv2.rectangle(image, (x, y),
                              (x + w, y + h),
                              (0, 0, 255), 2)

            # Showing the output Image
            cv2.imshow("HOG Detector", image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            else:
                cv2.imwrite("./Single_object/results/HOG/hog_det_" + str(i) + ".png", image)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Using Haar Cascade classifier
    # Create our body classifier
    body_classifier = cv2.CascadeClassifier('./Single_object\Haarcascades\haarcascade_fullbody.xml')

    # Initiate video capture for video file
    cap = cv2.VideoCapture('./Single_object/ped_walking.mp4')
    i = 0
    # While loop for video
    while cap.isOpened():
        i = i + 1  # for saving images
        # Reading the video stream
        # read frame
        ret, image = cap.read()

        # resize the frame
        image = imutils.resize(image,
                               width=min(400, image.shape[1]))

        # convert to grayscale
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # set body classifier
        body = body_classifier.detectMultiScale(grey, 1.5, 3)

        # Extract bounding boxes for any bodies identified
        for (x, y, w, h) in body:
            cv2.rectangle(image, (x, y),
                          (x + w, y + h),
                          (255, 255, 255), 2)
        # Showing the output Image
        cv2.imshow('HAAR Detector', image)
        k = cv2.waitKey(30) & 0XFF
        if k == 27:
            break
        else:
            cv2.imwrite("./Single_object/results/HAAR/haag_det_" + str(i) + ".png", image)

    cap.release()
    cv2.destroyAllWindows()
    y_true = ["positive", "negative", "negative", "positive", "positive", "positive", "negative", "positive",
              "negative", "positive", "positive", "positive", "positive", "negative", "negative", "negative"]

    pred_scores = [0.7, 0.3, 0.5, 0.6, 0.55, 0.9, 0.4, 0.2, 0.4, 0.3, 0.7, 0.5, 0.8, 0.2, 0.3, 0.35]

    thresholds = np.arange(start=0.2, stop=0.7, step=0.05)

    precisions, recalls = precision_recall_curve(y_true=y_true,
                                                 pred_scores=pred_scores,
                                                 thresholds=thresholds)

    matplotlib.pyplot.plot(recalls, precisions, linewidth=4, color="red")
    matplotlib.pyplot.xlabel("Recall", fontsize=12, fontweight='bold')
    matplotlib.pyplot.ylabel("Precision", fontsize=12, fontweight='bold')
    matplotlib.pyplot.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    matplotlib.pyplot.show()

    f1 = 2 * ((np.array(precisions) * np.array(recalls)) / (np.array(precisions) + np.array(recalls)))

    matplotlib.pyplot.plot(recalls, precisions, linewidth=4, color="red", zorder=0)
    matplotlib.pyplot.scatter(recalls[5], precisions[5], zorder=1, linewidth=6)

    matplotlib.pyplot.xlabel("Recall", fontsize=12, fontweight='bold')
    matplotlib.pyplot.ylabel("Precision", fontsize=12, fontweight='bold')
    matplotlib.pyplot.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    matplotlib.pyplot.show()
