import cv2
import imutils
import sys
import time
import os
from our_modules.ar_markers_homography import ar_markers_homography


# Based on pyimagesearch article: https://pyimagesearch.com/2021/01/04/opencv-augmented-reality-ar/

# ArUco markers are 2D binary patterns that computer vision algorithms can easily detect

# How it's done:
# 1. Detect each of the four ArUco markers
# 2. Sort them in top-left, top-right, bottom-left, and bottom-right order
# 3. Apply augmented reality by transforming a source image onto the region defined by markers - in this case
#    Pantone card

def augmented_reality_from_image():
    # Pantone color match card with ArUco markers on it
    target_image = cv2.imread('img_resources/ar_homography/pantone_card.png')

    # Source image that will be transformed onto the input
    source_image = cv2.imread('img_resources/ar_homography/pikachu-pokemon.jpg')

    folder_output = 'img_resources/output/'
    output_filename = folder_output + 'augmented_reality.jpg'

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    target_image = imutils.resize(target_image, width=600)

    # Dictionary_get function tells OpenCV which ArUco dictionary we are using, how to draw the tags, etc.
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

    # Initialize ArUco detector parameters
    arucoParams = cv2.aruco.DetectorParameters_create()

    output = ar_markers_homography(target_image, source_image, arucoDict, arucoParams)

    if output is not None:
        cv2.imwrite(output_filename, output)
    else:
        print('AR homography failed')
        sys.exit(0)


def augmented_reality_live():
    source_image = cv2.imread('img_resources/ar_homography/pikachu-pokemon.jpg')

    # try to show to a camera "pantone_card_close_up.png" image on a smartphone
    capture = cv2.VideoCapture(0)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()

    frame_rate = 10
    prev = 0

    while True:
        time_elapsed = time.time() - prev
        ret, target_image = capture.read()

        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            target_image = imutils.resize(target_image, width=600)
            output = ar_markers_homography(target_image, source_image, arucoDict, arucoParams)

            if output is not None:
                cv2.imshow('frame', output)
            else:
                cv2.imshow('frame', target_image)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    capture.release()

    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    # augmented_reality_from_image()
    augmented_reality_live()
