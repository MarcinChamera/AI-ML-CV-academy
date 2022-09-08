import cv2
import numpy as np

# @param target_image - target image
# @param source_image - source image
# @param arucoDict - specifies the type of markers which are detected
# @param arucoParams - detector parameters
# @return - result image
def ar_markers_homography(target_image, source_image, arucoDict, arucoParams):
    (trgH, trgW) = target_image.shape[:2]

    # Detect the ArUco markers in the input image
    #
    # Parameters:
    # - The first parameter is the image containing the markers to be detected
    # - The second parameter is the dictionary object, in this case one of the predefined dictionaries
    # - The third parameter is the object of type DetectionParameters. This object includes all the parameters that can
    #   be customized during the detection process
    #
    # Return values:
    # - 'corners' is the list of corners of the detected markers. For each marker, its four corners are returned in
    #   their original order (which is clockwise starting with top left). So, the first corner is the top left corner,
    #   followed by the top right, bottom right and bottom left.
    # - 'ids' is the list of ids of each of the detected markers in 'corners'. Note that the returned 'corners'
    #   and 'ids' vectors have the same size.
    # - 'rejected' is a returned list of marker candidates, i.e. shapes that were found and considered but did not
    #   contain a valid marker. Each candidate is also defined by its four corners, and its format is the same as the
    #   'corners' return value.
    (corners, ids, rejected) = cv2.aruco.detectMarkers(target_image, arucoDict, parameters=arucoParams)

    # if we have not found four markers in the input image then we cannot
    # apply our augmented reality technique
    if len(corners) != 4:
        return None

    ids = ids.flatten()
    refPts = []

    # hardcoded ids in order of occurrence on pantone card
    for i in (923, 1001, 241, 1007):
        # grab the index of the corner with the current ID and append the
        # corner (x, y)-coordinates to our list of reference points
        j = np.squeeze(np.where(ids == i))
        try:
            corner = np.squeeze(corners[j])
        except TypeError:
            print(f"only integer scalar arrays can be converted to a scalar index: {j} is not an integer")
        refPts.append(corner)

    # take special care to ensure the reference points of the ArUco markers are provided in top-left, top-right,
    # bottom-right, and bottom-left order.
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    trgMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    trgMat = np.array(trgMat)

    # provide the (x, y)-coordinates of the top-left, top-right, bottom-right, and bottom-left coordinates of the
    # source image
    (srcH, srcW) = source_image.shape[:2]
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

    # compute homography matrix and warp the source image such that it can fit into the area provided in the destination
    # matrix
    (H, _) = cv2.findHomography(srcMat, trgMat)
    warped = cv2.warpPerspective(source_image, H, (trgW, trgH))

    # create an empty mask with the same spatial dimensions as the target image
    mask = np.zeros(target_image.shape, dtype=np.uint8)

    # fill the polygon area with white, implying that the area we just drew is foreground and the rest is background
    cv2.fillConvexPoly(mask, trgMat.astype("int32"), (255, 255, 255))

    # Invert the mask color
    inverted_mask = cv2.bitwise_not(mask)

    # Bitwise AND the mask with the target image
    masked_image = cv2.bitwise_and(target_image, inverted_mask)

    # adding the resulting multiplications together
    output = cv2.bitwise_or(warped, masked_image)

    output = output.astype("uint8")
    return output
