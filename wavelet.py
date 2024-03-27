import numpy as np
import pywt
import cv2


import cv2
import numpy as np

def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # Convert to grayscale if the input image has multiple channels
    if len(imArray.shape) > 2 and imArray.shape[2] == 3:
        imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255;  # Normalize to [0,1]

    # Compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;  # Remove low-frequency
    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)

    return imArray_H
