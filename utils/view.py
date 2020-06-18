import cv2
import numpy as np

def gray_masking(img, mask):
    """
    Mask an image by grayscaling the background while mainting the color of the foreground

    Args:
    ---
        img(cv2 format): image to apply the mask on
        mask(cv2 format): one channel array to apply 
    """
    # Create the inverted mask
    mask_inv = cv2.bitwise_not(mask)

    # Convert to grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract the dimensions of the original image
    rows, cols, channels = img.shape
    img = img[0:rows, 0:cols]

    # Bitwise-OR mask and original image
    colored_portion = cv2.bitwise_or(img, img, mask = mask)
    colored_portion = colored_portion[0:rows, 0:cols]

    # Bitwise-OR inverse mask and grayscale image
    gray_portion = cv2.bitwise_or(gray, gray, mask = mask_inv)
    gray_portion = np.stack((gray_portion,)*3, axis=-1)

    # Combine the two images
    output = colored_portion + gray_portion

    return output
