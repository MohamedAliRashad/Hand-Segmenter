from PyQt5.QtGui import QImage, QPixmap
import cv2
import torch
import numpy as np

def cv2qimg(img):
    """
    Transform cv2 image to QImage (pyqt)
    
    Args:
    ---
    
        img (cv2 format): img needed to be transformed
    """
    # Get img dimensions for QImage
    height, width, channel = img.shape

    # Concatenate channels on width
    bytesPerLine = 3 * width

    # Transform to QImage (rgbSwapped: because of BGR)
    qImg = QImage(
        img.data, width, height, bytesPerLine, QImage.Format_RGB888
    ).rgbSwapped()

    return QPixmap.fromImage(qImg)
