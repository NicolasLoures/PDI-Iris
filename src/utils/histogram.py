import matplotlib.pyplot as plt
import numpy as np
import cv2

def histograma(img: np.ndarray, bins: int=256) -> np.ndarray:
    hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
    return hist