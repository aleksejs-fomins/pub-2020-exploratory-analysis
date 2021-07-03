from PIL import Image
import numpy as np


# Construct a list of 48 colors, incrementing R, G, B in steps of 16 consecutively
def _color_list_48():
    step = 16
    nCol = 16

    colLst = []
    for idxCol in range(3):
        for idxSh in range(nCol):
            col = np.array([0, 0, 0])
            col[idxCol] = np.clip((idxSh + 1) * step, 0, 255)
            colLst += [col]
    return colLst


# Convert an RGB image into a grayscale image
# Each area is numbered by ascending integers
# 0 - background
# 1 - borders
# 2-49 - areas
def remap_area_image(img):
    mask = lambda img, c: np.all(img == c, axis=2).astype(int)

    # Init resulting image
    rez = np.zeros(img.shape[:2]).astype(int)

    # Map borders
    m = mask(img, np.array([0, 0, 0]))
    rez += 1 * m

    # Map areas
    colLst = _color_list_48()
    for i, c in enumerate(colLst):
        rez += (i + 2) * mask(img, c)

    return rez


