import os
from mesostat.visualization.opencv_video import merge_images_cv2


def merge_image_sequence_movie(pathprefix, suffix, idxMin, idxMax, trgPathName=None, deleteSrc=False):
    srcPwdLst = []
    for idx in range(idxMin, idxMax):
        pwd = pathprefix + str(idx) + suffix
        if not os.path.isfile(pwd):
            raise FileNotFoundError('File not found', pwd)
        srcPwdLst += [pwd]

    if trgPathName is None:
        trgPathName = os.path.splitext(pathprefix + suffix)[0]

    merge_images_cv2(srcPwdLst, trgPathName, fps=30, FOURCC='MJPG', isColor=True)

    if deleteSrc:
        for pwd in srcPwdLst:
            os.remove(pwd)
