import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import data
from math import floor, ceil
from skimage.feature import greycomatrix, greycoprops

'''
P4-D ndarray
The gray-level co-occurrence histogram. 
The value P[i,j,d,theta] is 
    the number of times that gray-level j occurs 
    at a distance d and 
    at an angle theta from gray-level i. 
    If normed is False, the output is of type uint32, 
    otherwise it is float64. 
    The dimensions are: 
    levels x levels x number of distances x number of angles.
'''
def calcu_glcm(img, min_gray_value, max_gray_value, nbits, distances, angles):
    # compress
    bins = np.linspace(min_gray_value, max_gray_value + 1, nbits + 1)
    img_compressed = np.digitize(img, bins) - 1
    # calcu_glcm
    glcm = greycomatrix(img_compressed, distances, angles, levels=nbits, symmetric=True)
    return glcm


def calcu_evalue(glcm):
    evalue = []
    evalue.append(greycoprops(glcm, prop='mean'))
    evalue.append(greycoprops(glcm, prop='variance'))
    evalue.append(greycoprops(glcm, prop='homogeneity'))
    evalue.append(greycoprops(glcm, prop='contrast'))
    evalue.append(greycoprops(glcm, prop='dissimilarity'))
    evalue.append(greycoprops(glcm, prop='entroy'))
    evalue.append(greycoprops(glcm, prop='energy'))
    evalue.append(greycoprops(glcm, prop='correlation'))
    evalue.append(greycoprops(glcm, prop='ASM'))
    result = []
    for e in evalue:
        temp = 0
        for i in range(glcm.shape[2]):
            for j in range(glcm.shape[3]):
                temp += e[i][j]
        temp /= (glcm.shape[2] * glcm.shape[3])
        result.append(temp)
    return result
    