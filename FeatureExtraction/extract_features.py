import numpy as np
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

def extract_glcm_features(image):
    
    image_gray = image.convert('L')
    image_array = np.array(image_gray)
    
  
    glcm = graycomatrix(image_array, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm])

def extract_bit_features(image):
    return np.random.rand(10)
