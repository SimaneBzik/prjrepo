import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

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

def process_images(image_folder):
    glcm_features = []
    bit_features = []
    for subdir, _, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(subdir, filename)
                image = Image.open(image_path)
                glcm_feat = extract_glcm_features(image)
                bit_feat = extract_bit_features(image)
                glcm_features.append(glcm_feat)
                bit_features.append(bit_feat)
    glcm_features = np.array(glcm_features)
    bit_features = np.array(bit_features)
    return glcm_features, bit_features

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
image_folder = os.path.join(base_dir, 'datasets')
glcm_features, bit_features = process_images(image_folder)
np.save(os.path.join(base_dir, 'FeatureExtraction', 'glcm_features.npy'), glcm_features)
np.save(os.path.join(base_dir, 'FeatureExtraction', 'bit_features.npy'), bit_features)
print(f"Caractéristiques sauvegardées avec succès dans {os.path.join(base_dir, 'FeatureExtraction')}")
print(f"Chemin des caractéristiques GLCM: {os.path.join(base_dir, 'FeatureExtraction', 'glcm_features.npy')}")
print(f"Chemin des caractéristiques BiT: {os.path.join(base_dir, 'FeatureExtraction', 'bit_features.npy')}")
