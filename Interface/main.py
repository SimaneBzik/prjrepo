import sys
import os
import streamlit as st
import numpy as np
from PIL import Image


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
st.write(f"Chemin ajouté: {base_dir}")

try:
    from FeatureExtraction.utils import calculate_distance
    from FeatureExtraction.extract_features import extract_glcm_features, extract_bit_features
    st.write("")
except ImportError as e:
    st.write("")


try:
    st.write("")
    glcm_features_path = os.path.join(base_dir, 'FeatureExtraction', 'glcm_features.npy')
    bit_features_path = os.path.join(base_dir, 'FeatureExtraction', 'bit_features.npy')
    glcm_features = np.load(glcm_features_path)
    bit_features = np.load(bit_features_path)
    st.write("")
except Exception as e:
    st.write(f" {e}")


st.title('Aploading Images')

# Disposition de la barre latérale pour les options de recherche
with st.sidebar:
    st.header('Options de Recherche')
    descriptor = st.radio('Choisir le descripteur', ('GLCM', 'BiT'))
    distance_metric = st.radio('Choisir la mesure de distance', ('Manhattan', 'Euclidienne', 'Chebyshev', 'Canberra'))
    num_results = st.number_input('Nombre d\'images similaires à afficher', min_value=1, max_value=20, value=5)

uploaded_file = st.file_uploader("Téléverser une image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.write("Image téléversée.")
    image = Image.open(uploaded_file)
    st.image(image, caption='Image Téléversée.', use_column_width=True)

    if st.button('Rechercher'):
        st.write("Recherche lancée.")
        try:
            if descriptor == 'GLCM':
                feature = extract_glcm_features(image)
                dataset_features = glcm_features
            else:
                feature = extract_bit_features(image)
                dataset_features = bit_features

            
            st.write("Caractéristiques extraites de l'image téléversée avec succès.")

           
            distances = calculate_distance(feature, dataset_features, distance_metric)
            st.write("Distances calculées avec succès.")

            
            image_folder = os.path.join(base_dir, 'datasets')
            st.write(f"Chemin du dossier d'images: {image_folder}")

            
            if not os.path.exists(image_folder):
                st.write(f"Erreur: Le dossier {image_folder} n'existe pas.")
            else:
              
                image_files = []
                for subdir, _, files in os.walk(image_folder):
                    for file in files:
                        if file.endswith('.jpg') or file.endswith('.png'):
                            image_files.append(os.path.join(subdir, file))
                st.write(f" {len(image_files)}")

               
                if num_results > len(image_files):
                    num_results = len(image_files)

            
                sorted_indices = np.argsort(distances)[:num_results]
                st.write(f" {sorted_indices}")

                st.subheader('Images Similaires:')
                cols = st.columns(5)  # Disposer les images en colonnes
                for idx, col in zip(sorted_indices, cols):
                    result_image_path = image_files[idx]
                    result_image = Image.open(result_image_path)
                    col.image(result_image, caption=f'Similarité: {distances[idx]:.4f}', use_column_width=True)
        except Exception as e:
            st.write(f" {e}")
else:
    st.write("")
