import numpy as np

def calculate_distance(feature, dataset_features, distance_metric):
    if distance_metric == 'Euclidienne':
        distances = np.linalg.norm(dataset_features - feature, axis=1)
    elif distance_metric == 'Manhattan':
        distances = np.sum(np.abs(dataset_features - feature), axis=1)
    elif distance_metric == 'Chebyshev':
        distances = np.max(np.abs(dataset_features - feature), axis=1)
    elif distance_metric == 'Canberra':
        distances = np.sum(np.abs(dataset_features - feature) / (np.abs(dataset_features) + np.abs(feature)), axis=1)
    return distances
