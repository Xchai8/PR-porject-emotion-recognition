import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

n_neighbor = 6
n_component = 18

def knn_estimator(data, label, n_neighbors=n_neighbor):
    
    knn = Pipeline([('scaler', StandardScaler()),
                    ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))])
    knn.fit(data, label)
    
    return knn

def knn_pca_estimator(data, label, n_neighbors=n_neighbor, n_components=n_component):

    knn_pca = Pipeline([('scaler', StandardScaler()),
                        ('PCA', PCA(n_components=n_components)),
                        ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))])
    knn_pca.fit(data, label)

    return knn_pca