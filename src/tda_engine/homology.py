import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class TopologicalFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts scalar features from point clouds using Persistent Homology.
    
    Pipeline:
    1. Point Cloud -> Persistence Diagram (Vietoris-Rips Filtration)
    2. Diagram -> Entropy (Scalar value representing 'complexity' of the shape)
    """
    
    def __init__(self, homology_dimensions=(0, 1)):
        """
        homology_dimensions: 
            0 = Connected Components (Clusters)
            1 = Loops (Cycles/Holes)
            2 = Voids
        """
        self.homology_dimensions = homology_dimensions
        self.pipeline = Pipeline([
            ('persistence', VietorisRipsPersistence(
                metric='euclidean',
                homology_dimensions=homology_dimensions,
                n_jobs=-1 # Use all CPU cores
            )),
            ('entropy', PersistenceEntropy())
        ])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Input: Array of Point Clouds (Samples, Points, Dimension)
        Output: Array of Topological Features (Samples, Features)
        """
        # X is shape (N_samples, Window_Size, Embedding_Dim)
        
        # Calculate Persistence Entropy
        # This condenses the complex diagram into a single number per dimension
        features = self.pipeline.fit_transform(X)
        
        return features
