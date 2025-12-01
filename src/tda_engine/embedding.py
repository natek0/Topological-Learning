import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TakensEmbedding(BaseEstimator, TransformerMixin):
    """
    Transforms a batch of time series windows into a batch of point clouds.
    
    Input X: (n_samples, window_size)
    Output: (n_samples, n_points_in_cloud, embedding_dimension)
    """
    
    def __init__(self, outer_window_duration: int = 20, time_delay: int = 1, embedding_dimension: int = 3):
        self.outer_window_duration = outer_window_duration
        self.time_delay = time_delay
        self.embedding_dimension = embedding_dimension

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: 2D numpy array of shape (n_samples, window_size).
        We treat each ROW as a separate time series window to be embedded.
        """
        # Validate input shape matches expected window size
        if X.shape[1] != self.outer_window_duration:
            # We warn but proceed, or could raise an error
            pass

        # Apply embedding to every row independently
        # Result is a list of Point Clouds
        point_clouds = [self._embed_window(row) for row in X]
            
        return np.array(point_clouds)

    def _embed_window(self, window):
        # Create the trajectory matrix for a single window
        N = len(window)
        m = self.embedding_dimension
        tau = self.time_delay
        
        # Number of vectors we can form from this window
        M = N - (m - 1) * tau
        
        if M <= 0:
            # Fallback for very small windows to prevent crash
            return np.zeros((1, m))
            
        trajectory = np.zeros((M, m))
        
        for i in range(M):
            for j in range(m):
                trajectory[i, j] = window[i + j * tau]
                
        return trajectory
