import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TakensEmbedding(BaseEstimator, TransformerMixin):
    """
    Transforms a 1D time series into a high-dimensional point cloud
    using Takens' Time-Delay Embedding Theorem.
    
    Theory: V_t = [x_t, x_{t-tau}, ..., x_{t-(m-1)tau}]
    """
    
    def __init__(self, outer_window_duration: int = 20, time_delay: int = 1, embedding_dimension: int = 3):
        """
        args:
            outer_window_duration: How many time steps (days) to look at in total for one 'shape'.
            time_delay (tau): The stride between lags.
            embedding_dimension (m): The dimension of the resulting topological space.
        """
        self.outer_window_duration = outer_window_duration
        self.time_delay = time_delay
        self.embedding_dimension = embedding_dimension

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: 1D numpy array of time series data.
        Returns: A collection of Point Clouds.
        """
        # Simple sliding window implementation for the portfolio
        n_points = len(X)
        if n_points < self.outer_window_duration:
            raise ValueError("Data length is shorter than window duration")
            
        point_clouds = []
        
        # We slide a window over the whole series
        # For each window, we create a point cloud
        num_windows = n_points - self.outer_window_duration + 1
        
        for i in range(num_windows):
            window_data = X[i : i + self.outer_window_duration]
            
            # Embed this specific window into R^m
            cloud = self._embed_window(window_data)
            point_clouds.append(cloud)
            
        return np.array(point_clouds)

    def _embed_window(self, window):
        # Create the trajectory matrix
        N = len(window)
        m = self.embedding_dimension
        tau = self.time_delay
        
        # Number of vectors we can form from this window
        M = N - (m - 1) * tau
        
        if M <= 0:
            # Fallback for very small windows: just return noise to prevent crash
            return np.zeros((1, m))
            
        trajectory = np.zeros((M, m))
        
        for i in range(M):
            for j in range(m):
                trajectory[i, j] = window[i + j * tau]
                
        return trajectory
