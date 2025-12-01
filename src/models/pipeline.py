from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ..tda_engine.embedding import TakensEmbedding
from ..tda_engine.homology import TopologicalFeatures

def create_tda_pipeline():
    """
    Constructs the full end-to-end learning pipeline.
    
    Flow:
    Raw Time Series -> Takens Embedding (Point Cloud) -> TDA (Entropy) -> Classifier
    """
    return Pipeline([
        ('embedder', TakensEmbedding(
            outer_window_duration=30, 
            time_delay=2, 
            embedding_dimension=3
        )),
        ('topology', TopologicalFeatures(homology_dimensions=(0, 1))),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])
