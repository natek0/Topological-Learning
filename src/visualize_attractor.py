import pandas as pd
import plotly.express as px
from .tda_engine.embedding import TakensEmbedding

def plot_attractor(data_file="data/synthetic_regimes.csv"):
    df = pd.read_csv(data_file)
    values = df['value'].values
    
    # Embed into 3D space to see the "Shape"
    embedder = TakensEmbedding(outer_window_duration=10, time_delay=2, embedding_dimension=3)
    
    # We transform the whole series at once for visualization
    # Note: Takens returns a collection of point clouds. 
    # For visualization, we just want the trajectory of the single time series.
    # So we manually do a single embedding here:
    
    # Simple manual embedding for the plot
    x = values[:-2]
    y = values[1:-1]
    z = values[2:]
    
    # Color by regime (0=Stable, 1=Chaotic)
    regime = df['regime'].values[2:]
    
    fig = px.scatter_3d(
        x=x, y=y, z=z,
        color=regime,
        title="Phase Space Reconstruction (The 'Shape' of the Data)",
        labels={'x': 't', 'y': 't-1', 'z': 't-2'},
        opacity=0.6
    )
    return fig
