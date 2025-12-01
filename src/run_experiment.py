import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pipeline import create_tda_pipeline
from scripts.generate_data import generate_regime_data
from src.visualize_attractor import plot_attractor

def main():
    print("--- Topological Geometric Learning Experiment ---")
    
    # 1. Get Data
    data_path = "data/synthetic_regimes.csv"
    if not os.path.exists(data_path):
        print("Generating synthetic data...")
        df = generate_regime_data()
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
        
    raw_values = df['value'].values
    labels = df['regime'].values
    
    # 2. Prepare Sliding Windows for the Classifier
    # We need to map windows of data -> single label
    window_size = 30
    X_windows = []
    y_windows = []
    
    for i in range(len(raw_values) - window_size):
        X_windows.append(raw_values[i : i+window_size])
        # Use the label of the last point in the window
        y_windows.append(labels[i+window_size])
        
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    
    print(f"Data Shape: {X_windows.shape} samples")
    
    # 3. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, shuffle=False)
    
    # 4. Initialize Pipeline
    print("Initializing TDA Pipeline (Takens -> Persistence -> Entropy -> Random Forest)...")
    pipeline = create_tda_pipeline()
    
    # 5. Train
    print("Training... (This involves calculating homology for thousands of point clouds, may take a moment)")
    pipeline.fit(X_train, y_train)
    
    # 6. Predict
    print("Predicting on Test Set...")
    y_pred = pipeline.predict(X_test)
    
    # 7. Results
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # 8. Visualize
    print("\nGenerating Phase Space Plot...")
    fig = plot_attractor(data_path)
    fig.write_html("attractor_shape.html")
    print("Saved visualization to 'attractor_shape.html'")
    os.system("open attractor_shape.html")

if __name__ == "__main__":
    main()
