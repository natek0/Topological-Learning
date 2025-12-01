import numpy as np
import pandas as pd

def generate_regime_data(n_samples=1000):
    """
    Generates a time series with two distinct topological regimes.
    Regime 0: Stable periodic signal (Circle in phase space).
    Regime 1: Chaotic/Noisy signal (Filled ball/mess in phase space).
    """
    t = np.linspace(0, 50, n_samples)
    
    # First half: Clean Sine Wave (Stable Market)
    y1 = np.sin(t[:n_samples//2]) 
    labels1 = np.zeros(n_samples//2)
    
    # Second half: Noisy/Chaotic (Volatile Market)
    # We increase frequency and add heavy noise
    y2 = np.sin(3 * t[n_samples//2:]) + np.random.normal(0, 0.5, n_samples//2)
    labels2 = np.ones(n_samples - n_samples//2)
    
    y = np.concatenate([y1, y2])
    labels = np.concatenate([labels1, labels2])
    
    df = pd.DataFrame({'value': y, 'regime': labels})
    return df

if __name__ == "__main__":
    df = generate_regime_data()
    df.to_csv("data/synthetic_regimes.csv", index=False)
    print("Synthetic data generated: data/synthetic_regimes.csv")
