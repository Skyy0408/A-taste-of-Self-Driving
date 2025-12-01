import pandas as pd
import os
import pickle

file_path = 'traffic-data/processed/data_i80_v0/trajectories-0515-0530/car10.pkl'
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print("Keys:", data.keys())
        for k, v in data.items():
            print(f"Key: {k}, Type: {type(v)}")
            if hasattr(v, 'shape'):
                print(f"  Shape: {v.shape}")
            if hasattr(v, 'head'):
                print(f"  Head:\n{v.head()}")
except Exception as e:
    print(f"Error reading pickle: {e}")
