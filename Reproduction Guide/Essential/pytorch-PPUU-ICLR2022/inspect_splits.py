import torch
try:
    splits = torch.load("traffic-data/processed/data_i80_v0/splits.pth", weights_only=False)
    print("Keys:", splits.keys())
    print("Train:", splits.get("train_indx"))
    print("Val:", splits.get("valid_indx"))
    print("Test:", splits.get("test_indx"))
except Exception as e:
    print(e)
