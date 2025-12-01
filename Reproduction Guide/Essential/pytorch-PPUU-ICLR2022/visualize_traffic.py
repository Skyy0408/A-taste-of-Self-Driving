import sys
import os
import torch
import numpy as np
import imageio


# Add current directory to sys.path
sys.path.append(os.getcwd())

# from ppuu.data.dataloader import DataStore

def create_animation(data_path, output_path):
    # Construct path to all_data.pth
    # Assuming data_path is traffic-data/processed/data_i80_v0
    # We pick the first subdirectory
    subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if not subdirs:
        print(f"No subdirectories found in {data_path}")
        return
    
    pth_path = os.path.join(data_path, subdirs[0], "all_data.pth")
    print(f"Loading data from {pth_path}")
    
    if not os.path.exists(pth_path):
        print(f"{pth_path} does not exist!")
        return

    data = torch.load(pth_path)
    images_list = data.get("images")
    
    if not images_list or len(images_list) == 0:
        print("No images found!")
        return

    # Get the first episode
    episode_idx = 0
    images = images_list[episode_idx]
    
    print(f"Processing episode {episode_idx}, shape: {images.shape}")
    
    # Images are likely (T, C, H, W)
    T, C, H, W = images.shape
    
    # Create video writer
    fps = 10
    writer = imageio.get_writer(output_path, fps=fps)
    
    for t in range(T):
        img_tensor = images[t]
        # Permute to H, W, C
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Check value range and convert to uint8
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
            
        writer.append_data(img_np)
        
    writer.close()
    print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    data_path = "traffic-data/processed/data_i80_v0"
    output_path = "traffic_animation.mp4"
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist!")
        sys.exit(1)
        
    create_animation(data_path, output_path)
