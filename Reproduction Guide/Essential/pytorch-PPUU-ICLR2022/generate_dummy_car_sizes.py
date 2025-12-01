import torch
import os
import re

def generate_dummy_car_sizes(data_dir):
    print(f"Generating dummy car_sizes.pth in {data_dir}")
    
    car_sizes = {}
    default_size = [14.3, 6.4]
    
    # Iterate over subdirectories
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for subdir in subdirs:
        print(f"Processing {subdir}")
        car_sizes[subdir] = {}
        
        # Load all_data.pth to get car IDs
        pth_path = os.path.join(data_dir, subdir, "all_data.pth")
        if os.path.exists(pth_path):
            try:
                data = torch.load(pth_path, weights_only=False)
                ids = data.get("ids", [])
                
                for car_path in ids:
                    # Extract car ID
                    # Format: .../trajectories-0400-0415/car1.pkl
                    match = re.search(r"car(\d+).pkl", car_path)
                    if match:
                        car_id = int(match.group(1))
                        car_sizes[subdir][car_id] = torch.tensor(default_size)
            except Exception as e:
                print(f"Error loading {pth_path}: {e}")
        else:
            print(f"Warning: {pth_path} not found")

    output_path = os.path.join(data_dir, "car_sizes.pth")
    torch.save(car_sizes, output_path)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    data_dir = "traffic-data/processed/data_i80_v0"
    generate_dummy_car_sizes(data_dir)
