import torch

# Load the saved model
model_path = 'checkpoints/weights.pth'  # Replace with your model path
state_dict = torch.load(model_path, map_location='cpu')

# Initialize counters
total_params = 0
dtype_counts = {}

# Iterate through all parameters
for name, param in state_dict.items():
    num_params = param.numel()
    total_params += num_params
    
    # Track data types
    dtype = str(param.dtype)
    if dtype in dtype_counts:
        dtype_counts[dtype] += num_params
    else:
        dtype_counts[dtype] = num_params

# Print results
print(f"Total parameters: {total_params:,}")
print(f"Total parameters (millions): {total_params / 1e6:.2f}M")
print("\nData types:")
for dtype, count in dtype_counts.items():
    percentage = (count / total_params) * 100
    print(f"  {dtype}: {count:,} params ({percentage:.1f}%)")