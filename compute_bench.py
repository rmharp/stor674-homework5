import torch
import time

def run_computation(n, device):
    """Run computation on specified device"""
    torch.manual_seed(42)  # for reproducibility
    
    start_time = time.time()
    x = torch.randn(n, device=device) ** 4
    if device.type == "cuda":
        torch.cuda.synchronize()  # ensure GPU computation is complete
    end_time = time.time()
    
    return end_time - start_time

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU available: {device_name}")
    else:
        print("No GPU available, running on CPU only")
    
    # Test sizes
    sizes = [10**7, 10**8]
    
    # CPU computations
    print("\nCPU Computations:")
    for n in sizes:
        time_taken = run_computation(n, torch.device('cpu'))
        print(f"Time for {n:,} elements: {time_taken:.4f} seconds")
    
    # GPU computations (if available)
    if torch.cuda.is_available():
        print("\nGPU Computations:")
        for n in sizes:
            try:
                time_taken = run_computation(n, torch.device('cuda'))
                print(f"Time for {n:,} elements: {time_taken:.4f} seconds")
            except RuntimeError as e:
                print(f"Error processing {n:,} elements: {str(e)}")
    
    # Generate small sample (equivalent to x = rnorm(10))
    x = torch.randn(10)
    print(f"\nSmall sample of 10 random numbers:\n{x}")
    
    # Save the tensor (similar to save.image in R)
    torch.save({'sample': x}, "mydata.pt")
    print("\nSaved data to mydata.pt")

if __name__ == "__main__":
    main()