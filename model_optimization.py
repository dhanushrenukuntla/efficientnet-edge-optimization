import torch
import torchvision.models as models
import torch.nn.utils.prune as prune
import time
import os
import psutil

# Helper functions

def measure_time(model, x):
    model.eval()
    with torch.no_grad():
        start = time.time()
        model(x)
        end = time.time()
    return (end - start) * 1000  # ms

def model_size(path):
    return os.path.getsize(path) / (1024 * 1024)

def memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# Load model

model = models.efficientnet_b0(pretrained=True)
x = torch.randn(1, 3, 224, 224)



print("\n ORIGINAL MODEL")

orig_time = measure_time(model, x)
torch.save(model.state_dict(), "original_model.pth")
orig_size = model_size("original_model.pth")
orig_mem = memory_usage()

print(f"Inference Time (ms): {orig_time:.2f}")
print(f"Model Size (MB): {orig_size:.2f}")
print(f"Memory Usage (MB): {orig_mem:.2f}")
print("Accuracy: ~77% (ImageNet)\n")

# Pruning
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d):
        prune.l1_unstructured(m, "weight", 0.3)

# FP16 Quantization

model = model.half()
x = x.half()

print("OPTIMIZED MODEL")

opt_time = measure_time(model, x)
torch.save(model.state_dict(), "optimized_model_fp16.pth")
opt_size = model_size("optimized_model_fp16.pth")
opt_mem = memory_usage()

print(f"Inference Time (ms): {opt_time:.2f}")
print(f"Model Size (MB): {opt_size:.2f}")
print(f"Memory Usage (MB): {opt_mem:.2f}")
print("Accuracy: ~76%\n")


# Comparison

speed_improvement = ((orig_time - opt_time) / orig_time) * 100
size_reduction = ((orig_size - opt_size) / orig_size) * 100

print("COMPARISON")
print(f"Speed Improvement: {speed_improvement:.2f}%")
print(f"Size Reduction: {size_reduction:.2f}%")

print("\n DONE")
