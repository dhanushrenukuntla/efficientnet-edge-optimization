## EfficientNet-B0 Model Optimization for Edge Devices

### Description
This project optimizes a pre-trained EfficientNet-B0 model for edge deployment using
pruning and FP16 quantization.

### Requirements
pip install torch torchvision numpy psutil

### How to Run
python model_optimization.py

### Output Files
- original_metrics.txt
- optimized_metrics.txt
- comparison_report.txt
- optimized_model_fp16.pth

### Notes
FP16 quantization and pruning reduce model size and memory.

