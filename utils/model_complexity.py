# model_complexity.py
import torch
from thop import profile, clever_format

def analyze_model_complexity(model, input_shape=(1, 4, 256, 256), device='cuda'):

    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    data_low = torch.randn(input_shape).to(device)
    data_high = torch.randn(input_shape).to(device)
    
    flops, params = profile(model, inputs=(data_low, data_high), verbose=False)
    
    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
    
    results = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'flops_formatted': flops_formatted,
        'params_formatted': params_formatted,
        'model_size_mb': total_params * 4 / 1024 / 1024 
    }
    
    return results

def print_model_complexity(results, input_size):

    print("" + "="*50)
    print("Model Analyze:")
    print("="*50)
    print(f"total_parameters: {results['total_params']:,}")
    print(f"trainable_parameters: {results['trainable_params']:,}")
    print(f"Model Size: {results['model_size_mb']:.2f} MB")
    print(f"Input Shape: {input_size}")
    print(f"FLOPs: {results['flops_formatted']}")
    print("="*50)
