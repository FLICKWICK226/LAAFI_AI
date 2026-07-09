import argparse
import io
import time
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Using the existing inference entrypoint for model loading
from inference import load_model


def get_dummy_image_bytes(size=(224, 224)):
    """Generate a dummy image to use for profiling."""
    img = Image.new("RGB", size, color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def profile_inference(model_path: str, num_runs: int = 50, device_str: str = "auto"):
    # Determine device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    print(f"Loading model on {device}...")
    model = load_model(model_path, device)
    
    # Transformation standard pour ResNet (from inference.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dummy_bytes = get_dummy_image_bytes()
    
    metrics = {
        "preprocess_time_ms": [],
        "model_forward_time_ms": [],
        "postprocess_time_ms": [],
        "total_latency_ms": [],
        "cpu_time_ms": [],
        "gpu_time_ms": [],
        "memory_allocated_mb": []
    }
    
    print(f"Starting profiling with {num_runs} warmup runs and {num_runs} active runs...")
    
    # Warmup
    for _ in range(num_runs):
        image = Image.open(io.BytesIO(dummy_bytes)).convert("RGB")
        tensor_img = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor_img)
            prob = torch.sigmoid(logits).item()
            
    # Active profiling
    for _ in range(num_runs):
        # Memory before
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
            
        t0 = time.perf_counter()
        
        # 1. Preprocess
        t_pre_start = time.perf_counter()
        image = Image.open(io.BytesIO(dummy_bytes)).convert("RGB")
        tensor_img = transform(image).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_pre_end = time.perf_counter()
        
        # 2. Model Forward
        t_fwd_start = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor_img)
            prob = torch.sigmoid(logits).item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        
        # 3. Postprocess
        t_post_start = time.perf_counter()
        _prediction = "Métastase" if prob >= 0.5 else "Sain"
        _result = {
            "diagnostic": _prediction,
            "probabilite": round(prob, 4),
            "risque_pourcentage": f"{prob * 100:.2f}%"
        }
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_post_end = time.perf_counter()
        
        t1 = time.perf_counter()
        
        # Memory after
        if device.type == "cuda":
            mem_peak = torch.cuda.max_memory_allocated()
            mem_used_mb = (mem_peak - mem_before) / (1024 * 1024)
        else:
            # Basic estimate for CPU if psutil is not used
            mem_used_mb = 0.0

        # Calculate times in ms
        pre_ms = (t_pre_end - t_pre_start) * 1000
        fwd_ms = (t_fwd_end - t_fwd_start) * 1000
        post_ms = (t_post_end - t_post_start) * 1000
        total_ms = (t1 - t0) * 1000
        
        # GPU / CPU time distinction (simplified)
        if device.type == "cuda":
            gpu_time_ms = fwd_ms + pre_ms # Tensor transfer happens in pre
            cpu_time_ms = post_ms
        else:
            gpu_time_ms = 0.0
            cpu_time_ms = total_ms

        metrics["preprocess_time_ms"].append(pre_ms)
        metrics["model_forward_time_ms"].append(fwd_ms)
        metrics["postprocess_time_ms"].append(post_ms)
        metrics["total_latency_ms"].append(total_ms)
        metrics["cpu_time_ms"].append(cpu_time_ms)
        metrics["gpu_time_ms"].append(gpu_time_ms)
        metrics["memory_allocated_mb"].append(mem_used_mb)

    # Compute averages
    avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
    
    print("\n--- Profiling Results (Averages over {} runs) ---".format(num_runs))
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save to CSV
    # Go up one level from src to project root, then metrics/
    project_root = Path(__file__).resolve().parent.parent
    metrics_dir = project_root / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)
    csv_path = metrics_dir / "profiling_baseline.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_metrics.keys()))
        writer.writeheader()
        writer.writerow(avg_metrics)
        
    print(f"\nMetrics saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../resnet50_finetuned_weights.pth", help="Path to weights")
    parser.add_argument("--runs", type=int, default=50, help="Number of inference runs")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    args = parser.parse_args()
    
    profile_inference(args.model_path, args.runs, args.device)
