"""Measures training-step time and peak memory of model configurations, on random data, on the available device.

    uv run scripts/benchmark_models.py
    uv run scripts/benchmark_models.py --batch_size 16 --variables 29

Compares the ViT with the mesh model at several mesh sizes and attention implementations. Use it to choose
`dense_attention_max_pairs` for your hardware: `auto` should match the faster of `dense` and `sparse`.
"""

import argparse
import gc
import time

import torch

from nanoweather.models.auto import model_class_for
from nanoweather.models.configuration_nanoweather import NanoWeatherConfig
from nanoweather.models.configuration_nanoweather_mesh import NanoWeatherMeshConfig


def benchmark(config, batch_size: int, device: torch.device, num_steps: int, bf16: bool) -> tuple[float, float]:
    """Mean time of a forward and backward pass in milliseconds, and peak memory in MB (CUDA only)."""
    model = model_class_for(config.model_type)(config).to(device)
    torch.nn.init.normal_(model.head.weight, std=0.02)
    states = torch.randn(batch_size, config.num_variables, *config.grid_size, device=device)
    static = torch.randn(config.num_static_channels, *config.grid_size, device=device)

    def step():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            loss = model(states, static=static, labels=states).loss
        loss.backward()

    step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / num_steps
    memory = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    return seconds * 1000, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--variables", type=int, default=29, help="Number of prognostic variables.")
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--refinements", type=int, nargs="+", default=[3, 4])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda"
    variables = [f"variable_{index}" for index in range(args.variables)]
    print(f"device {device}, bf16 {bf16}, batch size {args.batch_size}, {args.variables} variables")

    configs = [("vit", NanoWeatherConfig(variables=variables))]
    for refinement in args.refinements:
        for implementation in ("sparse", "dense", "auto"):
            config = NanoWeatherMeshConfig(
                variables=variables, mesh_refinement=refinement, attn_implementation=implementation
            )
            configs.append((f"mesh r={refinement} ({config.num_mesh_nodes} nodes) {implementation}", config))

    for name, config in configs:
        try:
            milliseconds, memory = benchmark(config, args.batch_size, device, args.num_steps, bf16)
            gc.collect()
            print(f"{name:36s} {milliseconds:8.1f} ms/step   peak memory {memory:8.0f} MB")
        except torch.OutOfMemoryError:
            print(f"{name:36s} out of memory")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
