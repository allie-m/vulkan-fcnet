One-Layer Fully Connected Net with Vulkan
=

This was a mistake. It has eaten up so much of my time getting this working before the symposium. At least my Machine Learning teacher liked it lol.

This crate binary implements a one layer FC net from scratch in Rust, on the CPU and GPU.

```
cargo run --release [cpu/gpu]
```

The implementations are both very slow in debug mode -- run it in release mode for the most accurate benchmark.

On my home machine, I need to use vulkan-radeon instead of amdvlk because amdvlk does not support the correct features (shader_shared_float32_atomic_add) which your vulkan implementation also needs to support if you want to run this terrible program.

The below commands run the different implementations:

```
cargo run --release cpu

AMD_VULKAN_ICD=RADV cargo run --release gpu
```

(AMD_VULKAN_ICD=RADV is just for me unless you too want to use radv instead of amdvlk on your linux machine when running this)

TODO: local Pytorch CPU/GPU benchmarks

## Results

(Results are rounded and approximate collected on an R5 3600/RX 5700XT Linux machine)

Batch size: 256

CPU
- Initialization time:          100us
- Rng/batching time:            520us
- Score+loss+gradient+backprop: 250us
- Per-iteration time:           800us
- Total time (5000 iterations): 4.08s

GPU
- Initialization time:          45ms
- Rng/batching time:            520us
- Batch load time:              150us
- Submit time:                  10us
- Score+loss+gradient+backprop: 700us
- Finalize time:                9ms
- Per-iteration time:           830us
- Total time (5000 iterations): 4.21s

Batch size: 512

CPU
- Initialization time:          176us
- Rng/batching time:            630us
- Score+loss+gradient+backprop: 490us
- Per-iteration time:           1.1ms
- Total time (5000 iterations): 5.72s

GPU
- Initialization time:          44ms
- Rng/batching time:            630us
- Batch load time:              10us
- Submit time:                  160us
- Score+loss+gradient+backprop: 930us
- Finalize time:                9ms
- Per-iteration time:           1.1ms
- Total time (5000 iterations): 5.46s
