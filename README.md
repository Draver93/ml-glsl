# GPU-Accelerated Neural Network (NNGL)

A high-performance neural network implementation leveraging OpenGL compute shaders for GPU acceleration. This project explores the intersection of graphics programming and machine learning, demonstrating how modern GPU compute capabilities can dramatically accelerate neural network training and inference.

## Overview

This project implements a fully GPU-accelerated neural network using OpenGL compute shaders. Unlike traditional CPU-based implementations or CUDA-specific solutions, this approach uses OpenGL for cross-platform GPU computing, making it accessible across different graphics hardware vendors.

### Key Features

- **Pure GPU Implementation**: All neural network operations (forward pass, backpropagation, weight updates) run on GPU
- **OpenGL Compute Shaders**: Cross-platform GPU acceleration without vendor lock-in
- **Interactive Testing Interface**: Real-time CLI for testing and benchmarking
- **Memory Efficient**: Direct GPU buffer management with minimal CPU-GPU transfers
- **Configurable Architecture**: Support for multiple layer types and activation functions

## Architecture

The implementation consists of several key components:

### Core Classes

- **`NeuralNetwork`**: Main orchestrator handling training pipeline and GPU resource management
- **`Layer`**: Individual network layer with GPU buffers for weights, biases, activations, and gradients
- **`Shader`**: OpenGL compute shader wrapper for GPU kernel execution

### GPU Kernels (Compute Shaders)

- `forward_pass.comp` - Matrix multiplication and activation functions
- `delta_loss.comp` - Output layer error calculation
- `backward_pass.comp` - Hidden layer gradient computation
- `update_weights.comp` - Weight parameter updates
- `update_biases.comp` - Bias parameter updates

## Training Pipeline

The GPU-accelerated training follows the standard backpropagation algorithm:

1. **Forward Pass**: Input propagation through all layers with activation
2. **Loss Calculation**: Error computation at output layer
3. **Backward Pass**: Gradient propagation through hidden layers
4. **Parameter Update**: Weight and bias updates using computed gradients

All operations are parallelized across GPU compute units, with careful memory barrier synchronization between stages.

## Example Use Case

The current implementation is trained to approximate the function `f(a,b) = sin(a) * sin(b)`, demonstrating the network's ability to learn complex non-linear relationships.

## Interactive Testing

The built-in CLI provides several testing modes:

```bash
nn> test 1.57 3.14          # Test specific inputs
nn> random                  # Test random values
nn> batch 100              # Batch test 100 samples
nn> benchmark              # Performance benchmark
nn> layer 0                # Inspect layer details
```

## Performance Characteristics

GPU acceleration provides significant performance improvements:

- **Parallel Processing**: Thousands of operations executed simultaneously
- **High Throughput**: Capable of processing thousands of inferences per second
- **Memory Bandwidth**: Efficient utilization of GPU memory hierarchy
- **Scalable**: Performance scales with GPU compute capability

## Technical Implementation Details

### Memory Management

The implementation uses OpenGL Shader Storage Buffer Objects (SSBOs) for efficient GPU memory management:

- Input/target data buffers for training batches
- Per-layer weight and bias parameter storage
- Activation and gradient buffers for forward/backward passes
- Automatic buffer sizing and overflow protection

### Compute Workgroup Optimization

Workgroup dispatch is carefully calculated to maximize GPU utilization:

```cpp
int workgroups_x = std::min((int)ceil(batch_size * input_size / 16.0f), 65535);
int workgroups_y = std::min((int)ceil(batch_size * output_size / 16.0f), 65535);
glDispatchCompute(workgroups_x, workgroups_y, 1);
```

### Synchronization

Memory barriers ensure proper data dependencies between compute shader stages:

```cpp
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
```

## Requirements

### Hardware
- GPU with OpenGL 4.3+ compute shader support
- Minimum 1GB GPU memory recommended

### Software Dependencies
- OpenGL 4.3+
- Graphics drivers with compute shader support
- C++17 compatible compiler

### Build Instructions

```bash
# Clone repository
git clone <repository-url>
cd nn-glsl-core

# Generate project files
premake5 vs2019    # For Visual Studio 2019
# or
premake5 vs2022    # For Visual Studio 2022
# or
premake5 gmake2    # For Unix Makefiles

# Windows (Visual Studio)
# Open generated .sln file and build, or use MSBuild:
msbuild nn-glsl-core.sln /p:Configuration=Release

# Linux (Make)
make config=release

# Run
./bin/Release-<system>-x86_64/nn-glsl-core/nn-glsl-core
```

## Learning Objectives

This project explores several key concepts:

### GPU Computing Fundamentals
- Compute shader programming and optimization
- GPU memory hierarchy and access patterns
- Parallel algorithm design considerations

### Neural Network Implementation
- Backpropagation algorithm implementation
- Gradient computation and parameter updates
- Numerical stability in GPU floating-point operations

### Performance Engineering
- GPU workload distribution and occupancy
- Memory bandwidth optimization
- CPU-GPU synchronization strategies

## Future Enhancements

Potential areas for expansion:

- **Advanced Optimizers**: Adam, RMSprop implementations
- **Regularization**: Dropout, batch normalization
- **Layer Types**: Convolutional, LSTM layers
- **Multi-GPU**: Distributed training across multiple GPUs
- **Mixed Precision**: FP16 training for improved performance
- **Dynamic Graphs**: Runtime network architecture modification


## Research Applications

This implementation serves as a foundation for exploring:

- GPU compute optimization techniques
- Parallel numerical algorithms
- Cross-platform high-performance computing
- Alternative ML acceleration approaches

## Contributing

This is a learning project focused on understanding GPU-accelerated neural networks. Contributions, optimizations, and educational improvements are welcome.

## License

This project is open-source and free to use for educational purposes.

## Acknowledgments

This project demonstrates practical applications of GPU computing in machine learning, bridging graphics programming and neural network implementation.