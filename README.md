# 🦀 Rust Neural Network

A simple, from-scratch implementation of a feedforward neural network in Rust. This implementation focuses on readability and educational value while maintaining good performance.

## Features

- Pure Rust implementation using only the standard library and `rand` crate
- Feedforward neural network with configurable architecture
- Multiple activation functions:
  - Sigmoid
  - ReLU (Rectified Linear Unit)
  - Leaky ReLU
  - Tanh (Hyperbolic Tangent)
- Advanced training features:
  - Momentum-based gradient descent
  - Dropout regularization
  - Xavier/Glorot weight initialization
- Example pattern recognition task included

## Usage

```bash
cargo run
```

The example code demonstrates pattern recognition on simple 9x9 pixel patterns (cross, circle, and square). The network architecture is:
- Input layer: 81 neurons (9x9 grid)
- First hidden layer: 32 neurons (ReLU activation)
- Second hidden layer: 16 neurons (ReLU activation)
- Output layer: 3 neurons (Sigmoid activation)

## Example Output

```
Training network to recognize patterns...
Epoch 0: mean squared error = 0.970517
...
Epoch 1900: mean squared error = 0.000052

Testing pattern recognition:
Pattern | Cross | Circle | Square
---------|--------|--------|--------
Cross   | 0.9957 | 0.0036 | 0.0032
Circle  | 0.0036 | 0.9954 | 0.0046
Square  | 0.0033 | 0.0041 | 0.9953
```

## Implementation Details

- Uses simple vector operations for good readability
- Implements backpropagation with momentum for faster convergence
- Features dropout regularization (dropout rate) to prevent overfitting
- Uses Xavier/Glorot initialization for better training stability

## Structure

- `main.rs`: Contains the entire implementation including:
  - Activation functions and their derivatives
  - Layer implementation with forward pass
  - Neural network implementation with training
  - Pattern recognition example

## Future Improvements

Potential areas for enhancement:
- Mini-batch gradient descent
- Additional optimization algorithms (Adam, RMSProp)
- Batch normalization
- Convolutional layers
- Loading/saving trained models