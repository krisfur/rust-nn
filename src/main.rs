use rand::Rng;
use std::f64::consts::E;

#[derive(Clone, Copy)]
#[allow(dead_code)]  // Allow unused variants as they might be used in future
enum Activation {
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
}

impl Activation {
    fn forward(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + E.powf(-x)),
            Activation::ReLU => if x > 0.0 { x } else { 0.0 },
            Activation::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            Activation::Tanh => x.tanh(),
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => {
                let sx = self.forward(x);
                sx * (1.0 - sx)
            },
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::LeakyReLU => if x > 0.0 { 1.0 } else { 0.01 },
            Activation::Tanh => {
                let tx = x.tanh();
                1.0 - tx * tx
            },
        }
    }
}

// Xavier/Glorot initialization scaling factor
fn weight_scale(n_inputs: usize, activation: Activation) -> f64 {
    match activation {
        Activation::Sigmoid | Activation::Tanh => (2.0 / n_inputs as f64).sqrt(),
        Activation::ReLU | Activation::LeakyReLU => (2.0 / n_inputs as f64).sqrt() * 2.0,
    }
}

// Simple Vec<f64> based neural network - no matrix operations needed
struct Layer {
    num_inputs: usize,
    num_neurons: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activation: Activation,
    dropout_rate: f64,
}

impl Layer {
    fn new(num_inputs: usize, num_neurons: usize, activation: Activation, dropout_rate: f64) -> Self {
        assert!(dropout_rate >= 0.0 && dropout_rate < 1.0, "Dropout rate must be between 0 and 1");
        
        let mut rng = rand::rng();
        let scale = weight_scale(num_inputs, activation);
        
        // Initialize weights and biases with Xavier/Glorot initialization
        let weights = (0..num_neurons)
            .map(|_| (0..num_inputs)
                .map(|_| rng.random_range(-scale..scale))
                .collect())
            .collect();
            
        let biases = (0..num_neurons)
            .map(|_| rng.random_range(-0.1..0.1))
            .collect();

        Layer {
            num_inputs,
            num_neurons,
            weights,
            biases,
            activation,
            dropout_rate,
        }
    }

    fn forward(&self, inputs: &[f64], is_training: bool) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
        let mut rng = rand::rng();
        let mut activations = Vec::with_capacity(self.num_neurons);
        let mut raw_outputs = Vec::with_capacity(self.num_neurons);
        let mut dropout_mask = vec![true; self.num_neurons];
        
        // During training, randomly disable neurons
        if is_training && self.dropout_rate > 0.0 {
            for mask in dropout_mask.iter_mut() {
                if rng.random_range(0.0..1.0) < self.dropout_rate {
                    *mask = false;
                }
            }
        }
        
        let scale = if is_training { 1.0 / (1.0 - self.dropout_rate) } else { 1.0 };
        
        for n in 0..self.num_neurons {
            let sum: f64 = (0..self.num_inputs)
                .map(|i| inputs[i] * self.weights[n][i])
                .sum();
            
            let raw = sum + self.biases[n];
            let activated = self.activation.forward(raw);
            
            raw_outputs.push(raw);
            activations.push(if dropout_mask[n] { activated * scale } else { 0.0 });
        }
        
        (activations, raw_outputs, dropout_mask)
    }
}

struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    fn new(architecture: &[usize], hidden_activation: Activation, output_activation: Activation, dropout_rate: f64) -> Self {
        let mut layers = Vec::new();
        
        // Create layers based on architecture
        for i in 0..architecture.len() - 2 {
            layers.push(Layer::new(
                architecture[i], 
                architecture[i + 1], 
                hidden_activation,
                dropout_rate  // Apply dropout to hidden layers
            ));
        }
        
        // Output layer (no dropout in output layer)
        layers.push(Layer::new(
            architecture[architecture.len() - 2],
            architecture[architecture.len() - 1],
            output_activation,
            0.0  // No dropout in output layer
        ));
        
        NeuralNetwork { layers }
    }
    
    fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut current = inputs.to_vec();
        
        // Forward propagate through each layer (no dropout during inference)
        for layer in &self.layers {
            let (activations, _, _) = layer.forward(&current, false);
            current = activations;
        }
        
        current
    }

    fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], epochs: usize, learning_rate: f64) {
        let momentum = 0.9;
        
        // Initialize momentum arrays
        let mut prev_weight_changes: Vec<Vec<Vec<f64>>> = self.layers.iter()
            .map(|l| vec![vec![0.0; l.num_inputs]; l.num_neurons])
            .collect();
            
        let mut prev_bias_changes: Vec<Vec<f64>> = self.layers.iter()
            .map(|l| vec![0.0; l.num_neurons])
            .collect();

        for epoch in 0..epochs {
            let mut total_error = 0.0;
            
            // Train on each input-target pair
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let mut layer_outputs = vec![input.clone()];
                let mut layer_raw_outputs = Vec::new();
                let mut dropout_masks = Vec::new();
                let mut current = input.clone();
                
                // Forward pass with dropout
                for layer in &self.layers {
                    let (activations, raw, mask) = layer.forward(&current, true);
                    current = activations.clone();
                    layer_outputs.push(current.clone());
                    layer_raw_outputs.push(raw);
                    dropout_masks.push(mask);
                }
                
                // Calculate error
                let output_error: Vec<f64> = layer_outputs.last().unwrap()
                    .iter()
                    .zip(target.iter())
                    .map(|(output, target)| output - target)
                    .collect();
                
                total_error += output_error.iter().map(|e| e * e).sum::<f64>();
                
                // Backward pass with momentum
                let mut deltas = output_error.iter()
                    .zip(layer_raw_outputs.last().unwrap())
                    .map(|(err, raw)| err * self.layers.last().unwrap().activation.derivative(*raw))
                    .collect::<Vec<f64>>();
                
                for layer_idx in (0..self.layers.len()).rev() {
                    let layer = &mut self.layers[layer_idx];
                    let inputs = &layer_outputs[layer_idx];
                    
                    // Update weights and biases with momentum
                    for n in 0..layer.num_neurons {
                        for i in 0..layer.num_inputs {
                            let weight_change = learning_rate * deltas[n] * inputs[i] 
                                + momentum * prev_weight_changes[layer_idx][n][i];
                            layer.weights[n][i] -= weight_change;
                            prev_weight_changes[layer_idx][n][i] = weight_change;
                        }
                        let bias_change = learning_rate * deltas[n] 
                            + momentum * prev_bias_changes[layer_idx][n];
                        layer.biases[n] -= bias_change;
                        prev_bias_changes[layer_idx][n] = bias_change;
                    }
                    
                    // Calculate deltas for next layer
                    if layer_idx > 0 {
                        let mut new_deltas = vec![0.0; layer.num_inputs];
                        let num_inputs = layer.num_inputs;
                        let num_neurons = layer.num_neurons;
                        
                        // Store what we need from the current layer
                        let layer_weights = layer.weights.clone();
                        
                        // Get what we need from the previous layer
                        let (layers_head, _) = self.layers.split_at_mut(layer_idx);
                        let prev_layer = &layers_head[layer_idx - 1];
                        let activation_prev = prev_layer.activation;
                        let prev_raw = &layer_raw_outputs[layer_idx - 1];
                        
                        for i in 0..num_inputs {
                            for n in 0..num_neurons {
                                new_deltas[i] += deltas[n] * layer_weights[n][i];
                            }
                            new_deltas[i] *= activation_prev.derivative(prev_raw[i]);
                        }
                        deltas = new_deltas;
                    }
                }
            }
            
            if epoch % 100 == 0 {
                println!("Epoch {}: mean squared error = {:.6}", 
                    epoch, 
                    total_error / inputs.len() as f64
                );
            }
        }
    }
}

// Pattern recognition example - recognize simple shapes encoded as 1s and 0s
fn create_pattern_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Define patterns (9x9 grids flattened to 81 inputs)
    // 1 = filled, 0 = empty
    
    // Pattern 1: Cross
    let cross: Vec<f64> = vec![
        0.,0.,0.,1.,1.,1.,0.,0.,0.,
        0.,0.,0.,1.,1.,1.,0.,0.,0.,
        0.,0.,0.,1.,1.,1.,0.,0.,0.,
        1.,1.,1.,1.,1.,1.,1.,1.,1.,
        1.,1.,1.,1.,1.,1.,1.,1.,1.,
        1.,1.,1.,1.,1.,1.,1.,1.,1.,
        0.,0.,0.,1.,1.,1.,0.,0.,0.,
        0.,0.,0.,1.,1.,1.,0.,0.,0.,
        0.,0.,0.,1.,1.,1.,0.,0.,0.,
    ];

    // Pattern 2: Circle
    let circle: Vec<f64> = vec![
        0.,0.,1.,1.,1.,1.,1.,0.,0.,
        0.,1.,1.,1.,1.,1.,1.,1.,0.,
        1.,1.,1.,0.,0.,0.,1.,1.,1.,
        1.,1.,0.,0.,0.,0.,0.,1.,1.,
        1.,1.,0.,0.,0.,0.,0.,1.,1.,
        1.,1.,0.,0.,0.,0.,0.,1.,1.,
        1.,1.,1.,0.,0.,0.,1.,1.,1.,
        0.,1.,1.,1.,1.,1.,1.,1.,0.,
        0.,0.,1.,1.,1.,1.,1.,0.,0.,
    ];

    // Pattern 3: Square
    let square: Vec<f64> = vec![
        1.,1.,1.,1.,1.,1.,1.,1.,1.,
        1.,1.,1.,1.,1.,1.,1.,1.,1.,
        1.,1.,0.,0.,0.,0.,0.,1.,1.,
        1.,1.,0.,0.,0.,0.,0.,1.,1.,
        1.,1.,0.,0.,0.,0.,0.,1.,1.,
        1.,1.,0.,0.,0.,0.,0.,1.,1.,
        1.,1.,0.,0.,0.,0.,0.,1.,1.,
        1.,1.,1.,1.,1.,1.,1.,1.,1.,
        1.,1.,1.,1.,1.,1.,1.,1.,1.,
    ];

    let inputs = vec![cross.clone(), circle.clone(), square.clone()];
    
    // Outputs: [1,0,0] = cross, [0,1,0] = circle, [0,0,1] = square
    let targets = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    (inputs, targets)
}

fn main() {
    // Create a neural network for pattern recognition:
    // 81 inputs (9x9 grid) -> 32 hidden (ReLU) -> 16 hidden (ReLU) -> 3 outputs (Sigmoid)
    let mut network = NeuralNetwork::new(
        &[81, 32, 16, 3],
        Activation::ReLU,    // hidden layers
        Activation::Sigmoid, // output layer
        0.2,                // 20% dropout rate for hidden layers
    );
    
    let (training_inputs, training_targets) = create_pattern_data();
    
    // Train the network
    println!("Training network to recognize patterns...");
    network.train(&training_inputs, &training_targets, 2000, 0.01);
    
    // Test the network
    println!("\nTesting pattern recognition:");
    println!("Pattern | Cross | Circle | Square");
    println!("---------|--------|--------|--------");
    for (i, input) in training_inputs.iter().enumerate() {
        let output = network.forward(input);
        let pattern_name = match i {
            0 => "Cross ",
            1 => "Circle",
            2 => "Square",
            _ => "Unknown",
        };
        println!("{}  | {:.4} | {:.4} | {:.4}", 
            pattern_name, output[0], output[1], output[2]);
    }
}