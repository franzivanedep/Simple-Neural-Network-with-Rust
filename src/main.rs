fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

fn main() {
    // Dataset: (x1, x2, expected_output)
    let dataset = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ];

    // Initialize weights and biases (mutable for training)
    let mut w1_h1 = 0.5;
    let mut w2_h1 = 0.5;
    let mut b_h1 = -0.7;

    let mut w1_h2 = 0.5;
    let mut w2_h2 = 0.5;
    let mut b_h2 = -0.7;

    let mut w1_o = 1.0;
    let mut w2_o = 1.0;
    let mut b_o = -1.0;

    let learning_rate = 0.5;
    let epochs = 10000;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for &(x1, x2, expected) in &dataset {
            // --- Forward pass ---
            let z_h1 = (x1 * w1_h1) + (x2 * w2_h1) + b_h1;
            let out_h1 = sigmoid(z_h1);

            let z_h2 = (x1 * w1_h2) + (x2 * w2_h2) + b_h2;
            let out_h2 = sigmoid(z_h2);
            

            let z_o = (out_h1 * w1_o) + (out_h2 * w2_o) + b_o;
            let output = sigmoid(z_o);

            // Calculate loss (MSE)
            let loss = 0.5 * (expected - output).powi(2);
            total_loss += loss;

            // --- Backpropagation ---

            // Output neuron error term
            let d_output = (output - expected) * sigmoid_derivative(output);

            // Hidden neurons error terms
            let d_h1 = d_output * w1_o * sigmoid_derivative(out_h1);
            let d_h2 = d_output * w2_o * sigmoid_derivative(out_h2);

            // --- Update output layer weights and bias ---
            w1_o -= learning_rate * d_output * out_h1;
            w2_o -= learning_rate * d_output * out_h2;
            b_o -= learning_rate * d_output;

            // --- Update hidden layer weights and biases ---
            w1_h1 -= learning_rate * d_h1 * x1;
            w2_h1 -= learning_rate * d_h1 * x2;
            b_h1 -= learning_rate * d_h1;

            w1_h2 -= learning_rate * d_h2 * x1;
            w2_h2 -= learning_rate * d_h2 * x2;
            b_h2 -= learning_rate * d_h2;
        }

        // Print loss every 1000 epochs
        if epoch % 1000 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, total_loss / dataset.len() as f64);
        }
    }

    // Final predictions after training
    println!("\nFinal predictions:");
    for &(x1, x2, expected) in &dataset {
        let out_h1 = sigmoid((x1 * w1_h1) + (x2 * w2_h1) + b_h1);
        let out_h2 = sigmoid((x1 * w1_h2) + (x2 * w2_h2) + b_h2);
        let output = sigmoid((out_h1 * w1_o) + (out_h2 * w2_o) + b_o);

        println!(
            "Input: ({}, {}), Expected: {:.1}, Output: {:.4}",
            x1, x2, expected, output
        );
    }
}
