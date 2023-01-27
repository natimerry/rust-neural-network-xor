use indicatif::ProgressBar;
use rand::{rngs::mock::StepRng, Rng};
use shuffle::{irs::Irs, shuffler::Shuffler};
use std::fs::File;
use std::{io::Write, usize};
use text_io::{self, read};

extern crate savefile_derive;
const NUM_INPUTS: usize = 2;
const NUM_HIDDEN_NODES: usize = 2;
const NUM_OUTPUTS: usize = 1;
const NUM_TRAINING_SETS: usize = 4;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}
fn d_sigmoid(x: f64) -> f64 {
    x * (1.0 - x)
}

#[derive(savefile_derive::Savefile)]
pub struct Network {
    epochs: i64,
    learning_rate: f64,
    output_layer: [f64; NUM_OUTPUTS],
    hidden_layer: [f64; NUM_HIDDEN_NODES],
    hidden_layer_bias: [f64; NUM_HIDDEN_NODES],
    output_layer_bias: [f64; NUM_OUTPUTS],
    hidden_weight: [[f64; NUM_INPUTS]; NUM_HIDDEN_NODES],
    output_weight: [[f64; NUM_HIDDEN_NODES]; NUM_OUTPUTS],
    training_input: [[f64; NUM_INPUTS]; NUM_TRAINING_SETS],
    training_output: [[f64; NUM_OUTPUTS]; NUM_TRAINING_SETS],
}
fn default() -> Network {
    let net = Network {
        epochs: 0,
        learning_rate: 0.5,
        output_layer: [0.0],
        hidden_layer: [0.0, 0.0],
        hidden_layer_bias: [0.0, 0.0],
        output_layer_bias: [rand::thread_rng().gen()],
        hidden_weight: [
            [rand::thread_rng().gen(), rand::thread_rng().gen()],
            [rand::thread_rng().gen(), rand::thread_rng().gen()],
        ],
        output_weight: [[0.0, 0.0]],
        training_input: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        training_output: [[0.0], [1.0], [1.0], [0.0]],
    };
    net
}
fn clear(){
    print!("\x1B[2J\x1B[1;1H");
}
impl std::fmt::Debug for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Network")
            .field("Epochs", &self.epochs)
            .field("Learning Rate", &self.learning_rate)
            .field("Output Layer", &self.output_layer)
            .field("Hidden Layer", &self.hidden_layer)
            .field("Hidden Layer Bias", &self.hidden_layer_bias)
            .field("Output Layer Bias", &self.output_layer_bias)
            .field("Hidden Weight", &self.hidden_weight)
            .field("Output Weight", &self.output_weight)
            .field("Training Input", &self.training_input)
            .field("Training Output", &self.training_output)
            .finish()
    }
}
impl Network {
    pub fn train(&mut self) {
        let mut training_set_order = vec![0, 1, 2, 3];
        // create progress bar
        let bar = ProgressBar::new(self.epochs.try_into().unwrap());
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("##-"),
        );

        for _epoch in 0..self.epochs {
            let _shuffle =
                Irs::default().shuffle(&mut training_set_order, &mut StepRng::new(2, 13));
            for x in 0..NUM_TRAINING_SETS {
                let i = training_set_order[x];
                if x % NUM_TRAINING_SETS == 0 {
                    bar.inc(1);
                }

                // Activate the hidden layer
                // initial step-through the hidden layer
                // steps through input layer to get the weighted sums of the INPUT*HIDDEN_WEIGHT and assign it to hidden node
                // pass weighted sum through the sigmoid (activation) function

                // TeX: σ(xᵢ⋅ wᵢ + b)
                for j in 0..NUM_HIDDEN_NODES {
                    let mut activation_hidden = self.hidden_layer_bias[j]; // assign current hidden layer bias value
                                                                           // to var

                    for k in 0..NUM_INPUTS {
                        activation_hidden += self.training_input[i][k] * self.hidden_weight[k][j];
                    }
                    self.hidden_layer[j] = sigmoid(activation_hidden);
                }
                // same as above, but we now step through the hidden layer and output nodes instead
                for j in 0..NUM_OUTPUTS {
                    let mut activation_output = self.output_layer_bias[j];

                    for k in 0..NUM_HIDDEN_NODES {
                        activation_output += self.hidden_layer[k] * self.output_weight[j][k];
                    }
                    self.output_layer[j] = sigmoid(activation_output);
                }

                // backward propagate to reinforce learning :)
                let mut delta_output: [f64; NUM_OUTPUTS] = [0.0];
                for j in 0..NUM_OUTPUTS {
                    let error: f64 = self.training_output[i as usize][j] - self.output_layer[j];
                    delta_output[j] = error * d_sigmoid(self.output_layer[j]);
                }

                let mut delta_hidden: [f64; NUM_HIDDEN_NODES] = [0.0, 0.0];
                for j in 0..NUM_HIDDEN_NODES {
                    let mut error: f64 = 0.0;
                    error += delta_output[0] * self.output_weight[0][j];
                    delta_hidden[j] = error * d_sigmoid(self.hidden_layer[j]);
                }

                // After finding out the delta error rate we can use that to apply changes on our network

                for j in 0..NUM_OUTPUTS {
                    self.output_layer_bias[j] += delta_output[j] * self.learning_rate;
                    for k in 0..NUM_HIDDEN_NODES {
                        self.output_weight[0][k] +=
                            self.hidden_layer[k] * delta_output[j] * self.learning_rate;
                    }
                }

                for j in 0..NUM_HIDDEN_NODES {
                    self.hidden_layer_bias[j] += delta_hidden[j] * self.learning_rate;
                    for k in 0..NUM_INPUTS {
                        self.hidden_weight[k][j] +=
                            self.training_input[i][k] * delta_hidden[j] * self.learning_rate;
                    }
                }
            }
        }
        bar.finish_with_message("Trained successfully");
    }

    pub fn run(&mut self) {
        for i in 0..NUM_TRAINING_SETS {
            for j in 0..NUM_HIDDEN_NODES {
                let mut activation_hidden = self.hidden_layer_bias[j];

                for k in 0..NUM_INPUTS {
                    activation_hidden +=
                        self.training_input[i as usize][k] * self.hidden_weight[k][j];
                }

                self.hidden_layer[j] = sigmoid(activation_hidden);
            }

            for j in 0..NUM_OUTPUTS {
                let mut activation_output = self.output_layer_bias[j];

                for k in 0..NUM_HIDDEN_NODES {
                    activation_output += self.hidden_layer[k] * self.output_weight[j][k];
                }
                self.output_layer[j] = sigmoid(activation_output);
            }
            println!(
                "Input: {:?}\tGot Output: {:?}\tExpected output: {:?}",
                (
                    self.training_input[i as usize][0],
                    self.training_input[i as usize][1]
                ),
                self.output_layer[0],
                self.training_output[i as usize][0]
            );
        }
    }
}

fn main() {
    let mut nn = default();
    loop {
        print!("1 --- Train the network\n2 --- Run the network\n3 --- Set custom training data\n");

        print!("\nEnter choice:");
        let choice: i64 = text_io::read!();

        if choice == 1 {
            print!("Enter number of epochs: ");
            let epochs: i64 = text_io::read!();
            nn.epochs = epochs;
            nn.train();
        }
        if choice == 2 {
            clear();
            nn.run();
        }
        if choice == 3 {
            print!("Enter the first number: ");
            let num1: i64 = text_io::read!();
            print!("\nEnter the second number: ");
            let num2: i64 = text_io::read!();
            nn.training_input[0][0] = num1 as f64;
            nn.training_input[0][1] = num2 as f64;
            nn.training_output[0][0] = sigmoid((num1 ^ num2) as f64);
            println!("\nNow run the network to try the new dataset!!!");
        }
    }
}
