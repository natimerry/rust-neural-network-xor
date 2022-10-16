#![allow(clippy::needless_range_loop)]
use rand::rngs::mock::StepRng;
use rand::Rng;
use shuffle::{irs::Irs, shuffler::Shuffler};
use text_io::{self, read};
use crate::files::{save_network, check_exist};
use indicatif::ProgressBar;
mod files;

#[macro_use]
extern crate savefile_derive;
const NUM_INPUTS: usize = 2;
const NUM_HIDDEN_NODES: usize = 2;
const NUM_OUTPUTS: usize = 1;
const NUM_TRAINING_SETS: usize = 4;

#[derive(Savefile)]
pub struct NueralNetwork{
    epochs: i64,
    learning_rate: f64,
    output_layer: [f64; NUM_OUTPUTS],
    hidden_layer: [f64; NUM_HIDDEN_NODES],
    hidden_layer_bias: [f64; NUM_HIDDEN_NODES],
    output_layer_bias: [f64; NUM_OUTPUTS],
    hidden_weight: [[f64; NUM_INPUTS]; NUM_HIDDEN_NODES],
    output_weight: [[f64; NUM_HIDDEN_NODES]; NUM_OUTPUTS],
    training_input: [[f64; NUM_INPUTS]; NUM_TRAINING_SETS],
    training_output: [[f64; NUM_OUTPUTS]; NUM_TRAINING_SETS] ,
}

fn init_weights() -> f64 {
    rand::thread_rng().gen()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}
fn d_sigmoid(x: f64) -> f64 {
    x * (1.0 - x)
}


fn activate_network(network:&mut NueralNetwork,to_print:bool) {
    // training
    let mut training_set_order = vec![0, 1, 2, 3];
    let to_loop_in_total = network.epochs;
    let bar_t0=ProgressBar::new(to_loop_in_total.try_into().unwrap());
    let sty = indicatif::ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    ).unwrap().progress_chars("##-");
    bar_t0.set_style(sty);

    for _epoch in 0..network.epochs {
        let _shuffle = Irs::default().shuffle(&mut training_set_order, &mut StepRng::new(2, 13));
        for x in 0..NUM_TRAINING_SETS {       
            let i = training_set_order[x];
            if x % 4 == 0{
                bar_t0.inc(1)
            }
            //computer hidden layer activation
            for j in 0..NUM_HIDDEN_NODES {
                let mut activation_hidden = network.hidden_layer_bias[j];

                for k in 0..NUM_INPUTS {
                    activation_hidden += network.training_input[i as usize][k] * network.hidden_weight[k][j];
                }

                network.hidden_layer[j] = sigmoid(activation_hidden);
            }

            for j in 0..NUM_OUTPUTS {
                let mut activation_output = network.output_layer_bias[j];

                for k in 0..NUM_HIDDEN_NODES {
                    activation_output += network.hidden_layer[k] * network.output_weight[j][k];
                }
                network.output_layer[j] = sigmoid(activation_output);
            }
            // reinforce learning in case of overflows
        
            if to_print {
                println!(
                    "Input: {:?}\tGot Output: {:?}\tExpected output: {:?}",
                    (network.training_input[i as usize][0], network.training_input[i as usize][1]),
                    network.output_layer[0],
                    network.training_output[i as usize][0]
                );
            }
            //backward prop

            //compute change in output weights

            let mut delta_output: [f64; NUM_OUTPUTS] = [0.0];
            for j in 0..NUM_OUTPUTS {
                let error: f64 = network.training_output[i as usize][j] - network.output_layer[j];
                delta_output[j] = error * d_sigmoid(network.output_layer[j]);
            }

            let mut delta_hidden: [f64; NUM_HIDDEN_NODES] = [0.0, 0.0];
            for j in 0..NUM_HIDDEN_NODES {
                let mut error: f64 = 0.0;
                error += delta_output[0] * network.output_weight[0][j];
                delta_hidden[j] = error * d_sigmoid(network.hidden_layer[j]);
            }

            // Change apply
            for j in 0..NUM_OUTPUTS {
                network.output_layer_bias[j] += delta_output[j] * network.learning_rate;
                for k in 0..NUM_HIDDEN_NODES {
                    network.output_weight[0][k] += network.hidden_layer[k] * delta_output[j] * network.learning_rate;
                }
            }

            network.hidden_layer_bias[0] += delta_hidden[0] * network.learning_rate;
            for k in 0..NUM_INPUTS {
                network.hidden_weight[k][0] +=
                network.training_input[i as usize][k] * delta_hidden[0] * network.learning_rate;
            }

            network.hidden_layer_bias[1] += delta_hidden[1] * network.learning_rate;
            for k in 0..NUM_INPUTS {
                network.hidden_weight[k][1] +=
                network.training_input[i as usize][k] * delta_hidden[1] * network.learning_rate;
            }
        }
    }
    bar_t0.finish_and_clear();
}

fn train(){
    if files::check_exist("model.data"){
        let mut nn:NueralNetwork = files::load_network();
        print!("Enter epochs:");
        nn.epochs=read!();
        if nn.epochs >= 200_000_000{
            println!("Epochs over 2 Million causes floating point exceptions\n\n");
            nn.epochs=200_000_000;
        }
        activate_network(&mut nn, false);
        save_network(&nn);
    }
    else{
        print!("Enter epochs (no read): ");
        let mut e:i64=read!();
        if e >= 200_000_000{
            println!("Epochs over 2 Million causes floating point exceptions\n\n");
            e=200_000_000;
        }
        let mut nn:NueralNetwork = NueralNetwork{
            epochs: e,
            learning_rate: 0.5,
            output_layer: [0.0],
            hidden_layer: [0.0, 0.0],
            hidden_layer_bias:[0.0, 0.0] ,
            output_layer_bias:[init_weights()],
            hidden_weight:  [
                [init_weights(), init_weights()],
                [init_weights(), init_weights()],
            ],
            output_weight:[[0.0, 0.0]],
            training_input:[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            training_output:[[0.0], [1.0], [1.0], [0.0]],
        };
        
        activate_network(&mut nn, false);
        save_network(&nn);
        
    }
}

fn use_network(){
    let mut nn= files::load_network();
    nn.epochs=1;
    activate_network(&mut nn, true);
}
fn main() {
    loop {
        print!("\n\n1- Train\n2- Run (requires trained model)\nEnter choice: ");
        let choice:i32= read!();
        if choice == 1{
            train();
        }
        if choice==2{
            if check_exist("model.data"){
                use_network();
            }
            else {
                print!("Model doesnt exist. Train first");
            }
        }
    }
}
