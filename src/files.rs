use savefile::prelude::*;


pub fn check_exist(path:&str) -> bool{
    std::path::Path::new(&path).exists()
}

pub fn load_network()-> crate::NueralNetwork{
    println!("Loading network...\n");
    load_file("model.data", 0).unwrap()
}
pub fn save_network(network:&crate::NueralNetwork)
{
    save_file("./model.data", 0, network).expect("Error saving the model");
}

