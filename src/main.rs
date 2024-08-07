mod data;

fn main() {
    let training_set = data::create_dataset("dataset/train.csv");
    let validation_set = data::create_dataset("dataset/validate.csv");

    println!("Hello, world!");
}
