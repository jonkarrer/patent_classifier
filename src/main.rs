mod batcher;
mod config;
mod data;
mod model;

fn main() {
    let device = config::get_device();
    let training_set = data::DataSet::new("dataset/train.csv");
    let validation_set = data::DataSet::new("dataset/validate.csv");
}
