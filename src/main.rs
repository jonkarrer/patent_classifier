mod batcher;
mod config;
mod data;
mod model;

fn main() {
    let device = config::get_device();
    let tokenizer = config::init_tokenizer();
    let training_set = data::DataSet::new("dataset/train.csv", &tokenizer);
    let validation_set = data::DataSet::new("dataset/validate.csv", &tokenizer);
}
