mod batcher;
mod config;
mod data;
mod model;

fn main() {
    let device = config::get_device();
    let training_set = data::create_dataset("dataset/train.csv");
    let validation_set = data::create_dataset("dataset/validate.csv");

    let mut b = training_set.chunks(2);

    let batch = batcher::create_batch(b.next().unwrap(), device);
    dbg!(&batch.features.to_data());
    dbg!(&batch.labels.to_data());
    dbg!(&batch.padding_masks.to_data());
}
