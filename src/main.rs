mod batcher;
mod config;
mod data;
mod model;

const MODEL_SIZE: usize = 512;
const BATCH_SIZE: usize = 2;
const NUM_CLASSES: usize = 5;

fn main() {
    let device = config::get_device();
    let tokenizer = config::init_tokenizer();

    let training_set = data::DataSet::new("dataset/train.csv", &tokenizer);
    let validation_set = data::DataSet::new("dataset/validate.csv", &tokenizer);

    let mut batcher = training_set.data_points.chunks(BATCH_SIZE);
    let batch = batcher::create_batch(
        batcher.next().unwrap(),
        training_set.max_seq_len,
        training_set.vocab_size,
        MODEL_SIZE,
        &device,
    );

    let model = model::Model::new(MODEL_SIZE, &device);
    let output = model.forward(batch.embeddings, batch.mask);
    dbg!(&output.shape());

    // Get cls token
    let classification = output
        .slice([0..BATCH_SIZE, 0..1])
        .reshape([BATCH_SIZE, NUM_CLASSES]);

    // run loss
    let loss = model.loss(classification, batch.labels);
    dbg!(loss.to_data());

    // run optimizer
}
