use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{ElementConversion, Int, Tensor},
};
use config::MyBackend;
use model::Model;

mod batcher;
mod config;
mod data;
mod model;

const MODEL_SIZE: usize = 512;
const BATCH_SIZE: usize = 100;
const NUM_CLASSES: usize = 5;
const LEARNING_RATE: f64 = 1e-3;
const ARTIFACT_DIR: &str = "/tmp/text-classification-patent";
const NUM_EPOCHS: usize = 3;

fn main() {
    let device = config::get_device();
    let tokenizer = config::init_tokenizer();
    let mut model = model::Model::new(MODEL_SIZE, &device);
    let loss_function = CrossEntropyLossConfig::new().init(&device);
    let mut optimizer = AdamWConfig::new().init::<MyBackend, Model>();

    // 0: load data
    let training_set = data::DataSet::new("dataset/train.csv", &tokenizer);
    let validation_set = data::DataSet::new("dataset/validate.csv", &tokenizer);

    // let mut validation_chunks = validation_set.data_points.chunks(BATCH_SIZE);
    // let validation_batch = batcher::create_batch(
    //     validation_chunks.next().unwrap(),
    //     validation_set.max_seq_len,
    //     validation_set.vocab_size,
    //     MODEL_SIZE,
    //     &device,
    // );

    let training_chunks = training_set.data_points.chunks(BATCH_SIZE);

    for (iteration, training_batch) in training_chunks.enumerate() {
        let training_batch = batcher::create_batch(
            training_batch,
            training_set.max_seq_len,
            training_set.vocab_size,
            MODEL_SIZE,
            &device,
        );

        // 2:run prediction
        let output = model.forward(training_batch.embeddings, training_batch.mask);
        let classification = output
            .slice([0..BATCH_SIZE, 0..1])
            .reshape([BATCH_SIZE, NUM_CLASSES]);

        // 3: calc loss
        let loss = loss_function.forward(classification.clone(), training_batch.labels.clone());

        // 4: run optimizer
        let grads = loss.backward();
        let grads = GradientsParams::from_grads::<MyBackend, Model>(grads, &model);
        model = optimizer.step(LEARNING_RATE, model, grads);

        // 5: calc accuracy
        let accuracy = accuracy(classification, training_batch.labels);

        println!(
            "[Train - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3} %",
            1,
            iteration,
            loss.clone().into_scalar(),
            accuracy,
        );
    }
}

/// Create out own accuracy metric calculation.
fn accuracy(output: Tensor<MyBackend, 2>, targets: Tensor<MyBackend, 1, Int>) -> f32 {
    let predictions = output.argmax(1).squeeze(1);
    let num_predictions: usize = targets.dims().iter().product();
    let num_corrects = predictions.equal(targets).int().sum().into_scalar();

    num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
}
