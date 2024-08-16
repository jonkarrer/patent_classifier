use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLoss,
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{ElementConversion, Int, Tensor},
};
use config::MyBackend;
use model::Model;

mod batcher;
mod config;
mod data;
mod model;

const MODEL_SIZE: usize = 512;
const BATCH_SIZE: usize = 250;
const NUM_CLASSES: usize = 5;
const LEARNING_RATE: f64 = 1e-4;
const ARTIFACT_DIR: &str = "/tmp/text-classification-patent";
const NUM_EPOCHS: usize = 3;

fn main() {
    let device = config::get_device();
    let tokenizer = config::init_tokenizer();
    let mut model: Model<MyBackend> = Model::new(MODEL_SIZE, &device);
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(5e-5)))
        .init();

    // 0: load data
    let training_set = data::DataSet::new("dataset/train.csv", &tokenizer);
    let validation_set = data::DataSet::new("dataset/validate.csv", &tokenizer);

    let training_chunks = training_set.data_points.chunks(BATCH_SIZE);
    let validation_chunks = validation_set.data_points.chunks(BATCH_SIZE);

    for (iteration, data) in training_chunks.enumerate() {
        let batch = batcher::create_batch(
            data,
            training_set.max_seq_len,
            training_set.vocab_size,
            MODEL_SIZE,
            &device,
        );

        // 2:run prediction
        let output = model.forward(batch.embeddings, batch.mask);
        let classification = output
            .slice([0..BATCH_SIZE, 0..1])
            .reshape([BATCH_SIZE, NUM_CLASSES]);

        // 3: calc loss
        let loss_function = CrossEntropyLoss::new(None, &device);
        let loss = loss_function.forward(classification.clone(), batch.labels.clone());

        // 4: run optimizer
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(LEARNING_RATE, model, grads);

        // 5: calc accuracy
        let accuracy = accuracy(classification, batch.labels);

        println!(
            "[Train - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3} %",
            1,
            iteration,
            loss.clone().into_scalar(),
            accuracy,
        );
    }

    // Get the model without autodiff.
    let model_valid = model.valid();

    // Implement our validation loop.
    for (iteration, data) in validation_chunks.enumerate() {
        let batch = batcher::create_batch(
            data,
            validation_set.max_seq_len,
            validation_set.vocab_size,
            MODEL_SIZE,
            &device,
        );
        let output = model_valid.forward(batch.embeddings.clone(), batch.mask.clone());
        let classification = output
            .slice([0..BATCH_SIZE, 0..1])
            .reshape([BATCH_SIZE, NUM_CLASSES]);

        let loss_function = CrossEntropyLoss::new(None, &device);
        let loss = loss_function.forward(classification.clone(), batch.labels.clone());

        let accuracy = accuracy(classification, batch.labels);

        println!(
            "[Valid - Epoch {} - Iteration {}] Loss {} | Accuracy {}",
            1,
            iteration,
            loss.clone().into_scalar(),
            accuracy,
        );
    }
}

/// Create out own accuracy metric calculation.
fn accuracy<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
    let predictions = output.argmax(1).squeeze(1);
    let num_predictions: usize = targets.dims().iter().product();
    let num_corrects = predictions.equal(targets).int().sum().into_scalar();

    num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
}
