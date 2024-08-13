use burn::{
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig,
    },
    tensor::{Device, Tensor},
};

use crate::{batcher::Batch, config::MyBackend};

pub struct Model {
    model_size: usize,
    feed_forward_dim: usize,
    attention_heads: usize,
    num_layers: usize,
    transformer: TransformerEncoder<MyBackend>,
}

impl Model {
    pub fn new(device: &Device<MyBackend>) -> Self {
        let model_size = 512;
        let feed_forward_dim = 2048;
        let attention_heads = 8;
        let num_layers = 4;
        let config = TransformerEncoderConfig::new(
            model_size,
            feed_forward_dim,
            attention_heads,
            num_layers,
        );

        Self {
            model_size,
            feed_forward_dim,
            attention_heads,
            num_layers,
            transformer: config.init(device),
        }
    }

    pub fn forward(&self, batch: Batch) -> Tensor<MyBackend, 3> {
        let input = TransformerEncoderInput::new(batch.embeddings).mask_pad(batch.mask);
        self.transformer.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use crate::config;

    use super::*;

    #[test]
    fn test_model() {
        let vocab_size = 100;
        let max_seq_len = 120;
        let device = config::get_device();
        let model = Model::new(&device);
    }
}
