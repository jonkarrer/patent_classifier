use burn::{
    module::Module,
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Linear, LinearConfig,
    },
    prelude::Backend,
    tensor::{Bool, Device, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    model_size: usize,
    feed_forward_dim: usize,
    attention_heads: usize,
    num_layers: usize,
    transformer: TransformerEncoder<B>,
    linear_layer: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(model_size: usize, device: &Device<B>) -> Self {
        let feed_forward_dim = 2048;
        let attention_heads = 8;
        let num_layers = 4;
        let num_classes = 5;

        let config = TransformerEncoderConfig::new(
            model_size,
            feed_forward_dim,
            attention_heads,
            num_layers,
        );

        let linear_layer = LinearConfig::new(model_size, num_classes).init(device);

        Self {
            model_size,
            feed_forward_dim,
            attention_heads,
            num_layers,
            transformer: config.init(device),
            linear_layer,
        }
    }

    pub fn forward(&self, embeddings: Tensor<B, 3>, mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let input = TransformerEncoderInput::new(embeddings).mask_pad(mask);
        self.linear_layer.forward(self.transformer.forward(input))
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
        // let model = Model::new(&device);

        // let m = model.forward(embeddings, mask, labels)
    }
}
