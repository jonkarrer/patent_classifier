use burn::{
    module::Module,
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Linear, LinearConfig,
    },
    tensor::{Bool, Device, Tensor},
};

use crate::config::MyBackend;

#[derive(Module, Debug, Clone)]
pub struct Model {
    model_size: usize,
    feed_forward_dim: usize,
    attention_heads: usize,
    num_layers: usize,
    transformer: TransformerEncoder<MyBackend>,
    linear_layer: Linear<MyBackend>,
}

impl Model {
    pub fn new(model_size: usize, device: &Device<MyBackend>) -> Self {
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

    pub fn forward(
        &self,
        embeddings: Tensor<MyBackend, 3>,
        mask: Tensor<MyBackend, 2, Bool>,
    ) -> Tensor<MyBackend, 3> {
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
