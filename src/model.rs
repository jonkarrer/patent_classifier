use burn::{
    nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    tensor::{Device, Tensor},
};

use crate::config::MyBackend;

pub struct Model {
    transformer: TransformerEncoder<MyBackend>,
}

impl Model {
    pub fn new(device: &Device<MyBackend>) -> Self {
        let config = TransformerEncoderConfig::new(512, 2048, 8, 4);
        Self {
            transformer: config.init(device),
        }
    }

    pub fn forward(&self, features: TransformerEncoderInput<MyBackend>) -> Tensor<MyBackend, 3> {
        self.transformer.forward(features)
    }
}
