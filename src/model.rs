use burn::nn::transformer::TransformerEncoder;

use crate::config::MyBackend;

pub struct Model {
    transformer: TransformerEncoder<MyBackend>,
}
