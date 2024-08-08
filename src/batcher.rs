use crate::data::DataPoint;
use burn::{
    prelude::Backend,
    tensor::{Device, Int, Tensor},
};
use tokenizers::Encoding;

pub fn tokenize(text: String) -> Encoding {
    let tokenizer = tokenizers::tokenizer::Tokenizer::from_pretrained("bert-base-cased", None)
        .expect("Tokenizer not initialized");

    tokenizer.encode(text, false).expect("Encoding failed")
}

fn feature_tensor<B: Backend>(encoded_text: &Encoding, device: Device<B>) -> Tensor<B, 1, Int> {
    let input_ids: Vec<i32> = encoded_text
        .get_ids()
        .into_iter()
        .map(|x| *x as i32)
        .collect();

    Tensor::from_ints(input_ids.as_slice(), &device)
}

fn label_tensor<B: Backend>(data_point: &DataPoint, device: Device<B>) -> Tensor<B, 1> {
    let label = data_point.label;

    Tensor::from_floats([label], &device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{
        wgpu::{AutoGraphicsApi, WgpuDevice},
        Autodiff, Wgpu,
    };
    type MyDevice = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyBackend = Autodiff<MyDevice>;

    #[test]
    fn test_tokenize() {
        let processed = DataPoint::pre_process("telephone", "communications", "DO3");
        let encoded_text = tokenize(processed);

        dbg!(&encoded_text);
        assert!(encoded_text.get_ids().len() == 18);
    }

    #[test]
    fn test_feature_tensor() {
        let processed = DataPoint::pre_process("telephone", "communications", "DO3");
        let encoded_text = tokenize(processed);

        let device = WgpuDevice::default();
        let tensor = feature_tensor::<MyBackend>(&encoded_text, device);

        assert!(tensor.shape().dims[0] == 18 as usize);
    }
}
