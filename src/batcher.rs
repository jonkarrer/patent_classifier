use crate::{config::MyBackend, data::DataPoint};
use burn::{
    nn::attention::{generate_padding_mask, GeneratePaddingMask},
    tensor::{Bool, Device, Float, Int, Tensor},
};

struct TokenizedDataPoint {
    input_ids: Vec<usize>,
    label: f32,
}

#[derive(Debug)]
pub struct Batch {
    pub features: Tensor<MyBackend, 2, Int>,
    pub labels: Tensor<MyBackend, 1, Float>,
    pub padding_masks: Tensor<MyBackend, 2, Bool>,
}

fn tokenize(text: &str) -> Vec<usize> {
    let tokenizer = tokenizers::tokenizer::Tokenizer::from_pretrained("bert-base-cased", None)
        .expect("Tokenizer not initialized");

    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    tokens.get_ids().into_iter().map(|x| *x as usize).collect()
}

pub fn create_batch(data_points: &[DataPoint], device: Device<MyBackend>) -> Batch {
    let mut max_seq_len = 0;
    let mut tokenized_data = Vec::new();

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for dp in data_points {
        let input_ids = tokenize(dp.feature.as_str());

        if input_ids.len() > max_seq_len {
            max_seq_len = input_ids.len();
        }

        tokenized_data.push(TokenizedDataPoint {
            input_ids,
            label: dp.label,
        });
    }

    for mut dp in tokenized_data {
        if dp.input_ids.len() < max_seq_len {
            let padding = vec![0; max_seq_len - dp.input_ids.len()];
            dp.input_ids.extend(padding);
        }

        features.push(dp.input_ids);
        labels.push(dp.label);
    }

    for dp in data_points {
        let input_ids = tokenize(dp.feature.as_str());

        if input_ids.len() > max_seq_len {
            max_seq_len = input_ids.len();
        }
    }

    let mask: GeneratePaddingMask<MyBackend> =
        generate_padding_mask(0, features, Some(max_seq_len), &device);

    Batch {
        features: mask.tensor,
        labels: Tensor::from_floats(labels.as_slice(), &device),
        padding_masks: mask.mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let processed = DataPoint::pre_process("telephone", "communications", "DO3");
        let encoded_text = tokenize(&processed);

        dbg!(&encoded_text);
        assert!(encoded_text.len() == 18);
    }
}
