use crate::{
    config::MyBackend,
    data::{DataPoint, DataSet},
};
use burn::{
    nn::attention::{generate_padding_mask, GeneratePaddingMask},
    tensor::{Bool, Device, Float, Int, Tensor},
};

#[derive(Debug)]
pub struct Batch {
    pub max_seq_len: usize,
    pub features: Tensor<MyBackend, 2, Int>,
    pub labels: Tensor<MyBackend, 1, Float>,
    pub padding_masks: Tensor<MyBackend, 2, Bool>,
}

pub struct Batcher {
    pub batch_size: usize,
    pub batches: Vec<Batch>,
}

impl Batcher {
    pub fn new(
        data_points: Vec<DataPoint>,
        max_seq_len: usize,
        batch_size: usize,
        device: &Device<MyBackend>,
    ) -> Self {
        let mut batches = Vec::new();
        for chk in data_points.chunks(batch_size) {
            batches.push(Batcher::create_batch(chk, max_seq_len, device));
        }

        Self {
            batch_size,
            batches,
        }
    }

    fn create_batch(
        data_points: &[DataPoint],
        max_seq_len: usize,
        device: &Device<MyBackend>,
    ) -> Batch {
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for mut dp in data_points.to_vec().into_iter() {
            if dp.seq_len < max_seq_len {
                let padding = vec![0; max_seq_len - dp.seq_len];
                dp.feature.extend(padding);
            }

            features.push(dp.feature);
            labels.push(dp.label);
        }

        let mask: GeneratePaddingMask<MyBackend> =
            generate_padding_mask(0, features, Some(max_seq_len), device);

        Batch {
            features: mask.tensor,
            labels: Tensor::from_floats(labels.as_slice(), device),
            padding_masks: mask.mask,
            max_seq_len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    #[test]
    fn test_create_batch() {
        let data = vec![
            DataPoint::new("hello", "cross", "DO3", 0.75),
            DataPoint::new("hello", "cross", "DO3", 0.75),
            DataPoint::new("hello", "cross", "DO3", 0.75),
            DataPoint::new("hello", "cross", "DO3", 0.75),
        ];

        let max_seq_len = 120;
        let batch = Batcher::create_batch(&data, max_seq_len, &config::get_device());
        assert_eq!(&batch.labels.shape().dims, &[4 as usize]);
        assert_eq!(
            &batch.features.shape().dims,
            &[4 as usize, max_seq_len as usize]
        );
        assert_eq!(
            &batch.padding_masks.shape().dims,
            &[4 as usize, max_seq_len as usize]
        );
    }
}
