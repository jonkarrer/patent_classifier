use crate::{
    config::MyBackend,
    data::{DataPoint, DataSet},
};
use burn::{
    nn::{
        attention::{generate_padding_mask, GeneratePaddingMask},
        EmbeddingConfig,
    },
    tensor::{Bool, Device, Float, Tensor},
};

#[derive(Debug)]
pub struct Batch {
    pub embeddings: Tensor<MyBackend, 3, Float>,
    pub labels: Tensor<MyBackend, 1, Float>,
    pub mask: Tensor<MyBackend, 2, Bool>,
}

pub struct Batcher {
    pub batch_size: usize,
    pub batches: Vec<Batch>,
}

impl Batcher {
    pub fn new(
        data_set: DataSet,
        batch_size: usize,
        model_size: usize,
        device: &Device<MyBackend>,
    ) -> Self {
        let mut batches = Vec::new();
        for chk in data_set.data_points.chunks(batch_size) {
            batches.push(Batcher::create_batch(
                chk,
                data_set.max_seq_len,
                data_set.vocab_size,
                model_size,
                device,
            ));
        }

        Self {
            batch_size,
            batches,
        }
    }

    fn create_batch(
        data_points: &[DataPoint],
        max_seq_len: usize,
        vocab_size: usize,
        model_size: usize,
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

        let features = mask.tensor.to_device(device);
        let labels = Tensor::from_floats(labels.as_slice(), device).to_device(device);
        let mask = mask.mask.to_device(device);

        let index_positions = Tensor::arange(0..max_seq_len as i64, device)
            .reshape([1, max_seq_len])
            .repeat(0, data_points.len());
        let embedded_features = EmbeddingConfig::new(vocab_size, model_size)
            .init(device)
            .forward(features);
        let embedded_positions = EmbeddingConfig::new(max_seq_len, model_size)
            .init(device)
            .forward(index_positions);

        let embeddings = embedded_features + embedded_positions;

        Batch {
            embeddings,
            labels,
            mask,
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
            DataPoint {
                feature: vec![1, 2, 3, 4],
                label: 0.0,
                seq_len: 4,
            },
            DataPoint {
                feature: vec![5, 6, 7, 8],
                label: 0.0,
                seq_len: 4,
            },
            DataPoint {
                feature: vec![9, 10, 11, 12],
                label: 0.0,
                seq_len: 4,
            },
        ];

        let max_seq_len = 120;
        let vocab_size = 100;
        let model_size = 512;

        let batch = Batcher::create_batch(
            &data,
            max_seq_len,
            vocab_size,
            model_size,
            &config::get_device(),
        );
        assert_eq!(&batch.labels.shape().dims, &[4 as usize]);
        dbg!(batch);
        assert!(false);
    }
}
