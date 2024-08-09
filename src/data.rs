use serde::{Deserialize, Serialize};

// Represents a single row in the csv dataset
#[derive(Serialize, Deserialize, Debug)]
struct PatentRecord {
    // Unique patent ID
    pub id: String,

    // The first phrase of the patent
    pub anchor: String,

    // The second phrase of the patent
    pub target: String,

    // CPC classification which indicates the subject within which the similarity is scored
    pub context: String,

    // The similarity score
    pub score: f32,
}

#[derive(Clone)]
pub struct DataPoint {
    pub feature: Vec<usize>,
    pub label: f32,
    pub seq_len: usize,
}

impl DataPoint {
    pub fn new(anchor: &str, target: &str, context: &str, score: f32) -> Self {
        let context_string = DataPoint::pre_process(anchor, target, context);
        let feature = DataPoint::tokenize(&context_string);
        let seq_len = feature.len();

        Self {
            feature,
            label: score,
            seq_len,
        }
    }

    fn tokenize(text: &str) -> Vec<usize> {
        let tokenizer = tokenizers::tokenizer::Tokenizer::from_pretrained("bert-base-cased", None)
            .expect("Tokenizer not initialized");

        let tokens = tokenizer.encode(text, false).expect("Encoding failed");
        tokens.get_ids().into_iter().map(|x| *x as usize).collect()
    }

    fn pre_process(anchor: &str, target: &str, context: &str) -> String {
        format!("PHR1: {} PHR2: {} CON: {}", anchor, target, context)
    }
}

pub struct DataSet {
    pub data_points: Vec<DataPoint>,
    pub max_seq_len: usize,
}

impl DataSet {
    pub fn new(path: &str) -> Self {
        let records = Self::collect(path);
        let mut max_seq_len = 0;
        let mut data_points = Vec::new();

        for rec in records.iter() {
            let dp = DataPoint::new(&rec.anchor, &rec.target, &rec.context, rec.score);
            let sl = dp.seq_len;
            if sl > max_seq_len {
                max_seq_len = sl;
            }

            data_points.push(dp);
        }

        Self {
            data_points,
            max_seq_len,
        }
    }

    fn collect(path: &str) -> Vec<PatentRecord> {
        let mut reader = csv::ReaderBuilder::new()
            .from_path(std::path::Path::new(path))
            .expect("Path not found");

        reader
            .deserialize()
            .map(|item| item.expect("Not a valid record"))
            .collect()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_collect() {
        let data = DataSet::collect("dataset/train.csv");
        dbg!(&data);
        assert!(data.len() > 0);
    }

    #[test]
    fn test_pre_process() {
        let data = DataSet::collect("dataset/train.csv");
        let PatentRecord {
            anchor,
            target,
            context,
            ..
        } = &data[0];

        let processed = DataPoint::pre_process(anchor, target, context);
        dbg!(&processed);
        assert!(processed.contains("PHR1:"));
    }

    #[test]
    fn test_tokenize() {
        let processed = DataPoint::pre_process("telephone", "communications", "DO3");
        let encoded_text = DataPoint::tokenize(&processed);

        dbg!(&encoded_text);
        assert!(encoded_text.len() == 18);
    }
}
