use derive_new::new;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

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

#[derive(Clone, new)]
pub struct DataPoint {
    pub feature: Vec<usize>,
    pub label: i32,
    pub seq_len: usize,
}

pub struct DataSet {
    pub data_points: Vec<DataPoint>,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl DataSet {
    pub fn new(path: &str, tokenizer: &Tokenizer) -> Self {
        let records = Self::collect(path);
        let mut max_seq_len = 0;
        let mut data_points = Vec::new();

        for r in records.iter() {
            let feature = Self::tokenize_text(tokenizer, &r.anchor, &r.target, &r.context);
            let seq_len = feature.len();

            let dp = DataPoint::new(feature, Self::format_label(r.score), seq_len);

            let sl = dp.seq_len;
            if sl > max_seq_len {
                max_seq_len = sl;
            }

            data_points.push(dp);
        }

        let vocab_size = tokenizer.get_vocab_size(true);

        Self {
            data_points,
            max_seq_len,
            vocab_size,
        }
    }

    fn format_label(score: f32) -> i32 {
        (score * 10.0) as i32
    }

    fn tokenize_text(
        tokenizer: &Tokenizer,
        anchor: &str,
        target: &str,
        context: &str,
    ) -> Vec<usize> {
        let text = format!("PHR1: {} PHR2: {} CON: {}", anchor, target, context);
        let tokens = tokenizer.encode(text, false).expect("Encoding failed");
        tokens.get_ids().into_iter().map(|x| *x as usize).collect()
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

pub fn class_name(label: i32) -> String {
    match label {
        0 => "Unrelated",
        2 => "Somewhat Related",
        5 => "Different Meaning Synonym",
        7 => "Close Synonym",
        10 => "Very Close Match",
        _ => panic!("Invalid label"),
    }
    .to_string()
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
    fn test_tokenize() {
        let tokenizer = tokenizers::tokenizer::Tokenizer::from_pretrained("bert-base-cased", None)
            .expect("Tokenizer not initialized");

        let encoded_text = DataSet::tokenize_text(&tokenizer, "telephone", "communications", "DO3");

        dbg!(&encoded_text);
        assert!(encoded_text.len() == 18);
    }

    #[test]
    fn test_format_label() {
        let score = DataSet::format_label(0.75);
        dbg!(&score);
        assert!(score == 5);
    }
}
