use burn::data::dataset::InMemDataset;
use serde::{Deserialize, Serialize};
use tokenizers::Encoding;

// Represents a single row in the csv dataset
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PatentRecord {
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataPoint {
    pub encoded_text: Encoding,
    pub classification: String,
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

fn preprocess(rec: &PatentRecord) -> String {
    format!(
        "PHR1: {} PHR2: {} CON: {}",
        rec.anchor, rec.target, rec.context
    )
}

fn tokenize(text: String) -> Encoding {
    let tokenizer = tokenizers::tokenizer::Tokenizer::from_pretrained("bert-base-cased", None)
        .expect("Tokenizer not initialized");

    tokenizer.encode(text, false).expect("Encoding failed")
}

fn classify(record: &PatentRecord) -> DataPoint {
    let text = preprocess(record);
    let tokenized = tokenize(text);
    DataPoint {
        encoded_text: tokenized,
        classification: record.score.to_string(),
    }
}

pub fn create_dataset(path: &str) -> InMemDataset<DataPoint> {
    let data = collect(path);
    let set = data.into_iter().map(|rec| classify(&rec)).collect();

    InMemDataset::new(set)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_collect() {
        let data = collect("dataset/train.csv");
        dbg!(&data);
        assert!(data.len() > 0);
    }

    #[test]
    fn test_preprocess() {
        let data = collect("dataset/train.csv");
        let processed = preprocess(&data[0]);
        dbg!(&processed);
        assert!(processed.contains("PHR1:"));
    }

    #[test]
    fn test_tokenize() {
        let data = collect("dataset/train.csv");
        let processed = preprocess(&data[0]);
        let tokenized = tokenize(processed);
        dbg!(&tokenized);
        assert!(tokenized.get_ids().len() > 0);
    }

    #[test]
    fn test_classify() {
        let data = collect("dataset/train.csv");
        let processed = classify(&data[0]);
        dbg!(&processed);
        assert!(processed.encoded_text.get_ids().len() > 0);
    }
}
