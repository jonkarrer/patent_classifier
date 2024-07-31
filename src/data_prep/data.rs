use anyhow::Ok;
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

pub struct ClassifiedPatentRecord {
    pub id: String,
    pub anchor: String,
    pub target: String,
    pub context: String,
    pub score: f32,
    pub classification: String,
}

fn process(rec: &PatentRecord) -> Encoding {
    let formatted_text = format!(
        "PHR1: {} PHR2: {} CON: {}",
        rec.anchor, rec.target, rec.context
    );

    let tokenizer = tokenizers::tokenizer::Tokenizer::from_pretrained("bert-base-cased", None)
        .expect("Tokenizer not initialized");

    tokenizer
        .encode(formatted_text, false)
        .expect("Encoding failed")
}

fn collect(path: &str) -> anyhow::Result<Vec<PatentRecord>> {
    let mut reader = csv::ReaderBuilder::new().from_path(std::path::Path::new(path))?;
    let rows: Vec<PatentRecord> = reader
        .deserialize()
        .map(|item| item.expect("Not a valid record"))
        .collect();

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_csv() {
        let data = collect("dataset/train.csv").unwrap();
        assert!(data.len() > 0);
    }

    #[test]
    fn test_preprocess() {
        let data = collect("dataset/train.csv").unwrap();
        let processed = process(&data[0]);
        assert!(processed.get_ids().len() == processed.get_tokens().len());
    }
}
