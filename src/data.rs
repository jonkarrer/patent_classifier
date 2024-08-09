use serde::{Deserialize, Serialize};

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
    pub feature: String,
    pub label: f32,
}

impl DataPoint {
    pub fn new(anchor: &str, target: &str, context: &str, score: f32) -> Self {
        let feature = DataPoint::pre_process(anchor, target, context);
        Self {
            feature,
            label: score,
        }
    }

    pub fn pre_process(anchor: &str, target: &str, context: &str) -> String {
        format!("PHR1: {} PHR2: {} CON: {}", anchor, target, context)
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

pub fn create_dataset(path: &str) -> Vec<DataPoint> {
    let data = collect(path);

    data.into_iter()
        .map(|r| DataPoint::new(&r.anchor, &r.target, &r.context, r.score))
        .collect()
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
    fn test_pre_process() {
        let data = collect("dataset/train.csv");
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
    fn test_create_dataset() {
        let set = create_dataset("dataset/validate.csv");
        assert!(!set.is_empty());
    }
}
