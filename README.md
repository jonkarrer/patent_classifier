# Patent Classifier

An attempt to train and nlp to classify patent types.

## Dataset

The data comes from [the PatentsView](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/data).

Their description is:

> In this dataset, you are presented pairs of phrases (an anchor and a target phrase) and asked to rate how similar they are on a scale from 0 (not at all similar) to 1 (identical in meaning). This challenge differs from a standard semantic similarity task in that similarity has been scored here within a patent's context, specifically its CPC classification (version 2021.05), which indicates the subject to which the patent relates.

### Inputs

- anchor - the first phrase
- target - the second phrase
- context - the CPC classification (version 2021.05), which indicates the subject within which the similarity is to be scored

### Labels

- Similarity Score
  - 1.0 - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. “the”, “and”, “or”).
  - 0.75 - Close synonym, e.g. “mobile phone” vs. “cellphone”. This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
  - 0.5 - Synonyms which don’t have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
  - 0.25 - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
  - 0.0 - Unrelated.

### Submission

- ID
- Similarity Score

### Notes

This is a **Nominal** dataset.

## Preparing Data

### Collecting

We need to get the data from the csv file and into a format that we can use in Rust. We will use the `csv` crate to do this, along with the `serde` crate to deserialize the data.

```rust
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

fn collect(path: &str) -> Vec<PatentRecord> {
    let mut reader = csv::ReaderBuilder::new()
        .from_path(std::path::Path::new(path))
        .expect("Path not found");

    reader
        .deserialize()
        .map(|item| item.expect("Not a valid record"))
        .collect()
}
```

### Tokenizing

Now we need to turn the data into tokens. We will use the `tokenizers` crate to do this. Before we can tokenize the strings, it is necessary to format them. This usually consists of the phrase and the CPC classification.

```rust
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
```

### Classifying

We need to classify the data entry now. This is simply combining our encoded text with the classification.

```rust
fn classify(record: &PatentRecord) -> DataPoint {
    let text = preprocess(record);
    let tokenized = tokenize(text);
    DataPoint {
        encoded_text: tokenized,
        classification: record.score.to_string(),
    }
}
```

### Creating Dataset

Now we can bring it all together, and store the DataPoint in an InMemDataset.

```rust
use burn::data::dataset::InMemDataset;

pub fn create_dataset(path: &str) -> InMemDataset<DataPoint> {
    let data = collect(path);
    let set = data.into_iter().map(|rec| classify(&rec)).collect();

    InMemDataset::new(set)
}
```

### Split Dataset

Now that we have those neat functions set up, we can easily split the dataset. We need to split it into a training, validation set, and test set. For now we will only worry about the first two.

```rust
    let training_set = data::create_dataset("dataset/train.csv");
    let validation_set = data::create_dataset("dataset/validate.csv");
```
