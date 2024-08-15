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

Tokenization is a two-step process.

1. First the tokenizer segments the data into small chunks. This is a mix of science and art, as you can split up a sentence a bunch of different ways.
2. Second is numericalization:. This is the process of encoding the tokens into integers so they can be processed by the neural network.

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

## Batching

We need to batch the data. This allows us to process the data in smaller chunks, and makes efficient use of the GPU.

Our dataset is classified and tokenized, but this is still not enough for a neural network to process. We need to create tensors for the data. But what exactly IS our data? You could say that we have:

- Variable length arrays of numbers that represent the tokens (features)
- Arrays of float numbers that represent the similarity score (label)

Problem #1: The neural network does not like variable length sequences. So we need to pad the sequences with 0.
Problem #2: The neural network does not like primitive arrays, it prefers tensors.
Problem #3: The neural network does not know the MEANING of the data.

### Padding masks

Padding masks are used to eliminate problem #1. First we need to pad the short sequences up to the max sequence length. Then a padding mask is a binary array that tells the neural network if a value in the array is a padding value or not.

Padding example:

> [[1, 2, 3], [4, 5, 6, 7, 8]] -> [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]

Padding mask example:

> [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

Our labels are fine, as they are not sequences.

### Creating tensors

Now that we have the padding masks, we can create the tensors.

### Embedding

Our tensors are ready to be fed into the neural network, but there is still one more issue. The NN does not know the meaning of the data. And how could it? It only sees the numbers as they are.

The answer is embeddings. Embeddings are numbers that represent the meaning of the data. Our tokenizer made chunks (tokens), and then converted them into numbers (tokens ids), which built up a vocabulary. An embedding will take these discrete token ids and represent them with dense vectors. This is known as the **embedding layer**.

Position is also a crucial factor. Words need context, because context changes the meaning of the word. Words can have various definitions, so context will help exact the meaning of the word. So we need a **position embedding**. This helps the model understand the position of the token in the sentence.

### Creating a batch of data

We can bring all these things together into a batch.

## Model

The current status quo for Natural Language Processing is to use Transformers. This is a neural network architecture designed to understand the relationship between sequential data points.

Our model will use a transformer to process the embeddings with a forward pass first. We will then pass the output of the transformer to a linear layer to get the predictions based on our classes. Lastly our loss function will calculate the loss between the predictions and the labels.
