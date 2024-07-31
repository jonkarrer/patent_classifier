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

## Gathering Data

We need to get the data from the csv file and into a format that we can use in Rust. We will use the `csv` crate to do this.
