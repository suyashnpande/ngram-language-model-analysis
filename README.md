# NLP Tokenization & Language Modeling

Implementation of tokenizers and n-gram language models for text generation.

## Quick Start
```bash
# 1. Place your data in raw_data/
mkdir raw_data
cp your_data.jsonl raw_data/cc100_en.jsonl

# 2. Run
python language_models.py
```

```
raw_data/
    cc100_en.jsonl
    cc100_mn.jsonl
```

This executes:

- Cleaning
- Splitting
- Tokenizer training
- LM training
- Perplexity calculation
- Autocomplete generation
- Model saving

---

##  Tokenizers

### 1️ Whitespace Tokenizer
- Splits on whitespace
- Separates punctuation

### 2️ Regex Tokenizer (Language-aware)

Different regex patterns for:

- English (Latin script)
- Mongolian (Cyrillic script)

Example patterns:

```
English:    [A-Za-z]+(?:['\-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]
Mongolian:  [А-Яа-яЁёӨөҮү]+(?:-[А-Яа-яЁёӨөҮү]+)*|\d+(?:\.\d+)?|[^\w\s]
```

### 3️ Byte Pair Encoding (BPE)

- Configurable vocabulary size
- Minimum frequency threshold
- Iterative merge-based subword learning
- Saves merges in JSON format

Example saved merge:

```python
[
   (("e","</w>"), "e</w>"),
   (("t","h"), "th")
]
```

---

##  Language Model

- 4-gram model
- Start token: `<SOS>`
- End token: `<EOS>`

### Supported Smoothing Methods:

| Mode | Description |
|------|------------|
| none | Maximum Likelihood |
| wb   | Witten–Bell |
| kn   | Kneser–Ney |

---

##  Perplexity

Perplexity is computed as:

\[
PPL = \exp\left(-\frac{1}{N} \sum \log P(w_i | context)\right)
\]

Each tokenizer is evaluated under:

- No smoothing
- Witten–Bell
- Kneser–Ney

---

##  Autocomplete

Greedy decoding is used:

1. Take last (n−1) tokens
2. Compute next-token probabilities
3. Select highest probability
4. Stop at `<EOS>` or max length

Example:

```
Prompt: The government
Completion: The government said that the
```

---

##  Configuration

Inside `language_models.py`:

```python
MAX_TRAIN_SENTENCES = 800_000
MAX_VAL_SENTENCES = 50_000

BPE_VOCAB_SIZE = 8000
BPE_MIN_FREQ = 3
```

You may reduce these values for testing.

---
## Output
```
outputs/
├── tokenizers/
│   └── en/
│       ├── whitespace_tokenizer.json
│       ├── regex_tokenizer.json
│       └── bpe_tokenizer.json
└── language_models/
    └── en/
        ├── whitespace_4gram.pkl
        ├── regex_4gram.pkl
        └── bpe_4gram.pkl
```

## Saved Outputs

### Tokenizers
```
outputs/tokenizers/{language}/
```

### Language Models
```
outputs/language_models/{language}/
```

Models are saved using `pickle`.

---

##  Multilingual Support

The pipeline automatically processes:

```python
LANGUAGES = {
    "en": "cc100_en.jsonl",
    "mn": "cc100_mn.jsonl"
}
```

Each language has:
- Independent splits
- Independent tokenizers
- Independent language models

---

## Academic Contributions

This implementation demonstrates:

- Language-aware regex tokenization
- Subword modeling with BPE
- Comparative smoothing analysis
- Multilingual evaluation
- Fully automated NLP pipeline

---
---