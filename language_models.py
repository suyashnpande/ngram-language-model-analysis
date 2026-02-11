import os
import math
import pickle
from collections import defaultdict, Counter
import random

from tokenizers import WhitespaceTokenizer, RegexTokenizer, BPETokenizer
from utils import (
    load_jsonl,
    clean_corpus,
    split_data,
    save_splits,
    load_text_file,
)

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "raw_data") 
PROCESSED_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TOKENIZER_DIR = os.path.join(OUTPUT_DIR, "tokenizers")
LM_DIR = os.path.join(OUTPUT_DIR, "language_models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)
os.makedirs(LM_DIR, exist_ok=True)

LANGUAGES = {
    "en": "cc100_en.jsonl",
    "mn": "cc100_mn.jsonl"
}

MAX_TRAIN_SENTENCES = 2_000
MAX_VAL_SENTENCES = 1_000

BPE_VOCAB_SIZE = 100
BPE_MIN_FREQ = 2

# --------------------------------------------------------
# NGRAM MODEL
# --------------------------------------------------------
class NGramLM:
    def __init__(self, n=4, discount=0.75):
        self.n = n
        self.discount = discount

        # one dictionary per order (1..n)
        self.ngram_counts = [defaultdict(Counter) for _ in range(n)]
        self.context_counts = [Counter() for _ in range(n)]

        self.vocab = set()
        self.total_tokens = 0
        self.total_unique_bigrams = 0

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def train(self, corpus):
        for sent in corpus:
            sent = ["<SOS>"]*(self.n-1) + sent + ["<EOS>"]

            for i in range(len(sent)):
                for order in range(1, self.n+1):
                    if i - order + 1 < 0:
                        continue

                    ngram = tuple(sent[i-order+1:i+1])
                    ctx = ngram[:-1]
                    tok = ngram[-1]

                    self.ngram_counts[order-1][ctx][tok] += 1
                    self.context_counts[order-1][ctx] += 1

                    if order == 1:
                        self.vocab.add(tok)
                        self.total_tokens += 1

        # total bigram types for KN continuation
        bigrams = self.ngram_counts[1]
        self.total_unique_bigrams = sum(
            len(bigrams[ctx]) for ctx in bigrams
        )

    # --------------------------------------------------
    # CONTINUATION PROBABILITY (KN BASE CASE)
    # --------------------------------------------------
    def continuation_prob(self, tok):
        bigrams = self.ngram_counts[1]
        count_contexts = sum(
            1 for ctx in bigrams if tok in bigrams[ctx]
        )

        if self.total_unique_bigrams == 0:
            return 1 / len(self.vocab)

        return count_contexts / self.total_unique_bigrams

    # --------------------------------------------------
    # INTERPOLATED KNESER–NEY
    # --------------------------------------------------
    def kn_prob(self, order, ctx, tok):

        # Unigram base case
        if order == 1:
            return self.continuation_prob(tok)

        if ctx not in self.ngram_counts[order-1]:
            return self.kn_prob(order-1, ctx[1:], tok)

        count = self.ngram_counts[order-1][ctx].get(tok, 0)
        total = self.context_counts[order-1][ctx]
        unique_cont = len(self.ngram_counts[order-1][ctx])

        discounted = max(count - self.discount, 0) / total
        lambda_ctx = (self.discount * unique_cont) / total

        return discounted + lambda_ctx * \
               self.kn_prob(order-1, ctx[1:], tok)

    def prob_kneser_ney(self, ctx, tok):
        return self.kn_prob(self.n, ctx, tok)

    # --------------------------------------------------
    # RECURSIVE WITTEN–BELL
    # --------------------------------------------------
    def wb_prob(self, order, ctx, tok):

        if order == 1:
            return 1 / len(self.vocab)

        if ctx not in self.ngram_counts[order-1]:
            return self.wb_prob(order-1, ctx[1:], tok)

        N = self.context_counts[order-1][ctx]
        T = len(self.ngram_counts[order-1][ctx])
        count = self.ngram_counts[order-1][ctx].get(tok, 0)

        if count > 0:
            return count / (N + T)

        backoff_weight = T / (N + T)

        return backoff_weight * \
               self.wb_prob(order-1, ctx[1:], tok)

    def prob_witten_bell(self, ctx, tok):
        return self.wb_prob(self.n, ctx, tok)

    # --------------------------------------------------
    # UNSMOOTHED
    # --------------------------------------------------
    def prob_unsmoothed(self, ctx, tok):
        if ctx not in self.ngram_counts[self.n-1]:
            return 0.0
        return self.ngram_counts[self.n-1][ctx].get(tok, 0) / \
               self.context_counts[self.n-1][ctx]

    # --------------------------------------------------
    # PERPLEXITY
    # --------------------------------------------------
    def perplexity(self, corpus, mode="none"):
        log_prob = 0
        N = 0

        for sent in corpus:
            sent = ["<SOS>"]*(self.n-1) + sent + ["<EOS>"]

            for i in range(len(sent)-self.n+1):
                ctx = tuple(sent[i:i+self.n-1])
                tok = sent[i+self.n-1]

                if mode == "none":
                    p = self.prob_unsmoothed(ctx, tok)
                elif mode == "wb":
                    p = self.prob_witten_bell(ctx, tok)
                else:
                    p = self.prob_kneser_ney(ctx, tok)

                p = max(p, 1e-12)
                log_prob += math.log(p)
                N += 1

        return math.exp(-log_prob / N)

    # --------------------------------------------------
    # AUTOCOMPLETE 
    # --------------------------------------------------
    def autocomplete(self, prompt_tokens, mode="kn", max_len=20):
        tokens = ["<SOS>"]*(self.n-1) + prompt_tokens
        output = []

        for _ in range(max_len):
            ctx = tuple(tokens[-(self.n-1):])

            best_tok = None
            best_prob = -1

            for tok in self.vocab:

                if mode == "none":
                    p = self.prob_unsmoothed(ctx, tok)
                elif mode == "wb":
                    p = self.prob_witten_bell(ctx, tok)
                else:
                    p = self.prob_kneser_ney(ctx, tok)

                if p > best_prob:
                    best_prob = p
                    best_tok = tok

            if best_tok == "<EOS>":
                break

            output.append(best_tok)
            tokens.append(best_tok)

        return output


# --------------------------------------------------------
# MAIN 
# --------------------------------------------------------
if __name__ == "__main__":

    for lang, filename in LANGUAGES.items():

        print("\n" + "="*80)
        print(f"PROCESSING LANGUAGE: {lang.upper()}")
        print("="*80)

        raw_path = os.path.join(DATA_DIR, filename)

        # -----------------------------
        # STEP 1: LOAD RAW
        # -----------------------------
        print("\nSTEP 1: LOAD RAW DATA")
        texts = load_jsonl(raw_path)

        # -----------------------------
        # STEP 2: CLEAN
        # -----------------------------
        print("\nSTEP 2: CLEAN CORPUS")
        cleaned_texts = clean_corpus(texts)

        # -----------------------------
        # STEP 3: SPLIT
        # -----------------------------
        print("\nSTEP 3: SPLIT DATA")
        splits = split_data(cleaned_texts)

        # Save splits per language
        lang_data_dir = os.path.join(PROCESSED_DIR, lang)
        os.makedirs(lang_data_dir, exist_ok=True)
        save_splits(splits, lang_data_dir, lang)

        train_texts = splits["train"][:MAX_TRAIN_SENTENCES]
        test_texts = splits["test"][:MAX_VAL_SENTENCES]

        # -----------------------------
        # STEP 4: TRAIN TOKENIZERS
        # -----------------------------
        print("\nSTEP 4: TRAIN TOKENIZERS")

        ws = WhitespaceTokenizer()
        rg = RegexTokenizer(language=lang)
        bpe = BPETokenizer(
            vocab_size=BPE_VOCAB_SIZE,
            min_frequency=BPE_MIN_FREQ
        )

        ws.train(train_texts)
        rg.train(train_texts)
        bpe.train(train_texts)

        # Save tokenizers per language
        lang_tok_dir = os.path.join(TOKENIZER_DIR, lang)
        os.makedirs(lang_tok_dir, exist_ok=True)

        ws.save(os.path.join(lang_tok_dir, "whitespace_tokenizer.json"))
        rg.save(os.path.join(lang_tok_dir, "regex_tokenizer.json"))
        bpe.save(os.path.join(lang_tok_dir, "bpe_tokenizer.json"))

        TOKENIZERS = {
            "whitespace": ws,
            "regex": rg,
            "bpe": bpe,
        }

        # -----------------------------
        # STEP 5: TOKENIZE CORPUS
        # -----------------------------
        print("\nSTEP 5: TOKENIZE CORPUS")

        tokenized_train = {}
        tokenized_test = {}

        for name, tok in TOKENIZERS.items():
            print(f"Tokenizing with {name}")
            tokenized_train[name] = [tok.tokenize(t) for t in train_texts]
            tokenized_test[name] = [tok.tokenize(t) for t in test_texts]

        # -----------------------------
        # STEP 6: TRAIN LMs
        # -----------------------------
        print("\nSTEP 6: TRAIN LANGUAGE MODELS")

        results = {}

        for name in TOKENIZERS:
            print(f"\nTraining LM ({name})")

            lm = NGramLM(n=4, discount=0.75)
            lm.train(tokenized_train[name])

            ppl_none = lm.perplexity(tokenized_test[name], "none")
            ppl_wb = lm.perplexity(tokenized_test[name], "wb")
            ppl_kn = lm.perplexity(tokenized_test[name], "kn")

            results[name] = {
                "none": ppl_none,
                "wb": ppl_wb,
                "kn": ppl_kn,
                "model": lm,
            }

            print(f"No smoothing: {ppl_none:.2f}")
            print(f"Witten-Bell:  {ppl_wb:.2f}")
            print(f"Kneser-Ney:   {ppl_kn:.2f}")

            # Save LM per language
            lang_lm_dir = os.path.join(LM_DIR, lang)
            os.makedirs(lang_lm_dir, exist_ok=True)

            with open(os.path.join(lang_lm_dir, f"{name}_4gram.pkl"), "wb") as f:
                pickle.dump(lm, f)

        # -----------------------------
        # STEP 7: AUTOCOMPLETE
        # -----------------------------
        print("\nSTEP 7: AUTOCOMPLETE")

        PROMPTS = [
            "The government",
            "Artificial intelligence",
        ]

        for name, tok in TOKENIZERS.items():
            print(f"\n{name.upper()} TOKENIZER")
            lm = results[name]["model"]

            for mode in ["none", "wb", "kn"]:
                print(f"\n--- {mode.upper()} ---")

                for prompt in PROMPTS:
                    tokens = tok.tokenize(prompt)
                    completion = lm.autocomplete(tokens, mode=mode, max_len=10)

                    if name == "bpe":
                        text_out = prompt + " " + "".join(completion).replace("</w>", " ")
                    else:
                        text_out = prompt + " " + " ".join(completion)

                    print(text_out)

        print("\n Language complete:", lang.upper())