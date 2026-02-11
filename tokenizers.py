import re
import json
from collections import Counter, defaultdict
# --------------------------------------------------------
# WHITESPACE TOKENIZER
# --------------------------------------------------------
class WhitespaceTokenizer:
    """
    Simple whitespace-based tokenizer that treats whitespace
    as separator and punctuation as separate tokens.
    """

    def __init__(self):
        self.name = "whitespace"
        self.punctuation = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

    def train(self, texts):
        pass

    def tokenize(self, text):
        tokens = []
        words = text.split()

        for word in words:
            current_token = []

            for char in word:
                if char in self.punctuation:
                    if current_token:
                        tokens.append("".join(current_token))
                        current_token.clear()
                    tokens.append(char)
                else:
                    current_token.append(char)

            if current_token:
                tokens.append("".join(current_token))

        return tokens

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({'name': self.name}, f)

    def load(self, filepath):
        pass


# --------------------------------------------------------
# REGEX TOKENIZER
# --------------------------------------------------------
REGEX_PATTERNS = {

    # English
    "english_simple": r"[A-Za-z]+|\d+|[^\w\s]",
    "english_advanced": r"[A-Za-z]+(?:['\-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]",

    # Mongolian (Cyrillic)
    "mongolian_simple": r"[А-Яа-яЁёӨөҮү]+|\d+|[^\w\s]",
    "mongolian_advanced": r"[А-Яа-яЁёӨөҮү]+(?:-[А-Яа-яЁёӨөҮү]+)*|\d+(?:\.\d+)?|[^\w\s]",
}

class RegexTokenizer:
    def __init__(self, language="en", advanced=True):
        self.name = "regex"
        self.language = language.lower()

        if self.language in ["en", "english"]:
            key = "english_advanced" if advanced else "english_simple"

        elif self.language in ["mn", "mongolian"]:
            key = "mongolian_advanced" if advanced else "mongolian_simple"

        else:
            raise ValueError(f"Unsupported language: {language}")

        self.pattern = REGEX_PATTERNS[key]
        self.regex = re.compile(self.pattern)

    def train(self, texts):
        pass

    def tokenize(self, text):
        return self.regex.findall(text)

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'name': self.name,
                'pattern': self.pattern,
                'language': self.language
            }, f, ensure_ascii=False)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.pattern = config['pattern']
            self.language = config.get('language', 'english')
            self.regex = re.compile(self.pattern)


# --------------------------------------------------------
# BPE TOKENIZER
# --------------------------------------------------------

class BPETokenizer:
    """
    Scalable Byte Pair Encoding tokenizer.
    """

    def __init__(self, vocab_size=1000, min_frequency=3):
        self.name = "bpe"
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.merges = []

    def _stream_texts(self, texts):
        for line in texts:
            line = line.strip()
            if line:
                yield line

    def _get_stats(self, words):
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, words):
        new_words = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word, freq in words.items():
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = freq

        return new_words

    def train(self, texts, verbose=True, log_every=50):
        word_freqs = Counter()

        for line in self._stream_texts(texts):
            word_freqs.update(line.split())

        words = {}
        for word, freq in word_freqs.items():
            if freq >= self.min_frequency:
                words[" ".join(word) + " </w>"] = freq

        vocab = set()
        for w in words:
            vocab.update(w.split())

        max_merges = self.vocab_size - len(vocab)

        for i in range(max_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_frequency:
                break

            words = self._merge_pair(best_pair, words)

            merged = "".join(best_pair)
            self.merges.append((best_pair, merged))
            vocab.add(merged)

            if verbose and (i + 1) % log_every == 0:
                print(f"[{i+1}] merge {best_pair} -> {merged}")

        print(f"\nTraining complete. Total merges: {len(self.merges)}")

    def tokenize(self, text):
        tokens = []

        for word in text.split():
            symbols = list(word) + ["</w>"]

            for pair, merged in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                        symbols[i:i+2] = [merged]
                    else:
                        i += 1

            if symbols[-1] == "</w>":
                symbols.pop()

            tokens.extend(symbols)

        return tokens

    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "merges": self.merges
            }, f, indent=2)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
            self.vocab_size = data["vocab_size"]
            self.min_frequency = data["min_frequency"]
            self.merges = [tuple(m) for m in data["merges"]]