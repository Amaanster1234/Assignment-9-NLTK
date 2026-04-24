"""
Assignment 9: NLTK
Author: Amaan Sadiq

Purpose:
Perform NLP analysis on multiple text files using tokenization,
stemming, lemmatization, named entity recognition, and trigrams.
"""

import string
from collections import Counter

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams


def setup_nltk():
    """
    Downloads the NLTK resources needed for this assignment.
    """
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
        "wordnet",
        "omw-1.4"
    ]

    for resource in resources:
        nltk.download(resource, quiet=True)


def load_text(file_name):
    """
    Opens and reads a text file.
    """
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: {file_name} was not found.")
        return ""


def tokenize_and_clean(text):
    """
    Tokenizes text, removes stop words, removes punctuation,
    and converts words to lowercase.
    """
    stop_words = set(stopwords.words("english"))
    punctuation = set(string.punctuation)

    tokens = word_tokenize(text)
    cleaned_tokens = []

    for token in tokens:
        token = token.lower()

        if token.isalpha() and token not in stop_words and token not in punctuation:
            cleaned_tokens.append(token)

    return cleaned_tokens


def stem_tokens(tokens):
    """
    Stems tokens using PorterStemmer.
    """
    stemmer = PorterStemmer()
    stemmed_tokens = []

    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens


def lemmatize_tokens(tokens):
    """
    Lemmatizes tokens using WordNetLemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []

    for token in tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(token))

    return lemmatized_tokens


def get_top_tokens(tokens, amount=20):
    """
    Finds the most common tokens.
    """
    freq_dist = FreqDist(tokens)
    return freq_dist.most_common(amount)


def get_named_entities(text):
    """
    Uses NLTK named entity recognition to find and count named entities.
    """
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    chunked_tokens = ne_chunk(tagged_tokens)

    entities = []

    for chunk in chunked_tokens:
        if hasattr(chunk, "label"):
            entity_name = " ".join(word for word, tag in chunk)
            entity_label = chunk.label()
            entities.append((entity_name, entity_label))

    return entities


def get_trigrams(tokens, amount=20):
    """
    Finds the most common 3-word phrases from cleaned tokens.
    """
    trigram_list = list(ngrams(tokens, 3))
    trigram_counts = Counter(trigram_list)

    return trigram_counts.most_common(amount)


def get_trigrams_for_comparison(text, amount=100):
    """
    Creates trigrams for author comparison.

    This version keeps stop words because repeated phrases can help
    show writing style. It only removes punctuation and numbers.
    """
    tokens = word_tokenize(text.lower())
    cleaned_tokens = []

    for token in tokens:
        if token.isalpha():
            cleaned_tokens.append(token)

    trigram_list = list(ngrams(cleaned_tokens, 3))
    trigram_counts = Counter(trigram_list)

    return trigram_counts.most_common(amount)


def analyze_text(title, text):
    """
    Runs the full analysis for one text.
    """
    print("\n" + "=" * 60)
    print(f"Analysis for: {title}")
    print("=" * 60)

    tokens = tokenize_and_clean(text)
    stemmed_tokens = stem_tokens(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens)

    print(f"Total tokens after cleaning: {len(tokens)}")
    print(f"Unique tokens after cleaning: {len(set(tokens))}")

    print("\nTop 20 Regular Tokens:")
    for word, count in get_top_tokens(tokens):
        print(f"{word}: {count}")

    print("\nTop 20 Stemmed Tokens:")
    for word, count in get_top_tokens(stemmed_tokens):
        print(f"{word}: {count}")

    print("\nTop 20 Lemmatized Tokens:")
    for word, count in get_top_tokens(lemmatized_tokens):
        print(f"{word}: {count}")

    entities = get_named_entities(text)

    print(f"\nNumber of Named Entities: {len(entities)}")

    print("\nSample Named Entities:")
    for entity, label in entities[:20]:
        print(f"{entity} ({label})")

    print("\nTop 20 Trigrams:")
    for trigram, count in get_trigrams(tokens):
        print(f"{' '.join(trigram)}: {count}")

    return tokens


def compare_texts(raw_text_dict):
    """
    Compares Text 4 to Texts 1-3 using top trigram overlap.
    """
    print("\n" + "=" * 60)
    print("Trigram Comparison: Text 4 Compared to Texts 1-3")
    print("=" * 60)

    text4_trigrams = set(
        trigram for trigram, count in get_trigrams_for_comparison(raw_text_dict["Text 4"], 100)
    )

    scores = {}

    for name in raw_text_dict:
        if name != "Text 4":
            other_trigrams = set(
                trigram for trigram, count in get_trigrams_for_comparison(raw_text_dict[name], 100)
            )

            overlap = text4_trigrams.intersection(other_trigrams)
            scores[name] = len(overlap)

            print(f"\nText 4 vs {name}")
            print(f"Matching top trigrams: {len(overlap)}")

            if len(overlap) > 0:
                print("Examples of matching trigrams:")
                for trigram in list(overlap)[:10]:
                    print(" ".join(trigram))

    likely_match = max(scores, key=scores.get)

    print("\nMost likely author match for Text 4:")
    print(likely_match)


def main():
    """
    Main function that controls the program.
    """
    setup_nltk()

    files = {
        "Text 1": "RJ_Martin.txt",
        "Text 2": "RJ_Tolkein.txt",
        "Text 3": "RJ_Lovecraft.txt",
        "Text 4": "Martin.txt"
    }

    raw_texts = {}
    text_tokens = {}

    for name, file_name in files.items():
        text = load_text(file_name)
        raw_texts[name] = text

        tokens = analyze_text(name, text)
        text_tokens[name] = tokens

    compare_texts(raw_texts)

    print("\nProgram complete.")


if __name__ == "__main__":
    main()