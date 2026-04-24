# Assignment 9: NLTK Text Analysis

## Purpose

The purpose of this project is to perform natural language processing (NLP) on multiple unstructured text files using the NLTK library in Python. The goal is to analyze patterns within the texts and compare them using techniques such as tokenization, stemming, lemmatization, named entity recognition, and n-gram analysis.

This project also attempts to determine the likely author of a fourth text by comparing writing patterns found in the first three texts.

---

## Files Used

* RJ_Martin.txt (Text 1)
* RJ_Tolkein.txt (Text 2)
* RJ_Lovecraft.txt (Text 3)
* Martin.txt (Text 4)

---

## Program Design

This program is written using a **function-based approach**, which keeps the structure simple and consistent with previous assignments.

Each major NLP task is separated into its own function. This makes the code easier to read, debug, and reuse.

---

## Implementation Details

### 1. NLTK Setup

The program begins by downloading all required NLTK resources such as tokenizers, stopwords, and models for part-of-speech tagging and named entity recognition.

---

### 2. Tokenization and Cleaning

Each text file is:

* Tokenized into individual words
* Converted to lowercase
* Cleaned by removing:

  * Stop words (common words like "the", "is", "and")
  * Punctuation
  * Non-alphabetic tokens

This step ensures that only meaningful words are analyzed.

---

### 3. Stemming

Stemming reduces words to their root form using the PorterStemmer.

Example:

* "running" → "run"
* "fighting" → "fight"

This helps group similar words together.

---

### 4. Lemmatization

Lemmatization converts words into their dictionary form using WordNet.

Example:

* "better" → "good"
* "was" → "be"

This produces more accurate base forms than stemming.

---

### 5. Word Frequency Analysis

The program calculates the most common words using NLTK’s `FreqDist`.

For each text, it outputs:

* Total number of tokens
* Number of unique tokens
* Top 20 most common words

This helps identify key themes and vocabulary differences between texts.

---

### 6. Named Entity Recognition (NER)

Named entities are extracted using:

* Part-of-speech tagging
* `nltk.ne_chunk()`

Entities may include:

* People
* Locations
* Organizations

The program counts and displays these entities to help identify the subject of each text.

---

### 7. Trigram Analysis (n = 3)

Trigrams are sequences of three consecutive words.

Example:

* "king of the"
* "battle of winter"

The program:

* Finds the most common trigrams in each text
* Uses them to analyze writing patterns

---

### 8. Author Comparison

To determine the likely author of Text 4:

* The program generates trigrams for all texts
* It compares the top 100 trigrams from Text 4 with Texts 1–3
* It counts overlapping trigrams

The text with the highest overlap is considered the most likely author match.

---

## Results Summary

* Token frequency reveals key vocabulary differences between authors
* Named entities help identify subjects and themes of each text
* Trigram patterns provide insight into writing style
* Text 4 is matched with the most similar text based on trigram overlap

---

## Limitations

* Removing stop words can reduce meaningful phrase matching
* Trigram comparison may not always be accurate for small datasets
* Named entity recognition is not perfect and may misclassify words
* The texts are relatively short, which limits statistical accuracy

---

## How to Run

1. Place all text files in the same directory as `main.py`
2. Run the program:

```
python main.py
```

3. The output will display analysis for each text and the author comparison results.

---

## Notes

This project follows concepts covered in class, including:

* Tokenization
* Stop word removal
* Word frequency analysis
* Stemming and lemmatization
* Named entity recognition
* N-grams

