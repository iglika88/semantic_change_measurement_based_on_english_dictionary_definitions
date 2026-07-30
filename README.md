# Measurement of Multilingual Diachronic Semantic Change Using English-Language Dictionary Definitions

This repository accompanies the article:

> **Measurement of Multilingual Diachronic Semantic Change Using English-Language Dictionary Definitions**

It contains the scripts required to replicate the experiments described in the paper.

## Scripts

The following files implement the semantic change calculation experiments based on the English-language dictionary definitions of lemmas in pairs of dictionaries representing two stages of a language's development:

- `hebrew_experiments.py`
- `greek_experiments.py`
- `latin_french_experiments.py`
- `latin_italian_experiments.py`

Each script reproduces the experimental pipeline for one language pair.


## Hebrew False Friends Dataset

This dataset includes 50 homographs that exist in both Ancient (Biblical) and Modern Hebrew but have significantly different meanings. The dataset may be suitable for automatic model evaluation as well as academic/educational applications.

The pairs were derived through a crawl of Biblical and Modern Hebrew dictionaries and the calculation of cosine distance based on the words' English definitions as represented by two transformer models ([**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [**distiluse-base-multilingual-cased-v2**](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)).

- [**false_friends_BH_vs_MH_final.xlsx**](./false_friends_BH_vs_MH_final.xlsx)  
  dataset in Excel format  
  
Note: The dataset is encoded in UTF-8. If Hebrew characters do not display correctly in the browser preview, please download the file to view it properly.

- [**false_friends_BH_vs_MH_final.csv**](./false_friends_BH_vs_MH_final.csv)  
  dataset in CSV format


## Dictionary resources used

Biblical Hebrew lexical data were extracted from:

- **Open Scriptures Hebrew Lexicon (Brown–Driver–Briggs Hebrew and English Lexicon)**  
  https://github.com/openscriptures/HebrewLexicon

Lemma forms were obtained from the accompanying **LexicalIndex.xml** index.

Modern Hebrew lexical items were obtained from:

- **Kaikki.org Hebrew dictionary dataset**  
  https://kaikki.org/dictionary/Hebrew/

Derived from Wiktionary using:

- **Wiktextract**  
  https://github.com/tatuylonen/wiktextract




## Citation

If you use this code in your research, please cite the corresponding article.


