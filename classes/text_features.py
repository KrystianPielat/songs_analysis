import warnings
import pandas as pd
import nltk
from nltk.tokenize import casual_tokenize, word_tokenize
from collections import Counter
from nltk.corpus import cmudict, stopwords, wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from textstat import textstat
from langdetect import detect, DetectorFactory
from sklearn.decomposition import PCA
import string
import statistics
from wordfreq import word_frequency
import pronouncing
import re
from nltk import pos_tag
from typing import List, Optional
from nltk.stem import SnowballStemmer
from classes.feature_extractor import FeatureExtractor
from langcodes import Language

# Set seed for reproducibility in langdetect
DetectorFactory.seed = 0

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)



class TextFeatureExtractor(FeatureExtractor):
    def __init__(self) -> None:
        """Initializes the TextFeatureExtractor with sentiment and TF-IDF analyzers."""
        self.sid = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer()
        self.default_stemmer = SnowballStemmer("english")

    @staticmethod
    def preprocess_text(text: str, stemmer: Optional[SnowballStemmer] = None, stopword_list: Optional[set] = None) -> str:
        """
        Preprocess text by removing stopwords, punctuation, numbers, 
        and applying stemming after tokenization.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        tokens = casual_tokenize(text)
        if text.startswith("letra de"):
            tokens = tokens[5:]  # Remove "letra de" + average title length
        if stopword_list is not None:
            tokens = [token for token in tokens if token not in stopword_list]
        if stemmer is not None:
            words = [stemmer.stem(token) for token in tokens]
        return " ".join(tokens)


    def get_stemmer_and_stopwords(self, language: str) -> (Optional[SnowballStemmer], Optional[set]):
        """Returns the appropriate stemmer and stopword list for the given language."""
        try:
            stemmer = SnowballStemmer(language)
        except ValueError:
            stemmer = None

        try:
            stopword_list = set(stopwords.words(language))
        except OSError:
            stopword_list = None

        return stemmer, stopword_list

    def extract_features(self, lyrics: str) -> dict:
        """Extracts various text features from a single lyrics string."""
        # Detect language
        try:
            language = Language.make(detect(lyrics)).display_name().lower()
        except Exception:
            language = "unknown"

        # Get the appropriate stemmer and stopword list
        stemmer, stopword_list = self.get_stemmer_and_stopwords(language)

        # Preprocess text with the detected stemmer and stopwords
        processed_lyrics = self.preprocess_text(lyrics, stemmer, stopword_list)

        # Tokenize and preprocess
        tokens = word_tokenize(processed_lyrics)
        unique_tokens = set(tokens)
    
        # Compute basic metrics
        word_count = len(tokens)
        unique_word_count = len(unique_tokens)
        lexical_richness = unique_word_count / word_count if word_count > 0 else 0
    
        # Semantic depth
        synset_depths = [max(len(ss.hypernym_paths()[0]) for ss in wn.synsets(word)) 
                         for word in tokens if wn.synsets(word)]
        semantic_depth = sum(synset_depths) / len(synset_depths) if synset_depths else 0
    
        # Syntactic complexity
        sentences = nltk.sent_tokenize(lyrics)
        syntactic_complexity = sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
    
        # Rhyme density
        lines = lyrics.split(".\n")
        rhyme_pairs = 0
        for line in lines:
            words = line.split()
            if len(words) > 1 and pronouncing.rhymes(words[-1]):
                rhymes = [w for w in pronouncing.rhymes(words[-1]) if w in words]
                rhyme_pairs += len(rhymes)
        rhyme_density = rhyme_pairs / word_count if word_count > 0 else 0
    
        # Sentiment variability
        scores = [self.sid.polarity_scores(line)['compound'] for line in lines]
        sentiment_variability = statistics.stdev(scores) if len(scores) > 1 else 0
    
        # Vader compound
        vader_compound = self.sid.polarity_scores(lyrics)['compound']

        # Sentiment polarity and subjectivity
        sentiment_polarity = TextBlob(lyrics).sentiment.polarity
        sentiment_subjectivity = TextBlob(lyrics).sentiment.subjectivity
    
        # Linguistic uniqueness
        clean_tokens = [word.strip(".,") for word in tokens]
        rare_words = [word for word in clean_tokens if word_frequency(word, 'en') < 1e-6]
        linguistic_uniqueness = len(rare_words) / len(clean_tokens) if clean_tokens else 0

        # Type-Token Ratio (TTR)
        ttr = len(unique_tokens) / len(tokens) if tokens else 0

        # Repetition count
        word_freq = Counter(tokens)
        repetition_count = sum(count - 1 for count in word_freq.values() if count > 1)
    
        # Readability metrics
        readability = {
            "flesch_reading_ease": textstat.flesch_reading_ease(lyrics),
            "gunning_fog": textstat.gunning_fog(lyrics),
            "dale_chall": textstat.dale_chall_readability_score(lyrics)
        }
    
        # Part of speech ratios
        tagged = pos_tag(tokens)
        noun_ratio = sum(1 for word, tag in tagged if tag.startswith('NN')) / word_count if word_count > 0 else 0
        verb_ratio = sum(1 for word, tag in tagged if tag.startswith('VB')) / word_count if word_count > 0 else 0

        return {
            "word_count": word_count,
            "unique_word_count": unique_word_count,
            "lexical_richness": lexical_richness,
            "semantic_depth": semantic_depth,
            "syntactic_complexity": syntactic_complexity,
            "rhyme_density": rhyme_density,
            "sentiment_variability": sentiment_variability,
            "linguistic_uniqueness": linguistic_uniqueness,
            "flesch_reading_ease": readability["flesch_reading_ease"],
            "gunning_fog": readability["gunning_fog"],
            "dale_chall": readability["dale_chall"],
            "vader_compound": vader_compound,
            "noun_ratio": noun_ratio,
            "verb_ratio": verb_ratio,
            "language": language,
            "sentiment_polarity": sentiment_polarity,
            "sentiment_subjectivity": sentiment_subjectivity,
            "type_token_ratio": ttr,
            "repetition_count": repetition_count,
            "preprocessed_lyrics": processed_lyrics
        }

    def add_features(self, df: pd.DataFrame, text_column: str = 'lyrics') -> pd.DataFrame:
        """Adds extracted text features and TF-IDF features to a DataFrame."""
        # Extract features for all rows
        all_features = [self.extract_features(row[text_column]) for _, row in df.iterrows()]
        
        # Create a DataFrame directly from the list of dictionaries
        features_df = pd.DataFrame(all_features)
        
        # Align and overwrite columns in the original DataFrame
        df = df.drop(columns=features_df.columns.intersection(df.columns), errors='ignore')
        
        return pd.concat([df.reset_index(drop=True), features_df], axis=1)



class TfidfFeatureExtractor(FeatureExtractor):
    def __init__(self, n_pca_components: int = 100) -> None:
        """Initializes the TF-IDF Feature Extractor with PCA."""
        self.tfidf = TfidfVectorizer()
        self.pca = PCA(n_components=n_pca_components)

    def extract_features(self, data: pd.Series) -> pd.DataFrame:
        """Extracts TF-IDF features and applies PCA for dimensionality reduction.

        Args:
            data (pd.Series): The text data to process.

        Returns:
            pd.DataFrame: The PCA-reduced TF-IDF features as a DataFrame.
        """
        # TF-IDF Transformation
        tfidf_matrix = self.tfidf.fit_transform(data).toarray()

        # Apply PCA for dimensionality reduction
        tfidf_pca = self.pca.fit_transform(tfidf_matrix)

        # Create a DataFrame for PCA-reduced TF-IDF features
        pca_columns = [f'tfidf_{i}' for i in range(tfidf_pca.shape[1])]
        return pd.DataFrame(tfidf_pca, columns=pca_columns)

    def add_features(self, df: pd.DataFrame, text_column: str = 'lyrics') -> pd.DataFrame:
        """Adds TF-IDF PCA-reduced features to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing text data.
            text_column (str, optional): Column containing the text data. Defaults to 'text'.

        Returns:
            pd.DataFrame: The original DataFrame with added TF-IDF PCA features.
        """
        # Extract TF-IDF features
        tfidf_features = self.extract_features(df[text_column])

        # Align and overwrite columns in the original DataFrame
        df = df.drop([c for c in df.columns if c.startswith('tfidf_') ], axis=1)
        
        # Combine with the original DataFrame
        return pd.concat([df.reset_index(drop=True), tfidf_features], axis=1)
