import warnings
import pandas as pd
import nltk
from nltk.tokenize import casual_tokenize
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re
from nltk import pos_tag
from textstat import textstat
from langdetect import detect, DetectorFactory
from classes.feature_extractor import FeatureExtractor
from typing import List
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import string

# Set seed for reproducibility in langdetect
DetectorFactory.seed = 0

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('cmudict', quiet=True)

class TextFeatureExtractor(FeatureExtractor):
    def __init__(self, n_pca_components: int = 0.95) -> None:
        """Initializes the TextFeatureExtractor with sentiment and TF-IDF analyzers."""
        self.sid = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer()
        self.stemmer = SnowballStemmer("english")
        self.pca = PCA(n_components=n_pca_components)
        self.d= cmudict.dict()
        
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text by removing stopwords, punctuation, numbers, 
        and applying stemming after tokenization.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: Processed text ready for analysis.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        
        # Tokenize text
        words = casual_tokenize(text)
        
        # Remove stopwords
        filtered_words = [word for word in words if word not in set(stopwords.words("english"))]
        
        # Apply stemming
        tokens = [SnowballStemmer("english").stem(word) for word in filtered_words]
        
        return " ".join(tokens)

    def syllable_count(self, word: str) -> int:
        """Calculates the syllable count for a given word.

        Args:
            word (str): The word to calculate syllables for.

        Returns:
            int: The syllable count of the word.
        """
        if word.lower() in self.d:
            return len([y for y in self.d[word.lower()][0] if y[-1].isdigit()])
        else:
            return len(re.findall(r'[aeiouy]+', word.lower())) or 1

    def detect_language(self, text: str) -> str:
        """Detects the language of the given text.

        Args:
            text (str): The text to detect the language of.

        Returns:
            str: The detected language code (e.g., 'en' for English).
        """
        try:
            return detect(text)
        except:
            return "unknown"

    def extract_features(self, lyrics: str) -> List[float]:
        """Extracts various text features from a single lyrics string.

        Args:
            lyrics (str): The song lyrics.

        Returns:
            List[float]: A list of extracted text features.
        """
        word_count = len(lyrics.split())
        unique_word_count = len(set(lyrics.split()))
        avg_word_length = sum(len(word) for word in lyrics.split()) / word_count if word_count > 0 else 0
        syllable_count = sum([self.syllable_count(word) for word in lyrics.split()])
        sentiment_polarity = TextBlob(lyrics).sentiment.polarity
        sentiment_subjectivity = TextBlob(lyrics).sentiment.subjectivity
        readability_score = textstat.flesch_kincaid_grade(lyrics)
        
        tokens = word_tokenize(lyrics)
        tagged = pos_tag(tokens)
        noun_ratio = sum(1 for word, tag in tagged if tag.startswith('NN')) / word_count
        verb_ratio = sum(1 for word, tag in tagged if tag.startswith('VB')) / word_count


        vader_compound = self.sid.polarity_scores(lyrics)['compound']

        words = lyrics.split()
        word_freq = Counter(words)
        repetition_count = len([word for word, count in word_freq.items() if count > 1])

        average_syllables_per_word = syllable_count / word_count if word_count > 0 else 0

        # Detect language
        language = self.detect_language(lyrics)

        return [
            word_count, unique_word_count, avg_word_length, syllable_count,
            sentiment_polarity, sentiment_subjectivity, readability_score,
            noun_ratio, verb_ratio, vader_compound, repetition_count,
            average_syllables_per_word, language
        ]

    def add_features(self, df: pd.DataFrame, text_column: str = 'lyrics') -> pd.DataFrame:
        """Adds extracted text features and TF-IDF features to a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing lyrics.
            text_column (str, optional): Name of the column containing lyrics. Defaults to 'lyrics'.

        Returns:
            pd.DataFrame: The original DataFrame with additional text features.
        """
        feature_columns = [
            'word_count', 'unique_word_count', 'avg_word_length', 'syllable_count',
            'sentiment_polarity', 'sentiment_subjectivity', 'readability_score',
            'noun_ratio', 'verb_ratio', 'vader_compound', 'repetition_count',
            'average_syllables_per_word', 'language'
        ]

        all_features = []
        for _, row in df.iterrows():
            lyrics = row[text_column]
            features = self.extract_features(lyrics)
            all_features.append(features)
        
        # Create DataFrame for the features
        features_df = pd.DataFrame(all_features, columns=feature_columns)

        # TF-IDF Transformation
        tfidf_matrix = self.tfidf.fit_transform(df[text_column]).toarray()

        tfidf_pca = self.pca.fit_transform(tfidf_matrix)

        # Create a DataFrame for PCA-reduced TF-IDF features
        pca_columns = [f'tfidf_{i}' for i in range(tfidf_pca.shape[1])]
        tfidf_pca_df = pd.DataFrame(tfidf_pca, columns=pca_columns)

        # Combine all features
        df_with_features = pd.concat([df.reset_index(drop=True), features_df, tfidf_pca_df], axis=1)
        
        return df_with_features
