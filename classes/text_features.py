import warnings
import pandas as pd
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re
from nltk import pos_tag
from textstat import textstat
from classes.feature_extractor import FeatureExtractor

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')



class TextFeatureExtractor(FeatureExtractor):
    """Class to handle the extraction of text features from song lyrics."""

    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.d = cmudict.dict()
        self.tfidf = TfidfVectorizer(max_features=500, stop_words="english")

    def syllable_count(self, word):
        if word.lower() in self.d:
            return len([y for y in self.d[word.lower()][0] if y[-1].isdigit()])
        else:
            return len(re.findall(r'[aeiouy]+', word.lower())) or 1

    def extract_features(self, lyrics):
        """Extract features from a single text input."""
        word_count = len(lyrics.split())
        unique_word_count = len(set(lyrics.split()))
        avg_word_length = sum(len(word) for word in lyrics.split()) / word_count if word_count > 0 else 0
        syllable_count = sum([self.syllable_count(word) for word in lyrics.split()])
        sentiment_polarity = TextBlob(lyrics).sentiment.polarity
        sentiment_subjectivity = TextBlob(lyrics).sentiment.subjectivity
        readability_score = textstat.flesch_kincaid_grade(lyrics)
        
        tokens = word_tokenize(lyrics)
        tagged = pos_tag(tokens)
        noun_count = sum(1 for word, tag in tagged if tag.startswith('NN'))
        verb_count = sum(1 for word, tag in tagged if tag.startswith('VB'))

        vader_compound = self.sid.polarity_scores(lyrics)['compound']

        words = lyrics.split()
        word_freq = Counter(words)
        repetition_count = len([word for word, count in word_freq.items() if count > 1])

        average_syllables_per_word = syllable_count / word_count if word_count > 0 else 0

        return [
            word_count, unique_word_count, avg_word_length, syllable_count,
            sentiment_polarity, sentiment_subjectivity, readability_score,
            noun_count, verb_count, vader_compound, repetition_count,
            average_syllables_per_word
        ]

    def add_features(self, df, text_column='lyrics'):
        """
        Adds text features to the given DataFrame.
    
        Parameters:
        - df: DataFrame containing lyrics.
        - text_column: Name of the column containing lyrics.
    
        Returns:
        - DataFrame: The original DataFrame with added text features.
        """
        feature_columns = [
            'word_count', 'unique_word_count', 'avg_word_length', 'syllable_count',
            'sentiment_polarity', 'sentiment_subjectivity', 'readability_score',
            'noun_count', 'verb_count', 'vader_compound', 'repetition_count',
            'average_syllables_per_word'
        ]

        all_features = []
        for _, row in df.iterrows():
            lyrics = row[text_column]
            features = self.extract_features(lyrics)
            all_features.append(features)
        
        # Create DataFrame for the features
        features_df = pd.DataFrame(all_features, columns=feature_columns)

        # TF-IDF Features
        tfidf_features = pd.DataFrame(self.tfidf.fit_transform(df[text_column]).toarray(), columns=self.tfidf.get_feature_names_out()).drop('lyrics', axis=1)
        
        # Combine all features
        df_with_features = pd.concat([df.reset_index(drop=True), features_df, tfidf_features], axis=1)
        
        return df_with_features