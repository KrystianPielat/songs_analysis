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
from empath import Empath
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

# Set seed for reproducibility in langdetect
DetectorFactory.seed = 0

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextFeatureExtractor:
    def __init__(self) -> None:
        """Initializes the TextFeatureExtractor with sentiment and Empath analyzers."""
        self.sid = SentimentIntensityAnalyzer()
        self.empath_analyzer = Empath()

    @staticmethod
    def preprocess_text(text: str, stemmer: Optional[SnowballStemmer] = None, stopword_list: Optional[set] = None, title: Optional[str] = None) -> str:
        """
        Preprocess text by:
        - Removing square bracket content like [Verse], [Chorus].
        - Removing "letra de" prefix and title words in the first 6 words of lyrics.
        - Removing stopwords and punctuation.
        - Applying stemming.
        """
        if not isinstance(text, str):
            return ""
    
        # Lowercase the text
        text = text.lower()
    
        # Remove square bracket content like [Chorus], [Verse 1]
        text = re.sub(r"\[.*?\]", "", text)
    
        # Remove "letra de" prefix and clean title-like words
        if text.startswith("letra de"):
            text = text[len("letra de"):].strip()
            if title:
                title_words = set(title.lower().split())
                lyrics_words = text.split()
                text = " ".join([word for i, word in enumerate(lyrics_words) if i >= 6 or word not in title_words])
    
        # Remove words like "verse", "chorus"
        text = re.sub(r"\b(verse|chorus|bridge|outro|intro)\b", "", text)
    
        # Remove numbers and punctuation
        text = re.sub(r"\d+", "", text.translate(str.maketrans("", "", string.punctuation)))
    
        # Tokenize and remove stopwords
        tokens = casual_tokenize(text)
        if stopword_list:
            tokens = [token for token in tokens if token not in stopword_list]
    
        # Apply stemming if a stemmer is provided
        if stemmer:
            tokens = [stemmer.stem(token) for token in tokens]
    
        return " ".join(tokens)


    def extract_empath_features(self, text: str) -> dict:
        """Extracts Empath features for a given text."""
        empath_scores = self.empath_analyzer.analyze(text, normalize=True)
        return {f'empath_{k}': v for k, v in empath_scores.items()}

    def extract_features(self, lyrics: str) -> dict:
        """Extracts various text features from a single lyrics string."""
        # Language detection
        try:
            language = Language.make(detect(lyrics)).display_name().lower()
        except Exception:
            language = "unknown"

        # Stopwords and stemmer
        stemmer, stopword_list = None, None
        if language in stopwords.fileids():
            stemmer = SnowballStemmer(language)
            stopword_list = set(stopwords.words(language))

        # Preprocess text
        processed_lyrics = self.preprocess_text(lyrics, stemmer, stopword_list)
        tokens = word_tokenize(processed_lyrics)
        unique_tokens = set(tokens)

        # Basic Metrics
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

        # Sentiment Variability
        scores = [self.sid.polarity_scores(line)['compound'] for line in lines if line]
        sentiment_variability = statistics.stdev(scores) if len(scores) > 1 else 0

        # Vader and TextBlob Sentiment
        vader_compound = self.sid.polarity_scores(lyrics)['compound']
        sentiment_polarity = TextBlob(lyrics).sentiment.polarity
        sentiment_subjectivity = TextBlob(lyrics).sentiment.subjectivity

        # Linguistic uniqueness
        clean_tokens = [word.strip(".,") for word in tokens]
        rare_words = [word for word in clean_tokens if word_frequency(word, 'en') < 1e-6]
        linguistic_uniqueness = len(rare_words) / len(clean_tokens) if clean_tokens else 0

        # Type-Token Ratio (TTR) and Repetition
        ttr = len(unique_tokens) / len(tokens) if tokens else 0
        word_freq = Counter(tokens)
        repetition_count = sum(count - 1 for count in word_freq.values() if count > 1)

        # Readability metrics
        readability = {
            "flesch_reading_ease": textstat.flesch_reading_ease(lyrics),
            "gunning_fog": textstat.gunning_fog(lyrics),
            "dale_chall": textstat.dale_chall_readability_score(lyrics)
        }

        # Part-of-Speech Ratios
        tagged = pos_tag(tokens)
        noun_ratio = sum(1 for _, tag in tagged if tag.startswith('NN')) / word_count if word_count > 0 else 0
        verb_ratio = sum(1 for _, tag in tagged if tag.startswith('VB')) / word_count if word_count > 0 else 0

        # Empath features
        empath_features = self.extract_empath_features(lyrics)

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
            "preprocessed_lyrics": processed_lyrics,
            **empath_features
        }


    @staticmethod
    def _process_batch(args):
        """Processes a batch of rows for feature extraction."""
        batch, text_column = args
        extractor = TextFeatureExtractor()
        results = []
        for _, row in batch.iterrows():
            try:
                features = extractor.extract_features(row[text_column])
                results.append(features)
            except Exception as e:
                logging.error(f"Error processing row: {e}")
                placeholder = {k: None for k in extractor.extract_features("").keys()}
                results.append(placeholder)
        return results

    def add_features(self, df: pd.DataFrame, text_column: str = 'lyrics', batch_size: int = 100) -> pd.DataFrame:
        """Adds extracted text features to a DataFrame in parallel batches."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        # Split DataFrame into batches
        batches = [(df.iloc[i:i + batch_size], text_column) for i in range(0, len(df), batch_size)]
        all_features = []

        # Process batches in parallel
        with Pool(cpu_count()) as pool:
            with tqdm(total=len(df), desc="Processing Features") as pbar:
                for batch_result in pool.imap(TextFeatureExtractor._process_batch, batches):
                    all_features.extend(batch_result)
                    pbar.update(len(batch_result))

        # Combine results into DataFrame
        features_df = pd.DataFrame(all_features)
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
