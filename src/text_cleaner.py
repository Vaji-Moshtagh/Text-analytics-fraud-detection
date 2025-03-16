"""Text cleaning utilities for NLP tasks."""

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

class TextCleaner:
    """Text cleaning utilities for NLP tasks."""
    
    def __init__(self, custom_stopwords=None):
        """Initialize with default and custom stopwords.
        
        Args:
            custom_stopwords (list, optional): Additional stopwords to include.
        """
        # Ensure nltk resources are available
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            WordNetLemmatizer().lemmatize('test')
        except LookupError:
            nltk.download('wordnet')
            
        self.stop = set(stopwords.words('english'))
        self.exclude = set(string.punctuation)
        self.lemma = WordNetLemmatizer()
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stop.update(custom_stopwords)
    
    def clean_text(self, text):
        """Clean text by removing stopwords, punctuation, and lemmatizing.
        
        Args:
            text (str): Text to clean.
            
        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            return ""
            
        # Strip whitespace
        text = text.strip()
        
        # Remove stopwords and digits
        stop_free = " ".join([word for word in text.lower().split() 
                             if (word not in self.stop) and (not word.isdigit())])
        
        # Remove punctuation
        punc_free = " ".join(word for word in stop_free if word not in self.exclude)
        
        # Lemmatize words
        normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
        
        return normalized
        
    def tokenize(self, text):
        """Clean text and return tokens.
        
        Args:
            text (str): Text to clean and tokenize.
            
        Returns:
            list: List of cleaned tokens.
        """
        cleaned_text = self.clean_text(text)
        return cleaned_text.split()
