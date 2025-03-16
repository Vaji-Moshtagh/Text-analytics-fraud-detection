"""Topic modeling tools for text analysis."""

import gensim
from gensim import corpora
import pandas as pd
import numpy as np

class TopicAnalyzer:
    """Topic modeling for text analysis."""
    
    def __init__(self):
        """Initialize the topic analyzer."""
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        
    def prepare_data(self, text_data):
        """Prepare text data for topic modeling.
        
        Args:
            text_data (list): List of tokenized documents (list of lists).
            
        Returns:
            TopicAnalyzer: Self for method chaining.
        """
        # Create dictionary
        self.dictionary = corpora.Dictionary(text_data)
        
        # Filter extreme values
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)
        
        # Create corpus (bag of words)
        self.corpus = [self.dictionary.doc2bow(text) for text in text_data]
        
        return self
        
    def build_model(self, num_topics=5, passes=5, alpha='auto', eta='auto'):
        """Build LDA topic model.
        
        Args:
            num_topics (int): Number of topics to extract.
            passes (int): Number of passes through the corpus during training.
            alpha (str or float): Prior document-topic distribution.
            eta (str or float): Prior topic-word distribution.
            
        Returns:
            TopicAnalyzer: Self for method chaining.
        """
        if not self.corpus or not self.dictionary:
            raise ValueError("Corpus and dictionary not prepared. Run prepare_data first.")
            
        # Create LDA model
        self.lda_model = gensim.models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            alpha=alpha,
            eta=eta
        )
        
        return self
        
    def get_topic_details(self):
        """Get topic details for each document.
        
        Returns:
            pandas.DataFrame: DataFrame with topic details for each document.
        """
        if not self.lda_model:
            raise ValueError("LDA model not built. Run build_model first.")
            
        # Create empty dataframe
        topics_df = pd.DataFrame()
        
        # Get dominant topic for each document
        for i, doc in enumerate(self.corpus):
            # Get topic distribution for document
            topic_dist = self.lda_model.get_document_topics(doc)
            
            # Find dominant topic (if no topics found, use -1)
            if topic_dist:
                dominant_topic = sorted(topic_dist, key=lambda x: x[1], reverse=True)[0]
                topic_id, topic_prob = dominant_topic
            else:
                topic_id, topic_prob = -1, 0
            
            # Get keywords for topic
            if topic_id != -1:
                topic_keywords = ", ".join([word for word, prob in 
                                         self.lda_model.show_topic(topic_id, topn=6)])
            else:
                topic_keywords = "No dominant topic"
            
            # Add to dataframe
            topics_df = pd.concat([topics_df, pd.DataFrame({
                'Document_Id': i,
                'Dominant_Topic': float(topic_id),
                'Topic_Prob': topic_prob,
                'Topic_Keywords': topic_keywords
            }, index=[0])], ignore_index=True)
            
        return topics_df
        
    def flag_topic(self, topics_df, suspicious_topic_id):
        """Flag documents associated with a specific topic.
        
        Args:
            topics_df (pandas.DataFrame): DataFrame with topic details.
            suspicious_topic_id (int or float): ID of the suspicious topic.
            
        Returns:
            pandas.DataFrame: DataFrame with added 'Flag' column.
        """
        result_df = topics_df.copy()
        result_df['Flag'] = np.where(
            (result_df['Dominant_Topic'] == float(suspicious_topic_id)),
            1,  # Flag suspicious content
            0   # Mark as normal
        )
        
        return result_df
        
    def get_topic_distribution(self):
        """Get the distribution of topics across the corpus.
        
        Returns:
            pandas.DataFrame: DataFrame with topic distribution.
        """
        if not self.lda_model:
            raise ValueError("LDA model not built. Run build_model first.")
            
        # Get topic details
        topic_details = self.get_topic_details()
        
        # Calculate distribution
        topic_dist = topic_details['Dominant_Topic'].value_counts().reset_index()
        topic_dist.columns = ['Topic', 'Count']
        topic_dist['Percentage'] = topic_dist['Count'] / len(topic_details) * 100
        
        return topic_dist
