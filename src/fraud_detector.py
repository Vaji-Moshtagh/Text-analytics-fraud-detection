"""Tools for detecting suspicious text in financial communications."""

import pandas as pd
import numpy as np

class FraudDetector:
    """Tools for detecting suspicious text in financial communications."""
    
    def __init__(self):
        """Initialize the fraud detector."""
        self.fraud_terms = []
        
    def set_fraud_dictionary(self, terms_list):
        """Set suspicious terms to search for.
        
        Args:
            terms_list (list): List of suspicious terms or phrases.
        """
        self.fraud_terms = terms_list
        
    def flag_suspicious_content(self, df, text_column):
        """Flag content containing suspicious terms.
        
        Args:
            df (pandas.DataFrame): DataFrame containing text data.
            text_column (str): Column name containing text to analyze.
            
        Returns:
            pandas.DataFrame: DataFrame with added 'flag' column.
        """
        if not self.fraud_terms:
            raise ValueError("Fraud terms dictionary is empty. Use set_fraud_dictionary first.")
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Create search pattern by joining terms with OR operator
        pattern = '|'.join(self.fraud_terms)
        
        # Handle missing values
        result_df[text_column] = result_df[text_column].fillna('')
        
        # Create flag column
        result_df['flag'] = np.where(
            result_df[text_column].str.contains(pattern, case=False, regex=True),
            1,  # Flag suspicious content
            0   # Mark as normal
        )
        
        return result_df
        
    def get_suspicious_content(self, df):
        """Return only flagged content.
        
        Args:
            df (pandas.DataFrame): DataFrame with 'flag' column.
            
        Returns:
            pandas.DataFrame: DataFrame containing only flagged rows.
        """
        if 'flag' not in df.columns:
            raise ValueError("DataFrame has no 'flag' column. Run flag_suspicious_content first.")
            
        return df[df['flag'] == 1]
        
    def find_matching_terms(self, text, return_all=False):
        """Find which terms from the fraud dictionary match the text.
        
        Args:
            text (str): Text to analyze.
            return_all (bool): Whether to return all matches or just the first.
            
        Returns:
            list or str: Matching terms or first matching term.
        """
        if not self.fraud_terms:
            raise ValueError("Fraud terms dictionary is empty. Use set_fraud_dictionary first.")
            
        matches = []
        for term in self.fraud_terms:
            if term.lower() in text.lower():
                matches.append(term)
                if not return_all:
                    break
                    
        if not matches:
            return None
        
        return matches if return_all else matches[0]
