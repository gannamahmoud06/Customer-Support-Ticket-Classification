import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from collections import Counter
from nltk.util import ngrams

def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

setup_nltk()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def drop_unimportant_columns(df):
    cols_to_drop = [
        'Ticket ID', 'Customer Name', 'Customer Email', 'Customer Age',
        'Customer Gender', 'Customer Satisfaction Rating', 'Time to Resolution',
        'Resolution', 'First Response Time'
    ]
    
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    return df

def map_labels(df):

    label_map = {
        'Technical issue': 'Technical', 'Technical Issue': 'Technical',
        'Billing inquiry': 'Billing', 'Billing Inquiry': 'Billing',
        'Account access': 'Account', 'Account Access': 'Account',
        'Product inquiry': 'Other', 'Product Inquiry': 'Other',
        'Refund request': 'Billing', 'Cancellation request': 'Billing',
        'Shipping issue': 'Other', 'Other': 'Other'
    }
    if 'Ticket Type' in df.columns:
        df['Label'] = df['Ticket Type'].map(label_map)
        df = df.dropna(subset=['Label'])
    return df

def clean_text(text):
    """Clean Text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    
    tokens = [lemmatizer.lemmatize(w) for w in tokens 
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def prepare_input_column(df):

    df['input'] = df['Ticket Subject'].astype(str) + ' ' + df['Ticket Description'].astype(str)
    df['text_clean'] = df['input'].apply(clean_text)
    return df

# Visualizations Functios

def plot_token_length(df):
    df['token_length'] = df['input'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['token_length'], bins=50, kde=True, color='teal')
    plt.title('Distribution of Token Lengths')
    plt.show()

def plot_top_words(df, top_n=20):
    words = " ".join(df['text_clean']).split()
    fdist = FreqDist(words)
    plt.figure(figsize=(12, 6))
    fdist.plot(top_n, title=f'Top {top_n} Most Frequent Words')
    plt.show()

def plot_bigrams(df, top_n=15):
    words = " ".join(df['text_clean']).split()
    bigrams = list(ngrams(words, 2))
    bigram_counts = Counter(bigrams)
    top_bg = bigram_counts.most_common(top_n)
    labels, counts = zip(*[("-".join(bg), c) for bg, c in top_bg])
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(counts), y=list(labels), palette='magma')
    plt.title(f'Top {top_n} Bigrams Analysis')
    plt.show()