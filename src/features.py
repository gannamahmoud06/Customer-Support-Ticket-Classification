import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class FeatureExtractor:
    def __init__(self):
        self.tfidf = None
        self.le = None

    def load_pretrained_tools(self, tfidf_path, le_path):
        
        self.tfidf = joblib.load(tfidf_path)
        self.le = joblib.load(le_path)
        print(" Pre-trained tools loaded successfully!")

    def transform_text(self, text_list):

        if self.tfidf is None:
            raise Exception("Please load the tools first using load_pretrained_tools()")
        
        return self.tfidf.transform(text_list)

    def inverse_label(self, prediction):
        return self.le.inverse_transform(prediction)