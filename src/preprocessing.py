import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase & tokenize
    tokens = [t for t in tokens if t not in string.punctuation]  # Remove punctuation
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]  # Lemmatization & stopword removal
    return " ".join(tokens)

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df["cleaned_resume"] = df["Resume"].apply(preprocess_text)
    return df

if __name__ == "__main__":
    df = load_and_preprocess("../data/resume_dataset.csv")
    print(df.head())
