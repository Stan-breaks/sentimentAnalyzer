import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV

# Dataset description:
# "Users assessed tweets related to various brands and products, providing evaluations on whether the sentiment conveyed was positive, negative, or neutral.
# Additionally, if the tweet conveyed any sentiment, contributors identified the specific brand or product targeted by that emotion."

# Dataset columns:
# "tweet_text"
# "emotion_in_tweet_is_directed_at"
# "is_there_an_emotion_directed_at_a_brand_or_product"

# Set display options for better visualization
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)

try:
    # Load datasets
    test_df = pd.read_csv("dataset/Dataset - Test.csv")
    train_df = pd.read_csv("dataset/Dataset - Train.csv")

    def preprocess_text(text):
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove mentions and hashtags (hashtags are replaced with a space to preserve words)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", " ", text)

        # Remove punctuation but keep emoticons like :) :(
        text = re.sub(r"[^\w\s:)(;]", "", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize text using nltk
        words = nltk.word_tokenize(text)

        # Remove stopwords while preserving key negative words
        stop_words = set(stopwords.words("english")) - {
            "no",
            "not",
            "nor",
            "neither",
            "never",
            "none",
            "n't",
            "ain't",
        }
        words = [word for word in words if word not in stop_words]

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return " ".join(words)

    # Apply preprocessing to the training dataset using 'tweet_text'
    print("Preprocessing text data for training set...")
    train_df["processed_text"] = train_df["tweet_text"].apply(preprocess_text)

    # Determine the correct column name for text in the testing dataset
    test_text_column = "tweet_text" if "tweet_text" in test_df.columns else "Tweet"

    print("Preprocessing text data for testing set...")
    test_df["processed_text"] = test_df[test_text_column].apply(preprocess_text)

    # Show sample processed texts from training data
    print("\nSample processed texts:")
    for i in range(3):
        print(f"Original: {train_df['tweet_text'].iloc[i]}")
        print(f"Processed: {train_df['processed_text'].iloc[i]}")
        print()
    print("Preprocessing complete.")

    # Define X and y for training
    X_train = train_df["processed_text"]
    y_train = train_df[
        "is_there_an_emotion_directed_at_a_brand_or_product"
    ]  # Adjust column name if different

    # Define X and y for testing
    X_test = test_df["processed_text"]
    y_test = test_df[
        "is_there_an_emotion_directed_at_a_brand_or_product"
    ]  # Adjust column name if different

    # 1. Naive Bayes Pipeline
    nb_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("classifier", MultinomialNB()),
        ]
    )

    # 2. Support Vector Machine Pipeline
    svm_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("classifier", LinearSVC(C=1, max_iter=10000)),
        ]
    )

    # 3. Logistic Regression Pipeline
    lr_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(C=1, max_iter=10000)),
        ]
    )

    # Train models
    print("Training Naive Bayes model...")
    nb_pipeline.fit(X_train, y_train)

    print("Training SVM model...")
    svm_pipeline.fit(X_train, y_train)

    print("Training Logistic Regression model...")
    lr_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred_nb = nb_pipeline.predict(X_test)
    y_pred_svm = svm_pipeline.predict(X_test)
    y_pred_lr = lr_pipeline.predict(X_test)

except Exception as e:
    print(f"Error: {str(e)}")
