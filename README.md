# Tweet Sentiment Analyzer

## Overview

The Tweet Sentiment Analyzer is a Python-based project that processes tweets to determine the sentiment expressed toward brands or products. It leverages natural language processing (NLP) techniques with NLTK for text preprocessing and scikit-learn for building classification models. The project trains several machine learning pipelines—Naive Bayes, Support Vector Machine (SVM), and Logistic Regression—to classify sentiment as either positive or negative. In addition, it features various visualizations such as confusion matrices, performance comparison charts (bar and radar charts), and word clouds to help understand model performance and textual patterns.

## Project Structure

- **Data Loading:**  
  The project reads training and test datasets from CSV files. The training dataset contains three columns:

    - `tweet_text`: The raw tweet.
    - `emotion_in_tweet_is_directed_at`: The brand or product targeted by the tweet.
    - `is_there_an_emotion_directed_at_a_brand_or_product`: The sentiment label (e.g., "Positive emotion" or "Negative emotion").

- **Preprocessing:**  
  The code preprocesses tweets by:

    - Converting text to lowercase.
    - Removing URLs, mentions, and hashtags.
    - Removing unnecessary punctuation while preserving emoticons.
    - Tokenizing text using NLTK.
    - Removing common stopwords (with a custom set that preserves critical negative words).
    - Lemmatizing tokens using NLTK's WordNetLemmatizer.

- **Model Pipelines:**  
  Three pipelines are built using scikit-learn:

    1. **Naive Bayes Pipeline:** Uses TF-IDF vectorization and a Multinomial Naive Bayes classifier.
    2. **SVM Pipeline:** Uses TF-IDF vectorization and a Linear Support Vector Classifier.
    3. **Logistic Regression Pipeline:** Uses TF-IDF vectorization and a Logistic Regression classifier.

- **Evaluation and Visualization:**

    - The project computes key performance metrics: Accuracy, Precision, Recall, and F1 Score.
    - It generates confusion matrices, bar charts, and radar charts to compare model performance.
    - Word clouds are created to visualize common words in tweets for both positive and negative sentiments.

- **Interactive Analysis:**  
  After training and evaluation, the best-performing model is selected based on the F1 score. An interactive interface allows users to input new tweets for real-time sentiment prediction.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/tweet-sentiment-analyzer.git
    cd tweet-sentiment-analyzer
    Install Required Libraries:
    ```

2. Ensure you have Python 3 installed. Then run:

```bash
pip install -r requirements.txt
```

- The requirements.txt should include:

    - pandas
    - numpy
    - nltk
    - matplotlib
    - seaborn
    - scikit-learn
    - wordcloud

3. Download NLTK Data (if not auto-downloaded):

Open a Python shell and execute:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

```

4. Datasets:

- Place your CSV files in a folder named dataset:

    Dataset - Train.csv

    Dataset - Test.csv

5. Running the Project

- To run the entire pipeline—including data preprocessing, model training, evaluation, visualization, and interactive analysis—execute:

```bash
python sentimentAnaylzer.py

```

# In-Depth Explanation

## Preprocessing

- The preprocess_text function is responsible for cleaning and normalizing the tweet text. It:

    - Converts the text to lowercase.

    - Removes URLs, mentions, and hashtags.

    - Filters out punctuation while keeping emoticons.

    - Tokenizes the text using NLTK.

    - Removes stopwords while preserving key negative words (e.g., "not", "never").

    - Lemmatizes the tokens using NLTK’s WordNetLemmatizer.

- This step reduces noise in the data and ensures that words are represented in their base form, improving model performance.

## Model Pipelines

- Each model pipeline consists of:

    - TF-IDF Vectorizer: Transforms text into numerical feature vectors.

    - Classifier: One of the classifiers (Naive Bayes, SVM, or Logistic Regression) that learns to map these features to sentiment labels.

    - Pipelines streamline the process by chaining these steps together.

## Evaluation

- The evaluate_model function calculates key metrics (accuracy, precision, recall, F1 score) and generates a confusion matrix for each model. This helps in:

    - Understanding overall performance.

    - Identifying which sentiment classes (positive/negative) the model struggles with.

## Visualization

- The project includes several visualizations:

    - Bar Chart and Radar Chart: Compare performance metrics across models.

    - Word Clouds: Show the most frequent words in tweets with positive and negative sentiment, which can provide insight into the language patterns used.

## Interactive Analysis

- After evaluating the models, the best-performing one (based on F1 score) is selected for interactive analysis. Users can enter new tweets, and the model predicts the sentiment in real time, along with an optional confidence score.

# Conclusion

This project provides a comprehensive framework for tweet sentiment analysis. It integrates data preprocessing, machine learning model building, evaluation, and visualization into a single pipeline. The interactive component allows users to test the model on new data, making it a robust starting point for further exploration into sentiment analysis and NLP.

Feel free to modify and extend the project to suit your specific needs!
