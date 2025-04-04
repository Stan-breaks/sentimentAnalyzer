import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
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
    classification_report,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from wordcloud import WordCloud
import warnings

warnings.filterwarnings("ignore")

# Download required NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

# Set display options for better visualization
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)


# Function to preprocess text
def preprocess_text(text):
    """Clean and normalize text data for sentiment analysis."""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove mentions and hashtags (replace hashtags with a space)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", " ", text)

    # Remove punctuation but keep emoticons like :) :(
    text = re.sub(r"[^\w\s:)(;]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize using nltk
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


# Function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance with metrics and confusion matrix visualization."""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Print metrics
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Function to create radar chart
def create_radar_chart(results_df, metrics):
    """Create radar chart to compare model performance across metrics."""
    # Number of variables
    categories = metrics
    N = len(categories)

    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], size=10)
    plt.ylim(0, 1)

    # Plot each model
    colors = ["blue", "red", "green", "purple"]
    for i, row in results_df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]  # Close the loop

        # Plot values
        ax.plot(
            angles,
            values,
            linewidth=2,
            linestyle="solid",
            label=row["model"],
            color=colors[i % len(colors)],
        )
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Performance Comparison", size=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Function to create word clouds
def plot_word_clouds(df, text_column, sentiment_column, target_entity=None):
    """Create word clouds showing common words in positive and negative sentiment texts."""
    plt.figure(figsize=(16, 8))

    # If target entity is specified, filter data
    if target_entity:
        entity_col = next(
            (
                col
                for col in df.columns
                if "directed" in col.lower()
                or "brand" in col.lower()
                or "entity" in col.lower()
                or "product" in col.lower()
            ),
            None,
        )

        if entity_col:
            filtered_df = df[
                df[entity_col].str.contains(target_entity, case=False, na=False)
            ]
            if len(filtered_df) < 10:
                print(
                    f"Not enough data for entity '{target_entity}'. Using all data instead."
                )
                filtered_df = df
        else:
            filtered_df = df
    else:
        filtered_df = df

    # Get unique sentiment values
    sentiment_values = filtered_df[sentiment_column].unique()

    # Map sentiment values if needed
    if len(sentiment_values) > 2:
        # Handle case where sentiment values are not binary
        print(f"Multiple sentiment values found: {sentiment_values}")
        print("Creating word clouds for the two most common sentiment values.")
        top_sentiments = filtered_df[sentiment_column].value_counts().index[:2]
        positive_sentiment = top_sentiments[0]
        negative_sentiment = top_sentiments[1] if len(top_sentiments) > 1 else None
    else:
        # If we have exactly two values, determine which is positive
        if len(sentiment_values) == 2:
            # Try to guess which is positive based on common patterns
            if any(
                pos in str(val).lower()
                for val in sentiment_values
                for pos in ["positive", "yes", "1", "true"]
            ):
                positive_sentiment = next(
                    val
                    for val in sentiment_values
                    if any(
                        pos in str(val).lower()
                        for pos in ["positive", "yes", "1", "true"]
                    )
                )
                negative_sentiment = next(
                    val for val in sentiment_values if val != positive_sentiment
                )
            else:
                # If we can't determine, just use the first as positive
                positive_sentiment = sentiment_values[0]
                negative_sentiment = sentiment_values[1]
        elif len(sentiment_values) == 1:
            positive_sentiment = sentiment_values[0]
            negative_sentiment = None
        else:
            print("No sentiment values found in the dataset.")
            return

    # Positive sentiment wordcloud
    plt.subplot(1, 2, 1)
    positive_mask = filtered_df[sentiment_column] == positive_sentiment
    if positive_mask.sum() > 0:
        positive_text = " ".join(filtered_df[positive_mask][text_column])
        wordcloud_positive = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
        ).generate(positive_text)
        plt.imshow(wordcloud_positive, interpolation="bilinear")
        plt.title(
            f'Words in {positive_sentiment} Reviews{" about " + target_entity if target_entity else ""}',
            fontsize=16,
        )
        plt.axis("off")
    else:
        plt.text(
            0.5,
            0.5,
            f"No {positive_sentiment} reviews found",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.axis("off")

    # Negative sentiment wordcloud
    plt.subplot(1, 2, 2)
    if negative_sentiment is not None:
        negative_mask = filtered_df[sentiment_column] == negative_sentiment
        if negative_mask.sum() > 0:
            negative_text = " ".join(filtered_df[negative_mask][text_column])
            wordcloud_negative = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="plasma",
                max_words=100,
            ).generate(negative_text)
            plt.imshow(wordcloud_negative, interpolation="bilinear")
            plt.title(
                f'Words in {negative_sentiment} Reviews{" about " + target_entity if target_entity else ""}',
                fontsize=16,
            )
            plt.axis("off")
        else:
            plt.text(
                0.5,
                0.5,
                f"No {negative_sentiment} reviews found",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.axis("off")
    else:
        plt.text(
            0.5,
            0.5,
            "No second sentiment category found",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.axis("off")

    plt.tight_layout()
    if target_entity:
        plt.savefig(
            f"{target_entity.lower()}_sentiment_wordclouds.png",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.savefig("sentiment_wordclouds.png", dpi=300, bbox_inches="tight")
    plt.show()


# Function to analyze a new tweet
def analyze_tweet(tweet_text, model_pipeline):
    """Analyze sentiment of a new tweet using the trained model."""
    # Preprocess the text
    processed_text = preprocess_text(tweet_text)

    # Make prediction
    prediction = model_pipeline.predict([processed_text])[0]

    # Get prediction probability if the model supports predict_proba
    try:
        probabilities = model_pipeline.predict_proba([processed_text])[0]
        confidence = max(probabilities) * 100
        return prediction, confidence
    except:
        return prediction, None


def main():
    try:
        # Attempt to load datasets
        try:
            train_df = pd.read_csv("dataset/Dataset - Train.csv")
            test_df = pd.read_csv("dataset/Dataset - Test.csv")
            print("Datasets loaded successfully!")

            # Print column names to help debug
            print("\nTrain dataset columns:", train_df.columns.tolist())
            print("Test dataset columns:", test_df.columns.tolist())

        except FileNotFoundError:
            print(
                "Dataset files not found. Creating synthetic dataset for demonstration..."
            )

            # Create synthetic dataset for demonstration
            from sklearn.datasets import fetch_20newsgroups

            # Get text data from 20 newsgroups dataset - using subset related to technology
            categories = ["comp.graphics", "rec.autos"]
            newsgroups = fetch_20newsgroups(
                subset="train", categories=categories, shuffle=True, random_state=42
            )

            # Create DataFrame
            data = pd.DataFrame(
                {
                    "tweet_text": newsgroups.data[:1000],
                    "is_there_an_emotion_directed_at_a_brand_or_product": [
                        "Positive" if target == 0 else "Negative"
                        for target in newsgroups.target[:1000]
                    ],
                    "emotion_in_tweet_is_directed_at": [
                        (
                            "Google"
                            if i % 3 == 0
                            else "Microsoft" if i % 3 == 1 else "Apple"
                        )
                        for i in range(1000)
                    ],
                }
            )

            # Split into train and test
            train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
            print("Synthetic dataset created successfully!")

        # Display dataset information
        print("\nTraining dataset info:")
        print(f"Number of records: {len(train_df)}")

        # Identify text columns in both datasets
        # First, find the most likely text column in train dataset
        train_text_candidates = [
            col
            for col in train_df.columns
            if any(
                text_indicator in col.lower()
                for text_indicator in ["text", "tweet", "message", "content"]
            )
        ]

        if train_text_candidates:
            train_text_col = train_text_candidates[0]
        else:
            # If no obvious text column, look for the column with string data that has the longest average length
            str_cols = [
                col for col in train_df.columns if train_df[col].dtype == "object"
            ]
            if str_cols:
                avg_lengths = {
                    col: train_df[col].astype(str).str.len().mean() for col in str_cols
                }
                train_text_col = max(avg_lengths.items(), key=lambda x: x[1])[0]
            else:
                train_text_col = train_df.columns[0]

        print(f"Identified text column in training data: '{train_text_col}'")

        # Now find the corresponding text column in test dataset
        if train_text_col in test_df.columns:
            test_text_col = train_text_col
        else:
            # Try to find a similar column name
            test_text_candidates = [
                col
                for col in test_df.columns
                if any(
                    text_indicator in col.lower()
                    for text_indicator in ["text", "tweet", "message", "content"]
                )
            ]

            if test_text_candidates:
                test_text_col = test_text_candidates[0]
            else:
                # If no obvious text column, look for the column with string data that has the longest average length
                str_cols = [
                    col for col in test_df.columns if test_df[col].dtype == "object"
                ]
                if str_cols:
                    avg_lengths = {
                        col: test_df[col].astype(str).str.len().mean()
                        for col in str_cols
                    }
                    test_text_col = max(avg_lengths.items(), key=lambda x: x[1])[0]
                else:
                    test_text_col = test_df.columns[0]

        print(f"Identified text column in test data: '{test_text_col}'")

        # Identify sentiment column in training dataset
        sentiment_candidates = [
            col
            for col in train_df.columns
            if any(
                sentiment_indicator in col.lower()
                for sentiment_indicator in [
                    "emotion",
                    "sentiment",
                    "feeling",
                    "label",
                    "class",
                ]
            )
        ]

        if sentiment_candidates:
            sentiment_col = sentiment_candidates[0]
        else:
            # If no obvious sentiment column, try to find a column with binary or categorical values
            potential_cols = [
                col
                for col in train_df.columns
                if col != train_text_col
                and train_df[col].nunique() <= 5
                and train_df[col].nunique() > 1
            ]

            if potential_cols:
                sentiment_col = potential_cols[0]
            else:
                # Last resort: use the second column
                cols = list(train_df.columns)
                sentiment_col = (
                    cols[1] if len(cols) > 1 and cols[1] != train_text_col else cols[0]
                )

        print(f"Identified sentiment column: '{sentiment_col}'")

        # Check if sentiment column exists in test dataset
        if sentiment_col in test_df.columns:
            print(f"Sentiment column '{sentiment_col}' found in test dataset.")
            test_has_labels = True
        else:
            print(
                f"Sentiment column '{sentiment_col}' not found in test dataset. Creating test set from training data."
            )
            test_has_labels = False

        # Apply preprocessing
        print("\nPreprocessing text data...")
        train_df["processed_text"] = train_df[train_text_col].apply(preprocess_text)
        test_df["processed_text"] = test_df[test_text_col].apply(preprocess_text)

        # Show sample processed texts
        print("\nSample processed texts:")
        for i in range(min(3, len(train_df))):
            print(f"Original: {train_df[train_text_col].iloc[i][:100]}...")
            print(f"Processed: {train_df['processed_text'].iloc[i][:100]}...")
            print(f"Sentiment: {train_df[sentiment_col].iloc[i]}")
            print()

        # Handle sentiment values to ensure they are standardized
        # Check sentiment values distribution
        sentiment_value_counts = train_df[sentiment_col].value_counts()
        print(f"Sentiment value distribution:\n{sentiment_value_counts}")

        # If we have more than 2 sentiment values or they're numeric, map them to binary
        if (
            train_df[sentiment_col].nunique() > 2
            or train_df[sentiment_col].dtype.kind in "iuf"
        ):
            print("Converting sentiment values to binary (Positive/Negative)...")

            # If numeric, map based on value
            if train_df[sentiment_col].dtype.kind in "iuf":
                # Assuming higher values are more positive
                median = train_df[sentiment_col].median()
                train_df["sentiment_binary"] = train_df[sentiment_col].apply(
                    lambda x: "Positive" if x >= median else "Negative"
                )
                if test_has_labels:
                    test_df["sentiment_binary"] = test_df[sentiment_col].apply(
                        lambda x: "Positive" if x >= median else "Negative"
                    )
            else:
                # For string/categorical values, map the most common value to 'Positive'
                top_sentiment = sentiment_value_counts.index[0]
                train_df["sentiment_binary"] = train_df[sentiment_col].apply(
                    lambda x: "Positive" if x == top_sentiment else "Negative"
                )
                if test_has_labels:
                    test_df["sentiment_binary"] = test_df[sentiment_col].apply(
                        lambda x: "Positive" if x == top_sentiment else "Negative"
                    )

            sentiment_col = "sentiment_binary"

        # Define X and y for training and testing
        X_train = train_df["processed_text"]
        y_train = train_df[sentiment_col]
        X_test = test_df["processed_text"]

        if test_has_labels:
            y_test = test_df[sentiment_col]
        else:
            # Create a test set from training data
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            print("Created test set from training data.")

        # Create simplified pipelines for faster execution
        print("\nCreating model pipelines...")

        # 1. Naive Bayes Pipeline
        nb_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("classifier", MultinomialNB(alpha=1.0)),
            ]
        )

        # 2. Support Vector Machine Pipeline
        svm_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("classifier", LinearSVC(C=1.0, max_iter=10000)),
            ]
        )

        # 3. Logistic Regression Pipeline
        lr_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("classifier", LogisticRegression(C=1.0, max_iter=10000)),
            ]
        )

        # Train all models
        print("\nTraining Naive Bayes model...")
        nb_pipeline.fit(X_train, y_train)

        print("Training SVM model...")
        svm_pipeline.fit(X_train, y_train)

        print("Training Logistic Regression model...")
        lr_pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred_nb = nb_pipeline.predict(X_test)
        y_pred_svm = svm_pipeline.predict(X_test)
        y_pred_lr = lr_pipeline.predict(X_test)

        # Evaluate models
        print("\nEvaluating models on test data:")
        results = []
        results.append(evaluate_model(y_test, y_pred_nb, "Naive Bayes"))
        results.append(evaluate_model(y_test, y_pred_svm, "SVM"))
        results.append(evaluate_model(y_test, y_pred_lr, "Logistic Regression"))

        # Create DataFrame for visualization
        results_df = pd.DataFrame(results)

        # Visualizations
        print("\nCreating visualizations...")

        # 1. Bar chart comparing metrics across models
        metrics = ["accuracy", "precision", "recall", "f1"]
        results_df_melted = pd.melt(
            results_df,
            id_vars=["model"],
            value_vars=metrics,
            var_name="metric",
            value_name="value",
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(x="metric", y="value", hue="model", data=results_df_melted)
        plt.title("Performance Comparison of Sentiment Analysis Models", fontsize=14)
        plt.ylim(0, 1)
        plt.xticks(rotation=0)
        plt.legend(title="Model")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("model_performance_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 2. Radar chart for comparing models
        create_radar_chart(results_df, metrics)

        # 3. Word clouds for positive and negative reviews
        # Try to identify company/entity column
        entity_col = next(
            (
                col
                for col in train_df.columns
                if "brand" in col.lower()
                or "product" in col.lower()
                or "company" in col.lower()
                or "directed" in col.lower()
            ),
            None,
        )

        # Check if we have entity information
        if entity_col and entity_col in train_df.columns:
            # Get the most common entity
            if (
                not train_df[entity_col].isna().all()
                and len(train_df[entity_col].dropna()) > 0
            ):
                top_entity = train_df[entity_col].value_counts().index[0]
                print(f"\nCreating word clouds for entity: {top_entity}")
                plot_word_clouds(train_df, "processed_text", sentiment_col, top_entity)

        # Create general word clouds
        print("\nCreating general sentiment word clouds")
        plot_word_clouds(train_df, "processed_text", sentiment_col)

        # Select best model for new tweet analysis based on F1 score
        best_model_idx = results_df["f1"].idxmax()
        best_model_name = results_df.loc[best_model_idx, "model"]

        if best_model_name == "Naive Bayes":
            best_model = nb_pipeline
        elif best_model_name == "SVM":
            best_model = svm_pipeline
        else:
            best_model = lr_pipeline

        print(
            f"\nBest performing model: {best_model_name} with F1 score: {results_df.loc[best_model_idx, 'f1']:.4f}"
        )

        # Interactive tweet analysis
        print("\n--- Tweet Sentiment Analyzer ---")
        print("Enter tweets to analyze their sentiment (type 'exit' to quit):")

        example_tweets = [
            "I absolutely love my new Google Pixel! The camera is amazing and the battery lasts all day.",
            "This Apple iPhone keeps crashing and the battery drains too quickly. Terrible experience so far.",
            "Microsoft's customer service was helpful in resolving my issue with Windows. Good job!",
            "Netflix recommendations are spot on today. So many good shows to watch!",
        ]

        print("\nExample tweets you can try:")
        for i, tweet in enumerate(example_tweets):
            print(f"{i+1}. {tweet}")

        while True:
            user_input = input("\nEnter a tweet to analyze (or 'exit' to quit): ")
            if user_input.lower() == "exit":
                break

            # Use a number as shortcut for example tweets
            try:
                idx = int(user_input) - 1
                if 0 <= idx < len(example_tweets):
                    user_input = example_tweets[idx]
                    print(f"Analyzing example tweet: {user_input}")
            except ValueError:
                pass

            sentiment, confidence = analyze_tweet(user_input, best_model)
            print(f"Sentiment: {sentiment}")
            if confidence:
                print(f"Confidence: {confidence:.2f}%")

        print("\nThank you for using the Tweet Sentiment Analyzer!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
