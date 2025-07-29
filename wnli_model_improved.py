#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, regexp_replace, lower, concat_ws, length, array_intersect, size, lit, when
from pyspark.sql.types import FloatType, ArrayType, IntegerType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler, HashingTF, Word2Vec
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as roc_auc_score, confusion_matrix, precision_recall_curve
import seaborn as sns

# 设置 SPARK_HOME 环境变量，根据实际情况修改路径
os.environ['SPARK_HOME'] = 'D:\spark-3.3.4-bin-hadoop3'
os.environ['PYSPARK_PYTHON'] = r'D:\anaconda3\envs\pyspark-3.3.4\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'D:\anaconda3\envs\pyspark-3.3.4\python.exe'

# Initialize Spark session
spark = SparkSession.builder \
    .appName("WNLI Natural Language Inference - Improved") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Download NLTK resources (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load training data
train_df = spark.read.csv("WNLI/train.tsv", sep="\t", header=True, inferSchema=True)
print("Training data loaded, count:", train_df.count())
train_df.printSchema()

# Load test data
test_df = spark.read.csv("WNLI/test.tsv", sep="\t", header=True, inferSchema=True)
print("Test data loaded, count:", test_df.count())
test_df.printSchema()


# Data preprocessing
def clean_text(text):
    """Clean text by removing special characters and extra whitespaces"""
    if text is None:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Remove extra whitespaces
    text = re.sub('\s+', ' ', text).strip()
    return text


# Lemmatization function
def lemmatize_text(tokens):
    """Lemmatize tokens"""
    if tokens is None:
        return []
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


# Check for negation words
def contains_negation(tokens):
    """Check if tokens contain negation words"""
    if tokens is None:
        return 0
    negation_words = {"no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "hardly",
                      "scarcely", "barely", "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't",
                      "can't", "don't"}
    return 1 if any(token in negation_words for token in tokens) else 0


# Word overlap ratio
def word_overlap_ratio(tokens1, tokens2):
    """Calculate word overlap ratio between two token lists"""
    if not tokens1 or not tokens2:
        return 0.0

    set1 = set(tokens1)
    set2 = set(tokens2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    return len(intersection) / len(union) if union else 0.0


# Register UDFs
clean_text_udf = udf(clean_text, StringType())
lemmatize_udf = udf(lemmatize_text, ArrayType(StringType()))
contains_negation_udf = udf(contains_negation, IntegerType())
word_overlap_ratio_udf = udf(word_overlap_ratio, FloatType())

# Apply preprocessing to training data
train_processed = train_df.withColumn("sentence1_clean", clean_text_udf(col("sentence1"))) \
    .withColumn("sentence2_clean", clean_text_udf(col("sentence2"))) \
    .withColumn("combined_text", concat_ws(" ", col("sentence1_clean"), col("sentence2_clean")))

# Apply preprocessing to test data
test_processed = test_df.withColumn("sentence1_clean", clean_text_udf(col("sentence1"))) \
    .withColumn("sentence2_clean", clean_text_udf(col("sentence2"))) \
    .withColumn("combined_text", concat_ws(" ", col("sentence1_clean"), col("sentence2_clean")))

# Feature engineering pipeline
# Tokenization
tokenizer = Tokenizer(inputCol="combined_text", outputCol="tokens")
tokenizer1 = Tokenizer(inputCol="sentence1_clean", outputCol="tokens1")
tokenizer2 = Tokenizer(inputCol="sentence2_clean", outputCol="tokens2")

# Stop words removal
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
stopwords_remover1 = StopWordsRemover(inputCol="tokens1", outputCol="filtered_tokens1")
stopwords_remover2 = StopWordsRemover(inputCol="tokens2", outputCol="filtered_tokens2")

# TF-IDF features
cv = CountVectorizer(inputCol="filtered_tokens", outputCol="tf_features", minDF=2.0)
idf = IDF(inputCol="tf_features", outputCol="tf_idf_features")

cv1 = CountVectorizer(inputCol="filtered_tokens1", outputCol="tf_features1", minDF=2.0)
idf1 = IDF(inputCol="tf_features1", outputCol="tf_idf_features1")

cv2 = CountVectorizer(inputCol="filtered_tokens2", outputCol="tf_features2", minDF=2.0)
idf2 = IDF(inputCol="tf_features2", outputCol="tf_idf_features2")

# Word2Vec for semantic features
word2Vec = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered_tokens", outputCol="word2vec_features")
word2Vec1 = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered_tokens1", outputCol="word2vec_features1")
word2Vec2 = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered_tokens2", outputCol="word2vec_features2")

# Create pipeline stages
stages = [
    tokenizer, tokenizer1, tokenizer2,
    stopwords_remover, stopwords_remover1, stopwords_remover2,
    cv, idf, cv1, idf1, cv2, idf2,
    word2Vec, word2Vec1, word2Vec2
]

# Create initial pipeline
featurePipeline = Pipeline(stages=stages)

# Fit the feature pipeline
print("Fitting feature pipeline...")
featureModel = featurePipeline.fit(train_processed)

# Transform the data with feature pipeline
train_features = featureModel.transform(train_processed)
test_features = featureModel.transform(test_processed)

# Add additional features
print("Adding additional features...")

# Add length features
train_features = train_features.withColumn("len_diff",
                                           length(col("sentence1_clean")) - length(col("sentence2_clean")))
train_features = train_features.withColumn("len_ratio",
                                           when(length(col("sentence2_clean")) > 0,
                                                length(col("sentence1_clean")) / length(
                                                    col("sentence2_clean"))).otherwise(0))

# Add word overlap features
train_features = train_features.withColumn("token_overlap",
                                           size(array_intersect(col("filtered_tokens1"), col("filtered_tokens2"))))
train_features = train_features.withColumn("token_overlap_ratio",
                                           when(size(col("filtered_tokens1")) + size(col("filtered_tokens2")) > 0,
                                                size(
                                                    array_intersect(col("filtered_tokens1"), col("filtered_tokens2"))) /
                                                (size(col("filtered_tokens1")) + size(col("filtered_tokens2")) -
                                                 size(array_intersect(col("filtered_tokens1"),
                                                                      col("filtered_tokens2"))))).otherwise(0))

# Add negation features
train_features = train_features.withColumn("negation_s1", contains_negation_udf(col("filtered_tokens1")))
train_features = train_features.withColumn("negation_s2", contains_negation_udf(col("filtered_tokens2")))
train_features = train_features.withColumn("negation_diff",
                                           when(col("negation_s1") != col("negation_s2"), 1).otherwise(0))

# Same additional features for test data
test_features = test_features.withColumn("len_diff",
                                         length(col("sentence1_clean")) - length(col("sentence2_clean")))
test_features = test_features.withColumn("len_ratio",
                                         when(length(col("sentence2_clean")) > 0,
                                              length(col("sentence1_clean")) / length(
                                                  col("sentence2_clean"))).otherwise(0))
test_features = test_features.withColumn("token_overlap",
                                         size(array_intersect(col("filtered_tokens1"), col("filtered_tokens2"))))
test_features = test_features.withColumn("token_overlap_ratio",
                                         when(size(col("filtered_tokens1")) + size(col("filtered_tokens2")) > 0,
                                              size(array_intersect(col("filtered_tokens1"), col("filtered_tokens2"))) /
                                              (size(col("filtered_tokens1")) + size(col("filtered_tokens2")) -
                                               size(array_intersect(col("filtered_tokens1"),
                                                                    col("filtered_tokens2"))))).otherwise(0))
test_features = test_features.withColumn("negation_s1", contains_negation_udf(col("filtered_tokens1")))
test_features = test_features.withColumn("negation_s2", contains_negation_udf(col("filtered_tokens2")))
test_features = test_features.withColumn("negation_diff",
                                         when(col("negation_s1") != col("negation_s2"), 1).otherwise(0))

# Assemble all features
assembler = VectorAssembler(
    inputCols=[
        "tf_idf_features", "tf_idf_features1", "tf_idf_features2",
        "word2vec_features", "word2vec_features1", "word2vec_features2",
        "len_diff", "len_ratio", "token_overlap", "token_overlap_ratio",
        "negation_s1", "negation_s2", "negation_diff"
    ],
    outputCol="features",
    handleInvalid="skip"
)

# Create a Random Forest classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10)

# Create parameter grid for cross-validation
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Create cross-validator
crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="label"),
                          numFolds=5)

# Assemble features and train the model
print("Assembling features...")
train_assembled = assembler.transform(train_features)
test_assembled = assembler.transform(test_features)

# Train the model with cross-validation
print("Training the model with cross-validation...")
model = crossval.fit(train_assembled)

# Get best model
best_model = model.bestModel
print(f"Best model parameters: numTrees={best_model.getNumTrees}, maxDepth={best_model.getMaxDepth()}")

# Make predictions on training data to evaluate model performance
train_predictions = best_model.transform(train_assembled)


# Function to calculate and display comprehensive evaluation metrics
def evaluate_model(predictions_df, dataset_name):
    """
    Calculate and display comprehensive evaluation metrics for model predictions
    """
    print(f"\n===== {dataset_name} Evaluation Metrics =====")

    # Check if this dataset has labels (test data might not have labels)
    has_labels = "label" in predictions_df.columns

    # Convert to Pandas for easier manipulation and visualization
    if has_labels:
        pandas_df = predictions_df.select("label", "prediction", "probability").toPandas()
        # Extract positive class probability for ROC curve
        pandas_df["probability_pos"] = pandas_df["probability"].apply(lambda x: float(x[1]))
    else:
        pandas_df = predictions_df.select("prediction", "probability").toPandas()
        pandas_df["probability_pos"] = pandas_df["probability"].apply(lambda x: float(x[1]))
        print("No 'label' column found. Skipping evaluation metrics that require ground truth labels.")

    metrics = {}

    if has_labels:
        # Calculate regression metrics
        regression_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")

        mse = regression_evaluator.setMetricName("mse").evaluate(predictions_df)
        rmse = regression_evaluator.setMetricName("rmse").evaluate(predictions_df)
        mae = regression_evaluator.setMetricName("mae").evaluate(predictions_df)
        r2 = regression_evaluator.setMetricName("r2").evaluate(predictions_df)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² (Coefficient of Determination): {r2:.4f}")

        # Calculate classification metrics
        binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
        multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

        auc_value = binary_evaluator.evaluate(predictions_df)
        accuracy = multi_evaluator.setMetricName("accuracy").evaluate(predictions_df)
        precision = multi_evaluator.setMetricName("weightedPrecision").evaluate(predictions_df)
        recall = multi_evaluator.setMetricName("weightedRecall").evaluate(predictions_df)
        f1 = multi_evaluator.setMetricName("f1").evaluate(predictions_df)

        print(f"AUC: {auc_value:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "auc": auc_value,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Create directory for plots if it doesn't exist
    os.makedirs("evaluation_plots", exist_ok=True)

    if has_labels:
        # Plot confusion matrix
        y_true = pandas_df["label"].values
        y_pred = pandas_df["prediction"].values
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {dataset_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"evaluation_plots/{dataset_name.lower()}_confusion_matrix.png")
        plt.close()

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, pandas_df["probability_pos"])
        roc_auc = roc_auc_score(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {dataset_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"evaluation_plots/{dataset_name.lower()}_roc_curve.png")
        plt.close()

        # Plot Precision-Recall curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, pandas_df["probability_pos"])

        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {dataset_name}")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig(f"evaluation_plots/{dataset_name.lower()}_precision_recall_curve.png")
        plt.close()

    # Plot prediction distribution (works without labels)
    plt.figure(figsize=(8, 6))
    predictions = pandas_df["prediction"].values
    plt.hist(predictions, bins=2, alpha=0.7, color="blue")
    plt.title(f"Prediction Distribution - {dataset_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.xticks([0, 1])
    plt.tight_layout()
    plt.savefig(f"evaluation_plots/{dataset_name.lower()}_prediction_distribution.png")
    plt.close()

    # Plot probability distribution (works without labels)
    plt.figure(figsize=(8, 6))
    probabilities = pandas_df["probability_pos"].values
    plt.hist(probabilities, bins=10, alpha=0.7, color="green")
    plt.title(f"Probability Distribution - {dataset_name}")
    plt.xlabel("Probability of Positive Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"evaluation_plots/{dataset_name.lower()}_probability_distribution.png")
    plt.close()

    # Plot feature importance if available
    if hasattr(best_model, "featureImportances"):
        # Get feature names from the assembler
        feature_names = assembler.getInputCols()
        importances = best_model.featureImportances.toArray()

        # If importances array is larger than feature names (e.g., due to one-hot encoding)
        # Just use indices instead
        if len(importances) > len(feature_names):
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        # Sort features by importance
        indices = np.argsort(importances)[::-1]

        # Plot top 15 features or all if less than 15
        top_n = min(15, len(feature_names))

        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_n} Feature Importances")
        plt.bar(range(top_n), importances[indices[:top_n]], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
        plt.tight_layout()
        plt.savefig(f"evaluation_plots/{dataset_name.lower()}_feature_importance.png")
        plt.close()

    # Return metrics as a dictionary for further use
    return metrics


# Evaluate model on training data
print("\nEvaluating model on training data...")
train_metrics = evaluate_model(train_predictions, "Training")

# Make predictions on test data
print("\nMaking predictions on test data...")
test_predictions = best_model.transform(test_assembled)

# Evaluate model on test data
test_metrics = evaluate_model(test_predictions, "Test")

# Compare training and test metrics if test data has labels
if test_metrics:  # If test_metrics is not empty
    metrics_names = ["MSE", "RMSE", "MAE", "R²", "AUC", "Accuracy", "Precision", "Recall", "F1"]
    metrics_values_train = [train_metrics["mse"], train_metrics["rmse"], train_metrics["mae"],
                            train_metrics["r2"], train_metrics["auc"], train_metrics["accuracy"],
                            train_metrics["precision"], train_metrics["recall"], train_metrics["f1"]]
    metrics_values_test = [test_metrics["mse"], test_metrics["rmse"], test_metrics["mae"],
                           test_metrics["r2"], test_metrics["auc"], test_metrics["accuracy"],
                           test_metrics["precision"], test_metrics["recall"], test_metrics["f1"]]

    # Plot comparison of metrics
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics_names))
    width = 0.35

    plt.bar(x - width / 2, metrics_values_train, width, label="Training")
    plt.bar(x + width / 2, metrics_values_test, width, label="Test")

    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title("Model Performance Comparison: Training vs Test")
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_plots/metrics_comparison.png")
    plt.close()

    # Save metrics to CSV for future reference
    metrics_df = pd.DataFrame({
        'Metric': metrics_names,
        'Training': metrics_values_train,
        'Test': metrics_values_test
    })
    metrics_df.to_csv("evaluation_plots/metrics_summary.csv", index=False)
    print("Metrics summary saved to evaluation_plots/metrics_summary.csv")
else:
    # Save only training metrics if test data has no labels
    metrics_names = ["MSE", "RMSE", "MAE", "R²", "AUC", "Accuracy", "Precision", "Recall", "F1"]
    metrics_values_train = [train_metrics["mse"], train_metrics["rmse"], train_metrics["mae"],
                            train_metrics["r2"], train_metrics["auc"], train_metrics["accuracy"],
                            train_metrics["precision"], train_metrics["recall"], train_metrics["f1"]]

    # Save training metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': metrics_names,
        'Training': metrics_values_train
    })
    metrics_df.to_csv("evaluation_plots/metrics_summary.csv", index=False)
    print("Training metrics summary saved to evaluation_plots/metrics_summary.csv")

print("\nEvaluation complete. Visualizations saved in the 'evaluation_plots' directory.")

# Stop Spark session
spark.stop()


# Create a separate visualization for the specified metrics
def visualize_key_metrics(metrics_dict, dataset_name="Model"):
    """Create a bar chart visualization for key classification metrics"""
    # Key metrics to visualize
    key_metrics = ["auc", "accuracy", "precision", "recall", "f1"]
    metric_labels = ["AUC", "Accuracy", "Precision", "Recall", "F1 Score"]

    # Extract values for the key metrics
    metric_values = [metrics_dict.get(metric, 0) for metric in key_metrics]

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_labels, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # Customize the chart
    plt.ylim(0, 1.0)
    plt.title(f'Key Classification Metrics - {dataset_name}')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"evaluation_plots/{dataset_name.lower()}_key_metrics.png")
    plt.close()

    print(f"Key metrics visualization saved as evaluation_plots/{dataset_name.lower()}_key_metrics.png")


# Create the key metrics visualization if we have metrics
if train_metrics:
    visualize_key_metrics(train_metrics, "Training")

if test_metrics:
    visualize_key_metrics(test_metrics, "Test")