import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer, NGram
from pyspark.ml.feature import VectorAssembler, StandardScaler, Word2Vec
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, length, lit, lower, regexp_replace, size, array_intersect, \
    abs as sql_abs, levenshtein, udf
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.base import Transformer
import pandas as pd
from pyspark.sql.types import ArrayType, StringType, FloatType

# 设置 SPARK_HOME 环境变量，根据实际情况修改路径
os.environ['SPARK_HOME'] = 'D:\spark-3.3.4-bin-hadoop3'
os.environ['PYSPARK_PYTHON'] = r'D:\anaconda3\envs\pyspark-3.3.4\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'D:\anaconda3\envs\pyspark-3.3.4\python.exe'

# Initialize Spark Session with extreme memory-optimized settings
spark = SparkSession.builder \
    .appName("SentencePairClassification") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.2") \
    .config("spark.driver.maxResultSize", "512m") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.default.parallelism", "4") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "256m") \
    .getOrCreate()


# Custom transformer for token overlap
class TokenOverlapTransformer(Transformer):
    def __init__(self, inputCols=None, outputCol=None):
        super(TokenOverlapTransformer, self).__init__()
        self.inputCols = inputCols
        self.outputCol = outputCol

    def _transform(self, dataset):
        return dataset.withColumn(self.outputCol,
                                  size(array_intersect(col(self.inputCols[0]), col(self.inputCols[1]))))


# Custom transformer for Levenshtein distance
class LevenshteinDistanceTransformer(Transformer):
    def __init__(self, inputCols=None, outputCol=None):
        super(LevenshteinDistanceTransformer, self).__init__()
        self.inputCols = inputCols
        self.outputCol = outputCol

    def _transform(self, dataset):
        return dataset.withColumn(self.outputCol,
                                  levenshtein(col(self.inputCols[0]), col(self.inputCols[1])))


# Custom transformer for token overlap ratio
class TokenOverlapRatioTransformer(Transformer):
    def __init__(self, inputCols=None, outputCol=None):
        super(TokenOverlapRatioTransformer, self).__init__()
        self.inputCols = inputCols
        self.outputCol = outputCol

    def _transform(self, dataset):
        overlap = size(array_intersect(col(self.inputCols[0]), col(self.inputCols[1])))
        total = size(col(self.inputCols[0])) + size(col(self.inputCols[1]))
        return dataset.withColumn(self.outputCol,
                                  when(total > 0, (overlap * 2.0) / total)
                                  .otherwise(lit(0.0)))


def clean_text(text_col):
    """Clean text column by removing special characters and converting to lowercase"""
    return lower(regexp_replace(text_col, "[^a-zA-Z\\s]", " "))


# Load training data and handle missing values with minimal memory usage
def load_and_clean_data(file_path, is_train=True, sample_ratio=None):
    # Read the data
    df = spark.read.csv(file_path, sep="\t", header=True)

    # Apply sampling if ratio is provided
    if sample_ratio and sample_ratio < 1.0:
        df = df.sample(False, sample_ratio, seed=42)

    # For training data, map gold_label to numeric label
    if is_train:
        # Clean and preprocess text
        df = df.withColumn("sentence1_cleaned",
                           when((col("sentence1").isNull()) | (length(col("sentence1")) < 1),
                                lit("empty_sentence"))
                           .otherwise(clean_text(col("sentence1"))))

        df = df.withColumn("sentence2_cleaned",
                           when((col("sentence2").isNull()) | (length(col("sentence2")) < 1),
                                lit("empty_sentence"))
                           .otherwise(clean_text(col("sentence2"))))

        # Map gold_label to numeric label
        # entailment: 0, contradiction: 1, neutral: 2
        df = df.withColumn("label",
                           when(col("gold_label") == "entailment", lit(0.0))
                           .when(col("gold_label") == "contradiction", lit(1.0))
                           .when(col("gold_label") == "neutral", lit(2.0))
                           .otherwise(lit(2.0)))  # Default to neutral for any other values
    else:
        # For test data, just clean the text
        df = df.withColumn("sentence1_cleaned",
                           when((col("sentence1").isNull()) | (length(col("sentence1")) < 1),
                                lit("empty_sentence"))
                           .otherwise(clean_text(col("sentence1"))))

        df = df.withColumn("sentence2_cleaned",
                           when((col("sentence2").isNull()) | (length(col("sentence2")) < 1),
                                lit("empty_sentence"))
                           .otherwise(clean_text(col("sentence2"))))

    # Add basic statistical features
    df = df.withColumn("s1_length", length(col("sentence1_cleaned")))
    df = df.withColumn("s2_length", length(col("sentence2_cleaned")))
    df = df.withColumn("length_diff", sql_abs(col("s1_length") - col("s2_length")))
    df = df.withColumn("length_ratio",
                       when(col("s2_length") > 0, col("s1_length") / col("s2_length"))
                       .otherwise(lit(0.0)))

    return df


# Load and clean data with minimal sample ratio to save memory
sample_ratio = 0.25  # Use 25% of data for training - increased for better accuracy
train_data = load_and_clean_data("AX/AX_train.tsv", is_train=True, sample_ratio=sample_ratio)
test_data = load_and_clean_data("AX/AX_test.tsv", is_train=False)

# Print data info
print("Training data count:", train_data.count())
print("Test data count:", test_data.count())


def create_text_features(input_col, output_prefix):
    """Create a pipeline for text feature extraction with improved accuracy but low memory usage"""
    # Tokenization
    tokenizer = Tokenizer(inputCol=input_col, outputCol=f"{output_prefix}_tokens")

    # Remove stop words
    remover = StopWordsRemover(inputCol=f"{output_prefix}_tokens", outputCol=f"{output_prefix}_filtered")

    # Create n-grams
    ngram2 = NGram(n=2, inputCol=f"{output_prefix}_filtered", outputCol=f"{output_prefix}_ngrams2")

    # Word2Vec for semantic features
    word2Vec = Word2Vec(vectorSize=50, minCount=2,
                        inputCol=f"{output_prefix}_filtered",
                        outputCol=f"{output_prefix}_w2v")

    # CountVectorizer (more memory efficient than HashingTF)
    cv = CountVectorizer(inputCol=f"{output_prefix}_filtered", outputCol=f"{output_prefix}_cv",
                         vocabSize=300, minDF=2.0)

    # N-gram features
    cv_ngram = CountVectorizer(inputCol=f"{output_prefix}_ngrams2", outputCol=f"{output_prefix}_cv_ngram",
                               vocabSize=200, minDF=2.0)

    # TF-IDF with optimized feature dimensions
    hashingTF = HashingTF(inputCol=f"{output_prefix}_filtered", outputCol=f"{output_prefix}_tf", numFeatures=200)
    idf = IDF(inputCol=f"{output_prefix}_tf", outputCol=f"{output_prefix}_features", minDocFreq=2)

    return [tokenizer, remover, ngram2, word2Vec, cv, cv_ngram, hashingTF, idf]


# Data preprocessing with minimal memory usage but improved features
def preprocess_data(df, is_train=True):
    # Create text features for both sentences
    s1_stages = create_text_features("sentence1_cleaned", "s1")
    s2_stages = create_text_features("sentence2_cleaned", "s2")

    # Add token overlap transformer
    token_overlap = TokenOverlapTransformer(
        inputCols=["s1_tokens", "s2_tokens"],
        outputCol="token_overlap"
    )

    # Add token overlap ratio transformer
    token_overlap_ratio = TokenOverlapRatioTransformer(
        inputCols=["s1_tokens", "s2_tokens"],
        outputCol="token_overlap_ratio"
    )

    # Add Levenshtein distance transformer - helpful for contradiction detection
    levenshtein_distance = LevenshteinDistanceTransformer(
        inputCols=["sentence1_cleaned", "sentence2_cleaned"],
        outputCol="levenshtein_distance"
    )

    # Combine all features
    assembler = VectorAssembler(
        inputCols=["s1_features", "s2_features", "s1_cv", "s2_cv",
                   "s1_cv_ngram", "s2_cv_ngram", "s1_w2v", "s2_w2v",
                   "s1_length", "s2_length", "length_diff", "length_ratio",
                   "token_overlap", "token_overlap_ratio", "levenshtein_distance"],
        outputCol="assembled_features"
    )

    # Scale features - helps with logistic regression
    scaler = StandardScaler(inputCol="assembled_features", outputCol="features",
                            withStd=True, withMean=False)  # withMean=False to save memory

    # Create pipeline
    pipeline_stages = s1_stages + s2_stages + [token_overlap, token_overlap_ratio,
                                               levenshtein_distance, assembler, scaler]

    return Pipeline(stages=pipeline_stages)


# Print sample of data before processing
print("\nSample of training data:")
train_data.select("sentence1_cleaned", "sentence2_cleaned", "gold_label", "label",
                  "s1_length", "s2_length", "length_diff").show(3, truncate=False)

# Prepare training pipeline
print("\nPreparing training pipeline...")
train_pipeline = preprocess_data(train_data)
print("Transforming training data...")
model_data = train_pipeline.fit(train_data).transform(train_data)

# Try different classifiers for better accuracy
# 1. Logistic Regression - works well for multiclass problems
lr = LogisticRegression(maxIter=150, regParam=0.01, elasticNetParam=0.1,
                        featuresCol="features", labelCol="label", family="multinomial")

# 2. Random Forest - often good for complex classification tasks
rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                            numTrees=50, maxDepth=10, seed=42)

# Train models
print("\nTraining Logistic Regression model...")
lr_model = lr.fit(model_data)

print("\nTraining Random Forest model...")
rf_model = rf.fit(model_data)

# Evaluate models
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
lr_accuracy = evaluator.evaluate(lr_model.transform(model_data))
rf_accuracy = evaluator.evaluate(rf_model.transform(model_data))

print(f"Logistic Regression Training Accuracy: {lr_accuracy:.4f}")
print(f"Random Forest Training Accuracy: {rf_accuracy:.4f}")

# Select best model
if lr_accuracy > rf_accuracy:
    best_model = lr_model
    best_model_name = "Logistic Regression"
    best_accuracy = lr_accuracy
else:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_accuracy

print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Prepare test data
print("\nPreparing test data...")
test_data_transformed = train_pipeline.fit(test_data).transform(test_data)

# Make predictions
print("\nMaking predictions...")
predictions = best_model.transform(test_data_transformed)


# Calculate metrics
def calculate_metrics(predictions_df, label_col="label"):
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="weightedRecall")

    # Only calculate if we have labels
    if label_col in predictions_df.columns:
        accuracy = accuracy_evaluator.evaluate(predictions_df)
        f1 = f1_evaluator.evaluate(predictions_df)
        precision = precision_evaluator.evaluate(predictions_df)
        recall = recall_evaluator.evaluate(predictions_df)

        return {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        }
    return {}


# Calculate and print metrics for training data
if "label" in model_data.columns:
    print("\nCalculating metrics...")
    train_metrics = calculate_metrics(best_model.transform(model_data))
    print("\nTraining Data Performance Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")


# Create confusion matrix and visualize metrics
def plot_metrics(predictions_df, label_col="label", prediction_col="prediction"):
    # Convert to pandas
    pred_pd = predictions_df.select(label_col, prediction_col).toPandas()

    # Create confusion matrix
    cm = confusion_matrix(pred_pd[label_col], pred_pd[prediction_col])

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Entailment', 'Contradiction', 'Neutral'],
                yticklabels=['Entailment', 'Contradiction', 'Neutral'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    # Get classification report
    report = classification_report(pred_pd[label_col], pred_pd[prediction_col],
                                   target_names=['Entailment', 'Contradiction', 'Neutral'],
                                   output_dict=True)

    # Plot classification report
    plt.figure(figsize=(10, 6))
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop('support', axis=1)
    sns.barplot(data=report_df.iloc[0:3], palette='viridis')
    plt.title(f'Classification Metrics by Class - {best_model_name}')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('classification_metrics.png')

    # Plot overall metrics
    plt.figure(figsize=(8, 6))
    metrics_df = pd.DataFrame(train_metrics.items(), columns=['Metric', 'Value'])
    sns.barplot(x='Metric', y='Value', data=metrics_df, palette='viridis')
    plt.title(f'Overall Model Performance - {best_model_name}')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('overall_metrics.png')

    print("\nVisualization saved to:")
    print("- confusion_matrix.png")
    print("- classification_metrics.png")
    print("- overall_metrics.png")

    return cm, report


# Plot metrics for training data
if "label" in model_data.columns:
    print("\nCreating performance visualizations...")
    cm, report = plot_metrics(best_model.transform(model_data))
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose().round(3))

# Save test predictions to CSV
print("\nSaving predictions...")
predictions_df = predictions.select("index", "sentence1", "sentence2", "prediction", "probability").toPandas()

# Map numeric predictions back to text labels
label_map = {0.0: "entailment", 1.0: "contradiction", 2.0: "neutral"}
predictions_df["predicted_label"] = predictions_df["prediction"].map(label_map)

# Save to CSV
predictions_df.to_csv('ax_test_predictions.csv', index=False)

print("\nProcess completed!")
print(f"- Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
print("- Check ax_test_predictions.csv for test data predictions")
print("- Performance visualizations saved as PNG files")

# Clean up
spark.stop()