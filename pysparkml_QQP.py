from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, NGram
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, length, lit, isnull, lower, regexp_replace, size, array_intersect, \
    abs as sql_abs
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.base import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd

# Initialize Spark Session with optimized memory settings
spark = SparkSession.builder \
    .appName("QuoraQuestionPairs") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.default.parallelism", "20") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "4g") \
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


def clean_text(text_col):
    """Clean text column by removing special characters and converting to lowercase"""
    return lower(regexp_replace(text_col, "[^a-zA-Z\\s]", " "))


# Load training data and handle missing values
def load_and_clean_data(file_path, is_train=True, sample_ratio=None):
    # Read the data
    df = spark.read.csv(file_path, sep="\t", header=True)

    # Apply sampling if ratio is provided
    if sample_ratio and sample_ratio < 1.0:
        df = df.sample(False, sample_ratio, seed=42)

    # Clean and preprocess text
    df = df.withColumn("question1_cleaned",
                       when((col("question1").isNull()) | (length(col("question1")) < 1),
                            lit("empty_question"))
                       .otherwise(clean_text(col("question1"))))

    df = df.withColumn("question2_cleaned",
                       when((col("question2").isNull()) | (length(col("question2")) < 1),
                            lit("empty_question"))
                       .otherwise(clean_text(col("question2"))))

    # Add basic statistical features
    df = df.withColumn("q1_length", length(col("question1_cleaned")))
    df = df.withColumn("q2_length", length(col("question2_cleaned")))
    df = df.withColumn("length_diff", sql_abs(col("q1_length") - col("q2_length")))

    # For training data, handle is_duplicate column
    if is_train:
        df = df.withColumn("is_duplicate",
                           when(col("is_duplicate").isNull(), lit(0))
                           .otherwise(col("is_duplicate")))
        df = df.withColumn("label", col("is_duplicate").cast("double"))

    return df


# Load and clean data with sampling
sample_ratio = 0.3  # Use 30% of data for training
train_data = load_and_clean_data("QQP/train.tsv", is_train=True, sample_ratio=sample_ratio)
test_data = load_and_clean_data("QQP/test.tsv", is_train=False)

# Print data info
print("Training data count:", train_data.count())
print("Test data count:", test_data.count())


def create_text_features(input_col, output_prefix):
    """Create a pipeline for text feature extraction"""
    # Tokenization
    tokenizer = Tokenizer(inputCol=input_col, outputCol=f"{output_prefix}_tokens")

    # Remove stop words
    remover = StopWordsRemover(inputCol=f"{output_prefix}_tokens", outputCol=f"{output_prefix}_filtered")

    # Create word n-grams (reduced from 3-grams to 2-grams)
    ngram2 = NGram(n=2, inputCol=f"{output_prefix}_filtered", outputCol=f"{output_prefix}_ngrams2")

    # TF-IDF for different features (reduced feature dimensions)
    hashingTF = HashingTF(inputCol=f"{output_prefix}_filtered", outputCol=f"{output_prefix}_tf", numFeatures=1000)
    hashingTF_ngram2 = HashingTF(inputCol=f"{output_prefix}_ngrams2", outputCol=f"{output_prefix}_tf_ngram2",
                                 numFeatures=1000)

    idf = IDF(inputCol=f"{output_prefix}_tf", outputCol=f"{output_prefix}_features")
    idf_ngram2 = IDF(inputCol=f"{output_prefix}_tf_ngram2", outputCol=f"{output_prefix}_features_ngram2")

    return [tokenizer, remover, ngram2,
            hashingTF, hashingTF_ngram2,
            idf, idf_ngram2]


# Data preprocessing
def preprocess_data(df, is_train=True):
    # Create text features for both questions
    q1_stages = create_text_features("question1_cleaned", "q1")
    q2_stages = create_text_features("question2_cleaned", "q2")

    # Add token overlap transformer
    token_overlap = TokenOverlapTransformer(
        inputCols=["q1_tokens", "q2_tokens"],
        outputCol="token_overlap"
    )

    # Combine all features
    assembler = VectorAssembler(
        inputCols=["q1_features", "q2_features",
                   "q1_features_ngram2", "q2_features_ngram2",
                   "q1_length", "q2_length", "length_diff", "token_overlap"],
        outputCol="assembled_features"
    )

    # Scale features
    scaler = StandardScaler(inputCol="assembled_features", outputCol="features",
                            withStd=True, withMean=True)

    # Create pipeline
    pipeline_stages = q1_stages + q2_stages + [token_overlap, assembler, scaler]

    return Pipeline(stages=pipeline_stages)


# Print sample of data before processing
print("\nSample of training data:")
train_data.select("question1_cleaned", "question2_cleaned", "is_duplicate", "label",
                  "q1_length", "q2_length", "length_diff").show(5, truncate=False)

# Prepare training pipeline
print("\nPreparing training pipeline...")
train_pipeline = preprocess_data(train_data)
print("Transforming training data...")
model_data = train_pipeline.fit(train_data).transform(train_data)

# Create parameter grid for cross validation (reduced parameter combinations)
lr = LogisticRegression(maxIter=100, featuresCol="features", labelCol="label")
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 1.0]) \
    .addGrid(lr.threshold, [0.5]) \
    .build()

# Create cross validator with reduced folds
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="label"),
                          numFolds=2)  # Reduced from 3 to 2 folds

# Train model with cross validation
print("\nTraining model with cross validation...")
cv_model = crossval.fit(model_data)

# Get the best model
model = cv_model.bestModel
print("\nBest model parameters:")
print(f"RegParam: {model.getRegParam()}")
print(f"ElasticNetParam: {model.getElasticNetParam()}")
print(f"Threshold: {model.getThreshold()}")

# Prepare test data
print("\nPreparing test data...")
test_pipeline = preprocess_data(test_data, is_train=False)
test_data_transformed = train_pipeline.fit(test_data).transform(test_data)

# Make predictions
print("\nMaking predictions...")
predictions = model.transform(test_data_transformed)

# For training data evaluation
if "label" in train_data.columns:
    print("\nEvaluating model on training data...")
    train_predictions = model.transform(model_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    train_auc = evaluator.evaluate(train_predictions)


    def calculate_metrics(predictions_df):
        total = predictions_df.count()
        tp = predictions_df.filter((col("prediction") == 1) & (col("label") == 1)).count()
        tn = predictions_df.filter((col("prediction") == 0) & (col("label") == 0)).count()
        fp = predictions_df.filter((col("prediction") == 1) & (col("label") == 0)).count()
        fn = predictions_df.filter((col("prediction") == 0) & (col("label") == 1)).count()

        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": train_auc
        }


    # Calculate and print metrics for training data
    print("\nCalculating metrics...")
    train_metrics = calculate_metrics(train_predictions)
    print("\nTraining Data Performance Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")


    # Visualize ROC curve for training data
    def plot_roc_curve(predictions_df, title_prefix="Training"):
        # Convert to pandas in batches to save memory
        batch_size = 10000
        total_rows = predictions_df.count()

        # Get all predictions in one batch - simpler approach to avoid concatenation issues
        predictions_sample = predictions_df.select("label", "probability").sample(False, min(1.0, 10000.0 / total_rows),
                                                                                  seed=42)
        predictions_pd = predictions_sample.toPandas()

        # Extract probability scores
        y_true = predictions_pd["label"]
        y_score = predictions_pd["probability"].apply(lambda x: float(x[1]))

        # Calculate ROC curve points
        fpr, tpr, _ = roc_curve(y_true, y_score)

        # Calculate AUC
        roc_auc = auc(fpr, tpr)

        # Plot the curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title_prefix} Data ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'{title_prefix.lower()}_roc_curve.png')
        plt.close()


    # Plot ROC curve for training data
    print("\nPlotting ROC curve...")
    plot_roc_curve(train_predictions)

# Save test predictions to CSV in batches
print("\nSaving predictions...")
batch_size = 10000
total_rows = predictions.count()
for i in range(0, total_rows, batch_size):
    batch = predictions.select("id", "question1", "question2", "prediction", "probability") \
        .limit(batch_size).toPandas()
    mode = 'w' if i == 0 else 'a'
    header = i == 0
    batch.to_csv('test_predictions.csv', index=False, mode=mode, header=header)

print("\nProcess completed!")
print("- Check test_predictions.csv for test data predictions")
print("- Check training_roc_curve.png for the ROC curve visualization (if training data was evaluated)")

# Clean up
spark.stop()