from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import pandas as pd

# Start Spark session
spark = SparkSession.builder \
    .appName("Random Forest Binary Classification") \
    .getOrCreate()

# Load data (replace with the actual path to your data file)
df = spark.read.csv("hdfs://192.168.13.181:9000/user/sat3812/mental_health.csv", header=True, inferSchema=True)

# Drop rows with missing values
df = df.na.drop()

# Recode 'Mental_Health_Condition' as binary: 0 = 'None', 1 = 'Has Condition'
df = df.withColumn("Mental_Health_Binary", 
                   (df["Mental_Health_Condition"] != "None").cast("int"))

# String indexing for categorical columns
string_columns = ["Gender", "Job_Role", "Industry", "Work_Location", 
                  "Access_to_Mental_Health_Resources", "Region", 
                  "Stress_Level", "Productivity_Change", 
                  "Satisfaction_with_Remote_Work", "Physical_Activity", "Sleep_Quality"]

indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in string_columns]

# Feature columns
feature_columns = ["Age", "Years_of_Experience", "Hours_Worked_Per_Week", 
                   "Number_of_Virtual_Meetings", "Work_Life_Balance_Rating"] + [col + "_index" for col in string_columns]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Random Forest Classifier for Binary Classification
rf = RandomForestClassifier(labelCol="Mental_Health_Binary", featuresCol="features")

# Pipeline
pipeline = Pipeline(stages=indexers + [assembler, rf])

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=1234)

# Fit the model
model = pipeline.fit(train_df)

# Predictions
predictions = model.transform(test_df)

# Accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Mental_Health_Binary", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)

# Precision
precision_evaluator = MulticlassClassificationEvaluator(labelCol="Mental_Health_Binary", predictionCol="prediction", metricName="weightedPrecision")
precision = precision_evaluator.evaluate(predictions)

# Recall
recall_evaluator = MulticlassClassificationEvaluator(labelCol="Mental_Health_Binary", predictionCol="prediction", metricName="weightedRecall")
recall = recall_evaluator.evaluate(predictions)

# AUC
auc_evaluator = BinaryClassificationEvaluator(labelCol="Mental_Health_Binary", rawPredictionCol="prediction", metricName="areaUnderROC")
auc = auc_evaluator.evaluate(predictions)

# Print Results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"AUC: {auc:.2f}")

# Stop Spark session
spark.stop()
