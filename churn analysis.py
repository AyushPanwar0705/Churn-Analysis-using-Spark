from pyspark import SparkContext, SparkConf
sc = SparkContext.getOrCreate()

# Step 2: Load necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 3: Check the information provided about data
# Assuming the information is already provided in the problem statement

# Step 4: Import the data files provided from HDFS (Churn.csv and Churntest.csv)
spark = SparkSession.builder.appName("Churn Analysis").getOrCreate()
churn_df = spark.read.csv("Churn.csv", header=True, inferSchema=True)
churntest_df = spark.read.csv("Churntest.csv", header=True, inferSchema=True)

# Step 5: Display the data in Spark Dataframe
churn_df.show()

# Step 6: Data pre-processing
# Convert categorical variables from integer to string
categorical_cols = ["VMail.Plan", "State", "Int.l.Plan"]
for col_name in categorical_cols:
    churn_df = churn_df.withColumn(col_name, col(col_name).cast("string"))

# Step 7: Exploratory data analysis
# 7.1 - Describe the data
churn_df.describe().show()

# 7.2 - Create Histogram for Day minutes spent by customers for churn=0 and 1 values
churn_df.groupBy("Churn").agg({"Day.Mins": "mean"}).show()

# 7.3 - Create count plots for Number of customers opt voicemail plan with Churn values
churn_df.groupBy("VMail.Plan", "Churn").count().show()

# 7.4 - Create count plots for International Plan opt by the customer with Churn values
churn_df.groupBy("Int.l.Plan", "Churn").count().show()

# 7.5 - Plot Area Wise churner and non-churner
churn_df.groupBy("State", "Churn").count().show()

# 7.6 - Get correlation matrix
correlation_matrix = churn_df.select([col(c).cast("float") for c in churn_df.columns]).corr()
correlation_matrix.show()

# Step 8: Get the correlation between Predicting Variable and independent variable
churn_correlation = correlation_matrix.select("Churn")
churn_correlation.show()

# Step 9: Applying Machine Learning Model
# 9.2 - Create vectors of all independent variables
feature_cols = churn_df.columns
feature_cols.remove("Churn")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
churn_df = assembler.transform(churn_df)

# 9.3 - Apply Decision Tree Classifier
dt = DecisionTreeClassifier(featuresCol="features", labelCol="Churn")
pipeline_dt = Pipeline(stages=[dt])
model_dt = pipeline_dt.fit(churn_df)

# 9.4 - Create a pipeline to build the classifier
pipeline_rf = Pipeline(stages=[rf])
model_rf = pipeline_rf.fit(churn_df)

# 9.5 - Use stratified sampling to get a sample of data
train_data, test_data = churn_df.randomSplit([0.7, 0.3], seed=42)

# 9.6 - Split the data into train and test dataset
dt_model = dt.fit(train_data)
rf_model = rf.fit(train_data)
gbt_model = gbt.fit(train_data)

# 9.7 - Make predictions and validate your model by calculating the accuracy score
predictions_dt = dt_model.transform(test_data)
predictions_rf = rf_model.transform(test_data)
predictions_gbt = gbt_model.transform(test_data)

# 9.8 - Calculate recall and precision score
evaluator = MulticlassClassificationEvaluator(labelCol="Churn", predictionCol="prediction", metricName="accuracy")
accuracy_dt = evaluator.evaluate(predictions_dt)
accuracy_rf = evaluator.evaluate(predictions_rf)
accuracy_gbt = evaluator.evaluate(predictions_gbt)

# 9.9 - Test the model using test data and calculate accuracy, recall, and precision
print("Decision Tree Accuracy:", accuracy_dt)
print("Random Forest Accuracy:", accuracy_rf)
print("Gradient Boosted Trees Accuracy:", accuracy_gbt)
