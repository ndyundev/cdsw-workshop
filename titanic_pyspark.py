from pyspark.sql import SparkSession
import numpy as np

# Then call the `getOrCreate()` method of
# `SparkSession.builder` to connect to Spark. This
# example connects to Spark on YARN and gives a name to
# the Spark application:

spark = SparkSession.builder \
  .master("yarn") \
  .appName("titanic-pyspark") \
  .getOrCreate()
  
  
# Read the titanic dataset from HDFS. This data is in CSV
# format and includes a header row. Spark can infer the
# schema automatically from the data:

titanic_data = spark.read.csv("titanic/", header=True, inferSchema=True)

# The result is a Spark DataFrame named `titanic_data`.

# Print the number of rows:

titanic_data.count()

# Print the schema:

titanic_data.printSchema()

# Inspect one or more variables (columns):

titanic_data.describe("Fare").show()

# Print the first five rows:

titanic_data.limit(5).show()

# Or more concisely:

titanic_data.show(5)

# Print the first 20 rows (the default number is 20):

titanic_data.show()

# `show()` can cause rows to wrap onto multiple lines,
# making the output hard to read. To make the output
# more readable, use `toPandas()` to return a pandas
# DataFrame. For example, return the first five rows
# as a pandas DataFrame and display it


titanic_data_pd = titanic_data.limit(5).toPandas()
titanic_data_pd

# To display the pandas DataFrame in a scrollable
# grid, import pandas and set the pandas option
# `display.html.table_schema` to `True`:

import pandas as pd
pd.set_option("display.html.table_schema", True)

titanic_data_pd

# Caution: When working with a large Spark DataFrame,
# limit the number of rows before returning a pandas



# ### Using SQL Queries

# Instead of using Spark DataFrame methods, you can
# use a SQL query to achieve the same result.

# First you must create a temporary view with the
# DataFrame you want to query:

titanic_data.createOrReplaceTempView("titanic_data")

# Then you can use SQL to query the DataFrame:

spark.sql("""
  SELECT Pclass,
    COUNT(*) AS count,
    AVG(Fare) AS avg_fare
  FROM titanic_data
  WHERE Survived = 1
  GROUP BY Pclass
  ORDER BY avg_fare""").show()


# ### Visualizing Data from Spark

# You can create data visualizations in CDSW using Python
# plotting libraries such as Matplotlib.

# When using Matplotlib, you might need to first use this
# Jupyter magic command to ensure that the plots display
# properly in CDSW:

%matplotlib inline

# To visualize data from a Spark DataFrame with
# Matplotlib, you must first return the data as a pandas
# DataFrame.

# Caution: When working with a large Spark DataFrame,
# you might need to sample, filter, or aggregate before
# returning a pandas DataFrame.

# For example, you can select the columns from the dataset,
# randomly sample 5% of non-missing records, and return
# the result as a pandas DataFrame:

titanic_sample_pd = titanic_data \
  .select("Pclass", "Fare") \
  .dropna() \
  .sample(withReplacement=False, fraction=0.05) \
  .toPandas()

# Then you can create a scatterplot showing the
# relationship between columns:

titanic_sample_pd.plot.scatter(x="Pclass", y="Fare")






# ### Machine Learning with MLlib

# MLlib is Spark's machine learning library.

# As an example, let's examine the relationship between
# departure delay and arrival delay using a linear
# regression model.

# First, create a Spark DataFrame with only the relevant
# columns and with missing values removed:

titanic_to_model = titanic_data \
  .select("Survived","Pclass","SibSp","Parch", "Fare","male","Q","S") \
  .dropna()

# MLlib requires all predictor columns be combined into
# a single column of vectors. To do this, import and use
# the `VectorAssembler` feature transformer:

from pyspark.ml.feature import VectorAssembler

# In this example, there is only one predictor (input)
# variable: `dep_delay`.

assembler = VectorAssembler(inputCols=["Pclass","SibSp",'Parch', "Fare","male","Q","S"], outputCol="features")

# Use the `VectorAssembler` to assemble the data:

titanic_data_assembled = assembler.transform(titanic_to_model)
titanic_data_assembled.show(5)

# Randomly split the assembled data into a training
# sample (70% of records) and a test sample (30% of
# records):

(train, test) = titanic_data_assembled.randomSplit([0.7, 0.3])


test.show()

# Import and use `LinearRegression` to specify the linear
# regression model and fit it to the training sample:

from pyspark.ml.classification import LogisticRegression


lr = LogisticRegression(featuresCol="features", labelCol="Survived")

lr_model = lr.fit(train)

# Examine the model intercept and slope:

lr_model.intercept

lr_model.coefficients

# Evaluate the model on the test sample:
predictions = lr_model.transform(test)

testEntry = pd.DataFrame.from_records([{'Pclass': 3.0,'SibSp': 1.0,'Parch': 0.0,'Fare': 7.2292,'male':1.0,'Q':0.0,'S':0.0},
                                       {'Pclass': 1.0,'SibSp': 0.0,'Parch': 0.0,'Fare': 7.2292,'male':0.0,'Q':0.0,'S':0.0},
                                       {'Pclass': 3.0,'SibSp': 1.0,'Parch': 0.0,'Fare': 16.1,'male':0.0,'Q':0.0,'S':1.0},
                                       {'Pclass': 3.0,'SibSp': 0.0,'Parch': 0.0,'Fare': 8.0292,'male':0.0,'Q':0.0,'S':0.0}
                                      ])
#convert pandas df into spark df
testEntry = spark.createDataFrame(testEntry)

titanic_to_model.show()
testEntry.show()
testEntry_assembled = assembler.transform(testEntry)
testEntry_assembled.show(5)
predictions = lr_model.transform(testEntry_assembled)
predictions.select("prediction").show()
lr_summary = lr_model.evaluate(test)

# R-squared is the fraction of the variance in the test
# sample that is explained by the model:

lr_summary.pr.show()
lr_summary.areaUnderROC

# ### Cleanup

# Disconnect from Spark:

spark.stop()