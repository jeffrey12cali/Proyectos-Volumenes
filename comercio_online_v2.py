#!/usr/bin/env python
# coding: utf-8

# In[143]:


from pyspark.sql.functions import dayofweek
#import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col,when,count
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType,DoubleType
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import SparkConf, SparkContext
import os
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.mllib.stat import Statistics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import HashingTF,Tokenizer,IDF
import pyspark.sql.types as T
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.functions import col, explode, array, lit
from pyspark.ml.classification import LinearSVC
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.sql.functions import col, explode, array, lit
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics


# In[110]:


# mostrar el esquema
spark = SparkSession.builder.appName('ml-eshop').getOrCreate()
df = spark.read.format("csv").option("delimiter", ";").option("header",True).load("dataset.csv")
df.printSchema()


# In[111]:


# visualización previa con pandas
def head(df):
    return pd.DataFrame(df.take(10), columns = df.columns)
head(df)


# In[112]:


# atributos
df.columns


# In[113]:


# hay datos faltantes?
#df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show() 


# In[114]:


# tipos de datos cargados
df.dtypes


# In[115]:


# castear todos los numeros cargados como string a int
for col in df.columns:
    if col!= "page 2 (clothing model)":
        df = df.withColumn(col,df[col].cast(IntegerType()))


# In[116]:


# verificacion de la conversion
df.dtypes


# In[117]:


# borrar la columna año (todos son 2008)
df = df.drop('year')
head(df)


# In[118]:


# dataset shape
print(df.count(),len(df.columns))


# In[119]:


df.describe().toPandas()


# In[120]:


# convertir atributo categorico a numerico
indexer = StringIndexer(inputCol="page 2 (clothing model)", outputCol="page2_num")
df = indexer.fit(df).transform(df)
df = df.drop('page 2 (clothing model)')
head(df)


# In[121]:


# get correlation matrix
#import seaborn as sns
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df.columns, outputCol = vector_col)
df_vector = assembler.transform(df).select(vector_col)

matrix = Correlation.corr(df_vector, vector_col)
corr_matrix = list(matrix.collect()[0]["pearson({})".format(vector_col)].values)
#print(arr)
row_N,col_N,i,j = len(corr_matrix),len(df.columns),0,0
matrix_corr = []
while i < row_N:
    j = 0
    tmp = []
    while j < col_N:
        #print(corr_matrix[i].round(2),end = "\n" if j+1 == col_N else " ")
        tmp.append(corr_matrix[i])
        i += 1
        j += 1
    matrix_corr.append(list(tmp))
    #i += 1
#plt.figure(figsize = (13,13))
#ax = sns.heatmap(matrix_corr, square=True, annot=True)


# In[122]:


# borrar la corr > 0.9
#df = df.drop('month')
df = df.drop('session ID')
head(df)


# In[123]:


df.dtypes


# In[124]:


vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df.columns, outputCol = vector_col)
df_vector = assembler.transform(df).select(vector_col)

matrix = Correlation.corr(df_vector, vector_col)
corr_matrix = list(matrix.collect()[0]["pearson({})".format(vector_col)].values)
#print(arr)
row_N,col_N,i,j = len(corr_matrix),len(df.columns),0,0
matrix_corr = []
while i < row_N:
    j = 0
    tmp = []
    while j < col_N:
        #print(corr_matrix[i].round(2),end = "\n" if j+1 == col_N else " ")
        tmp.append(corr_matrix[i])
        i += 1
        j += 1
    matrix_corr.append(list(tmp))
    #i += 1
#plt.figure(figsize = (13,13))
#ax = sns.heatmap(matrix_corr, square=True, annot=True)


# In[125]:


labeled = df.groupby("price").count()
labeled.show()


# In[126]:


# selecciono los nombres de las columnas sin el target
names = list()
for i in df.columns:
    if i!="price":
        names.append(i)


# In[127]:


# convierto a vectorAssembler para poder usar el modelo
vectorAssembler = VectorAssembler(inputCols = names, outputCol = 'features')
vhouse_df = vectorAssembler.transform(df)
vhouse_df = vhouse_df.select(['features', 'price'])


splits = vhouse_df.randomSplit([0.7, 0.3])


# In[128]:


# parto en conjunto de prueba y entrenamiento
train_df = splits[0] 
test_df = splits[1] 
train_df.show()
test_df.show()


# # Modelos de regresión sobre el precio de los artículos

# # Modelo de Regresion Lineal

# In[129]:


# se crea y entrena el modelo
lr = LinearRegression(featuresCol = 'features', labelCol='price', maxIter=100, regParam=0.2, elasticNetParam = 0.2)
lr_model = lr.fit(train_df)

# ahora se pueden hacer algunas predicciones y evaluar el desempeño 
lr_predictions = lr_model.transform(test_df)
test_prediction = lr_predictions.select("prediction","price")

test_prediction.show()

evaluator = RegressionEvaluator(labelCol="price")

print("R Squared (R2) on test data = %g" % evaluator.evaluate(test_prediction, {evaluator.metricName: "r2"}))
print("Root Mean Squared Error (RMSE) on test data = %g" % evaluator.evaluate(test_prediction, {evaluator.metricName: "rmse"}))


# # Regresión Lineal General (Gamma)

# In[130]:


# se crea y entrena el modelo
glr = GeneralizedLinearRegression(featuresCol = 'features', labelCol='price',family="Gamma", link="identity", maxIter=10, regParam=0.3)

# Fit the model
model = glr.fit(train_df)

# ahora se pueden hacer algunas predicciones y evaluar el rendimiento 
lr_predictions = model.transform(test_df)
test_prediction_gamma = lr_predictions.select("prediction","price")

test_prediction_gamma.show()

evaluator = RegressionEvaluator(labelCol="price")

print("R Squared (R2) on test data = %g" % evaluator.evaluate(test_prediction_gamma, {evaluator.metricName: "r2"}))
print("Root Mean Squared Error (RMSE) on test data = %g" % evaluator.evaluate(test_prediction_gamma, {evaluator.metricName: "rmse"}))


# # Random Forest Regressor

# In[134]:


# se crea y entrena el modelo
rf = RandomForestRegressor(labelCol="price", maxBins=217)
model = rf.fit(train_df)

rf_prediction = model.transform(test_df)
test_prediction_rf = rf_prediction.select("prediction","price")

rf_prediction.show()

evaluator = RegressionEvaluator(labelCol="price")

print("R Squared (R2) on test data = %g" % evaluator.evaluate(test_prediction_rf, {evaluator.metricName: "r2"}))
print("Root Mean Squared Error (RMSE) on test data = %g" % evaluator.evaluate(test_prediction_rf, {evaluator.metricName: "rmse"}))


# # Modelos de clasificación sobre el tipo de foto de la modelo

# In[135]:


head(df)


# In[136]:


# distribucion de las clases
df.groupBy('model photography').count().show()


# # Es necesario balancear ( usaremos oversampling )

# In[137]:


from pyspark.sql.functions import col, explode, array, lit
major_df = df.filter(col("model photography") == 1)
minor_df = df.filter(col("model photography") == 2)
ratio = int(major_df.count()/minor_df.count())
print("ratio: {}".format(ratio))

a = range(ratio+1)
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows 
combined_df = major_df.unionAll(oversampled_df)
combined_df.groupBy('model photography').count().show()


# # Cuando el modelo de clasificación es binario pyspark requiere que los valores sean estrictamente 0 y 1

# In[138]:


#pasar 2 a 0
df = combined_df.withColumnRenamed('model photography', 'model_photography')

df = df.withColumn("model_photography",                          when(df["model_photography"] == 2,0).otherwise(df["model_photography"]))

df.groupBy('model_photography').count().show()


# In[142]:


names = list()
for elem in df.columns:
    if elem!="model_photography" and elem!="page2_num":
        names.append(elem)
    
from pyspark.sql.functions import col, countDistinct 
#df.agg(*(countDistinct(col(c)).alias(c) for c in df.columns)).show() 

df = df.drop("page2_num") # borramos esta columna porque los modelos de arboles no soportan mas de 32 valores distintos
#df.printSchema()


# convierto a vectorAssembler para poder usar el modelo
vectorAssembler = VectorAssembler(inputCols = names, outputCol = 'features')
vhouse_df = vectorAssembler.transform(df)
vhouse_df = vhouse_df.select(['features', 'model_photography'])
vhouse_df.groupBy("model_photography").count().show()

splits = vhouse_df.randomSplit([0.7, 0.3])
# parto en conjunto de prueba y entrenamiento
train_df = splits[0] 
test_df = splits[1] 


# # Modelo de árbol de decisión

# In[154]:


dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'model_photography')
dtModel = dt.fit(train_df)
predictions = dtModel.transform(test_df)
predictions.show()
#predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

#https://stackoverflow.com/questions/58404845/confusion-matrix-to-get-precsion-recall-f1score
#important: need to cast to float type, and order by prediction, else it won't work
preds_and_labels = predictions.select(['prediction','model_photography']).withColumn('model_photography', F.col('model_photography').cast(FloatType())).orderBy('prediction')

#select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','model_photography'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())

evaluator = BinaryClassificationEvaluator(labelCol = "model_photography")
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Precision: " + str(metrics.precision(1.0)))
print("Recall: " + str(metrics.recall(1.0)))
print("F1-Score: " + str(metrics.fMeasure(1.0)))


# # Modelo de máquinas de vecrtores de soporte

# In[155]:


# se construye y entrena el modelo
lsvc = LinearSVC(featuresCol = 'features', labelCol='model_photography',maxIter=10, regParam=0.1)

# Fit the model
lsvcModel = lsvc.fit(train_df)

# ahora se pueden hacer algunas predicciones y evaluar el rendimiento 
lsv_predictions = lsvcModel.transform(test_df)
test = test_df.rdd
# Instantiate metrics object

#important: need to cast to float type, and order by prediction, else it won't work
preds_and_labels = lsv_predictions.select(['prediction','model_photography']).withColumn('model_photography', F.col('model_photography').cast(FloatType())).orderBy('prediction')

#select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','model_photography'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())

evaluator = BinaryClassificationEvaluator(labelCol = "model_photography")
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Precision: " + str(metrics.precision(1.0)))
print("Recall: " + str(metrics.recall(1.0)))
print("F1-Score: " + str(metrics.fMeasure(1.0)))


# # Modelo Gradient-Boosted Tree

# In[156]:


gbt = GBTClassifier(featuresCol = 'features', labelCol='model_photography',maxIter=10)
gbtModel = gbt.fit(train_df)
predictions = gbtModel.transform(test_df)
#predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

#important: need to cast to float type, and order by prediction, else it won't work
preds_and_labels = predictions.select(['prediction','model_photography']).withColumn('model_photography', F.col('model_photography').cast(FloatType())).orderBy('prediction')

#select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','model_photography'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())

evaluator = BinaryClassificationEvaluator(labelCol = "model_photography")
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Precision: " + str(metrics.precision(1.0)))
print("Recall: " + str(metrics.recall(1.0)))
print("F1-Score: " + str(metrics.fMeasure(1.0)))


# # Modelo Random Forest

# In[157]:


rf = RandomForestClassifier(featuresCol = 'features', labelCol='model_photography')
rfModel = rf.fit(train_df)
predictions = rfModel.transform(test_df)
#predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

#important: need to cast to float type, and order by prediction, else it won't work
preds_and_labels = predictions.select(['prediction','model_photography']).withColumn('model_photography', F.col('model_photography').cast(FloatType())).orderBy('prediction')

#select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','model_photography'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())

evaluator = BinaryClassificationEvaluator(labelCol = "model_photography")
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Precision: " + str(metrics.precision(1.0)))
print("Recall: " + str(metrics.recall(1.0)))
print("F1-Score: " + str(metrics.fMeasure(1.0)))


# In[ ]:




