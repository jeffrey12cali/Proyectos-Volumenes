#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql.functions import dayofweek
import matplotlib.pyplot as plt
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


def split_by_week(df):
    max_day = df.select("day").rdd.max()[0]
    top = int(max_day*75)/100

    return (df.filter(df['day'] <= top),df.filter(df['day'] > top))

def solve(df):

    df.printSchema()
    # castear todos los numeros cargados como string a int
    for col in df.columns:
        if col!= "page 2 (clothing model)":
            df = df.withColumn(col,df[col].cast(IntegerType()))

    # borrar la columna año (todos son 2008)
    df = df.drop('year')

    # dataset shape
    print(df.count(),len(df.columns))

    # convertir atributo categorico a numerico
    indexer = StringIndexer(inputCol="page 2 (clothing model)", outputCol="page2_num")
    df = indexer.fit(df).transform(df)
    df = df.drop('page 2 (clothing model)')


    # borrar la corr > 0.9
    #df = df.drop('month')
    df = df.drop('session ID')


    # distribucion de las clases
    df.groupBy('model photography').count().show()

    # balanceo

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

    # Cuando el modelo de clasificación es binario pyspark requiere que los valores sean estrictamente 0 y 1
    #pasar 2 a 0
    df = combined_df.withColumnRenamed('model photography', 'model_photography')

    df = df.withColumn("model_photography",                              when(df["model_photography"] == 2,0).otherwise(df["model_photography"]))

    df.groupBy('model_photography').count().show()


    names = list()
    for elem in df.columns:
        if elem!="model_photography" and elem!="page2_num":
            names.append(elem)

    from pyspark.sql.functions import col, countDistinct 
    #df.agg(*(countDistinct(col(c)).alias(c) for c in df.columns)).show() 

    df = df.drop("page2_num") # borramos esta columna porque los modelos de arboles no soportan mas de 32 valores distintos

    # partimos por semanas cada batch
    splits = split_by_week(df)
    # parto en conjunto de prueba y entrenamiento
    train_df = splits[0] 
    test_df = splits[1]
    
    #convertir train_df y test_df en formato featuresCol and labelCOl ( usar vectorassembler)
    vectorAssembler = VectorAssembler(inputCols = names, outputCol = 'features')
    vhouse_train = vectorAssembler.transform(train_df)
    vhouse_test = vectorAssembler.transform(test_df)
    
    train_df = vhouse_train
    test_df = vhouse_test
    
    
    

    # Modelo de árbol de decisión
    print("---ÁRBOL DE DECISIÓN---")
    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'model_photography')
    dtModel = dt.fit(train_df)
    predictions = dtModel.transform(test_df)
    
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
    print()

    # Modelo de máquinas de vecrtores de soporte

    # se construye y entrena el modelo
    print("---MAQUINAS DE VECTORES DE SOPORTE---")
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
    print()

    print("---GRADIENT BOOSTED TREE---")
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
    print()

    
    print("---RANDOM FOREST---")
    # Modelo Random Forest
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
    print()
    
    





