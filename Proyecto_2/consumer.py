from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import dayofweek
import ml_lib as ml

spark = SparkSession.builder.appName("StructuredNetworkWordCount").getOrCreate()

# Define schema of the csv
userSchema = StructType().add("year", "integer").add("month", "integer").add("day","integer").add("order","integer").add("country","integer").add("session ID","integer").add("page 1 (main category)","integer").add("page 2 (clothing model)","string").add("colour","integer").add("location","integer").add("model photography","integer").add("price","integer").add("price 2","integer").add("page","integer")

# Read CSV files from set path
df = spark.readStream.option("sep", ",").option("header", "true").schema(userSchema).csv("tmp/")

"""
df.withColumn("year",df.year.cast("int"))
df.withColumn("month",df.month.cast("int"))
df.withColumn("day",df.day.cast("int"))
df.withColumn("order",df.order.cast("int"))
df.withColumn("country",df.country.cast("int"))
df.withColumn("sessionID",df.sessionID.cast("int"))
df.withColumn("page1maincategory",df.page1maincategory.cast("int"))
df.withColumn("colour",df.colour.cast("int"))
df.withColumn("location",df.location.cast("int"))
df.withColumn("modelphotography",df.modelphotography.cast("int"))
df.withColumn("price",df.price.cast("int"))
df.withColumn("price2",df.price2.cast("int"))
df.withColumn("page",df.page.cast("int"))
"""

#df.printSchema()



def foreach_batch_function(df, epoch_id):
    # Transform and write batchDF
    print("------BATCH",epoch_id,"------")
    print((df.count(), len(df.columns)))
    ml.solve(df)
    print("-----------------------------")




 
#query = df.writeStream.outputMode("complete").format("console").start()
#df.createOrReplaceTempView("TAB")
#totalSalary = spark.sql("select count(*) from TAB")
#query = totalSalary.writeStream.outputMode("complete").format("console").start()
query = df.writeStream.foreachBatch(foreach_batch_function).start()
query.awaitTermination()
