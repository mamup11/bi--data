from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover

sc = SparkContext('local')
spark = SparkSession(sc)

documents = sc.wholeTextFiles("hdfs:///user/mmurill5/datasets/miniGutenberg")
db =  StructType([StructField ("name" , StringType(), True) ,  StructField("text" , StringType(), True)])
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="tf", numFeatures=20)
idf = IDF(inputCol="tf", outputCol="features", minDocFreq=1)
kmeans = KMeans(k=2)

dataframe = spark.createDataFrame(documents,db)
words = tokenizer.transform(dataframe)
tf = hashingTF.transform(words)
tf.cache()
idfModel = idf.fit(tf)
final = idfModel.transform(tf)

kmeansModel = kmeans.fit(final)
predictionData = kmeansModel.transform(final)

predictionData.show()

predictionData.select("name", "prediction").show(20, False)