from pyspark.ml.feature import HashingTF as MLHashingTF
from pyspark.ml.feature import IDF as MLIDF
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import KMeans
import sys

#referencias: 
#https://spark.apache.org/docs/2.2.0/ml-pipeline.html
#https://stackoverflow.com/questions/35769489/adding-the-resulting-tfidf-calculation-to-the-dataframe-of-the-original-document
#https://spark.apache.org/docs/2.2.0/ml-features.html#tf-idf
if len(sys.argv) != 3:
    print("Error!")
    print("Usage: " + sys.argv[0] + " PATH " + "NumberOfClusters")
    exit(-1)

path = str(sys.argv[1])
cluster_number = int(sys.argv[2])

sc = SparkContext('local')
spark = SparkSession(sc)

#Ejemplo path= hdfs:///user/mmurill5/datasets/miniGutenberg
files = sc.wholeTextFiles(path)
documents = spark.createDataFrame(files, ["doc_id", "doc_text"])
#documents.printSchema()

df = (documents
  .rdd
  .map(lambda x : (x.doc_id,x.doc_text.split(" ")))
  .toDF()
  .withColumnRenamed("_1","doc_id")
  .withColumnRenamed("_2","text"))

htf = MLHashingTF(inputCol="text", outputCol="tf")
tf = htf.transform(df)
#tf.select("text", "tf").show()

idf = MLIDF(inputCol="tf", outputCol="features")
tfidf = idf.fit(tf).transform(tf)
#tfidf.select("tf", "features").show()

kmeans = KMeans(k=cluster_number)
kmeansModel = kmeans.fit(tfidf)
clusters = kmeansModel.transform(tfidf)
#clusters.show()
clusters.select("doc_id","prediction").show(100, False)