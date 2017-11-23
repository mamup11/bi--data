# Topicos de Télematica (Big Data)

By: 

 * Mariana Narvaez Berrio - mnarvae3@eafit.edu.co 
 * Mateo Murillo Penagos - mmurill5@eafit.edu.co 

Big Data
==================

Esta es la entrega final de la cuarta practica de Tópicos de Telematica (**BIG DATA**); este proyecto tiene como finalidad solucionar 2 problemas.

* Problema 1: Es necesario comparar 2 documentos entre si y sacar un valor para saber que tan similares son estos documentos.
* Problema 2: Es necesario agrupar los documentos leídos en grupos según el tema que se trate en cada documento (Agrupar los documentos que probablemente hablen del mimo tema)

Este desarrollo esta realizado en python, usando el ambiente de Apache Spark para procesar paralelamente varios documentos y reducir en gran medida el tiempo de analisis de los mismos.
También se presenta una solución serial al mismo problema, el cual realiza todos los calculos corriendo en un único núcleo.

---------

## Información

El codigo se divide en 2 secciones, primero los transformadores y por ultimo los estimadores (Acciones). Dentro de esta estructura tambien se puede separa el algoritmo en 3 partes, una parte inicial donde se leen los documentos y se almacenan en una base de datos, el calculo del vector TF-IDF y por ultimo la ejecucion del algoritmo de kmeans, siendo todas estas funciones dadas por las librerias de Spark.

### Lectura de documentos

La lectura de documentos se realiza con las funciones nativas de Spark y la variable de contexto del mismo, solicitandole que lea un path en el que encontrara muchos documentos y almacenando todos estos documentos en una variable RDD, en un formato en el que me de la direccion del archivo y el texto que este contiene, de la siguiente manera:

```python
sc = SparkContext('local')
spark = SparkSession(sc)

documents = sc.wholeTextFiles(path)
```
Dandonos una variable documents con la siguiente estructura:
name	|	text 
------------|------------
documento1 	|	La política de estos días se centra en los esclavos, pero en estos días de tragedia es primordial que la política deje de centrarse en los esclavos y se centre en el comercio y los cultivos. 
documento2		|	La ciencia de la computacion avanzó hasta puntos donde la etica nos hace cuestionarnos de lo que realmente significa ser humano, y si es posible que una maquina pueda llegar a convertirse en uno. 
...           |    ....
documentoN    |    ....

Y una vez se tiene esta estructura, es facilmente almacenada en una base de datos hive; para esto primero se debe crear un dataframe con la estructura necesaria para almacenarlo y guardar en este el contenido de la variable RDD con los documentos. De la siguiente manera:

```python
schema =  StructType([StructField ("path" , StringType(), True) ,  StructField("text" , StringType(), True)])

...

docDataFrame = spark.createDataFrame(documents,schema)
```

### Calculo del vector TF-IDF

Para realizar el calculo del vector TF-IDF se utiliza una libreria que se encuentra incluida dentro de Apache Spark. Para el calculo de este vector es necesario realizar 2 acciones, que son: el calculo de TF (Term Frecuency) y el calculo del vector IDF (Inverse Document Frecuency) por separado, pero antes de poder calcular estos es necesario dividir la entrada que tenemos la cual es un String, en una variable iterable que este compuesta por las palabras que conforman el texto.

> **Nota:** El vector TF-IDF realiza un calculo de peso para cada palabra, y a aquellas palabras que son repetidas una gran cantidad de veces y se encuentran en varios documentos les resta peso, con el objetivo de quitarle importancia a las palabas auxiliares que se usan en la lengua (StopWords), como lo es la palabra "the", "to" "for" y varias otras.

De la siguiente manera:

```python
tokenizer = Tokenizer(inputCol="text", outputCol="terms")
hashingTF = HashingTF(inputCol="filtered terms", outputCol="rawFeatures", numFeatures=20)

... 

wordsData = tokenizer.transform(docDataFrame)
featurizedData = hashingTF.transform(filteredData)
```