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

	files = sc.wholeTextFiles(path)
```
Dandonos una variable documents con la siguiente estructura:

|name   |text   |
|---|---|
|documento1   |La política de estos días se centra en los esclavos, pero en estos días de tragedia es primordial que la política deje de centrarse en los esclavos y se centre en el comercio y los cultivos.   |
|documento2   |La ciencia de la computacion avanzó hasta puntos donde la etica nos hace cuestionarnos de lo que realmente significa ser humano, y si es posible que una maquina pueda llegar a convertirse en uno.   |
|...   |....   |
|documentoN   |....   |


Y una vez se tiene esta estructura, es facilmente almacenada en una base de datos hive; para esto primero se debe crear un dataframe con la estructura necesaria para almacenarlo y guardar en este el contenido de la variable RDD con los documentos. Justo despues se realiza una transformacion para separar las palabras del texto completo. De la siguiente manera:

```python
	documents = spark.createDataFrame(files, ["doc_id", "doc_text"])
    df = (documents
  		.rdd
 		.map(lambda x : (x.doc_id,x.doc_text.split(" ")))
  		.toDF()
  		.withColumnRenamed("_1","doc_id")
  		.withColumnRenamed("_2","text"))
```

### Calculo del vector TF-IDF

Para realizar el calculo del vector TF-IDF se utiliza una libreria que se encuentra incluida dentro de Apache Spark. Para el calculo de este vector es necesario realizar 2 acciones, que son: el calculo de TF (Term Frecuency) y el calculo del vector IDF (Inverse Document Frecuency) por separado.

> **Nota:** El vector TF-IDF realiza un calculo de peso para cada palabra, y a aquellas palabras que son repetidas una gran cantidad de veces y se encuentran en varios documentos les resta peso, con el objetivo de quitarle importancia a las palabas auxiliares que se usan en la lengua (StopWords), como lo es la palabra "the", "to" "for" y varias otras.

De la siguiente manera:

```python
	htf = MLHashingTF(inputCol="text", outputCol="tf")
	tf = htf.transform(df)

	idf = MLIDF(inputCol="tf", outputCol="features")
	tfidf = idf.fit(tf).transform(tf)
```
### Ejecucion del algoritmo Kmeans

Una vez se tiene calculado el vector tf-idf la ejecucion del algoritmo de kmeans se vuelve tan sencilla como ejecutar el algoritmo de la libreria nativa de spark dandole el dataframe previamente creado, y kmeans utilizara los datos de la columna features y añadira al dataframe retornado un columna llamada prediction en el que indicara el cluster al que pertenece cada documento.

La ejecucion se realiza de la siguiente manera:

```python
	kmeans = KMeans(k=cluster_number)
	kmeansModel = kmeans.fit(tfidf)
	clusters = kmeansModel.transform(tfidf)
```

--------------

# Ejecución

#### Serial
Para la ejecución del algoritmo serial se necesita cumplir con lo siguiente:

- Clonar el repositorio
- Tener Python 2.7 instalado

Una vez cumplidos los requisitos unicamente se debe ejecutar el siguiente comando:
```
$ sudo python ControllerSerial.py path clusters

path = Ubicación donde se encuentran los documentos a leer
clusters = Numero de clusters o conjuntos de documentos esperados
```


#### Spark
Para la ejecución del programa en spark se necesita cumplir con lo siguiente:

- Clonar el repositorio
- Tener el ambiente de Apache Spark correctamente configurado
- Tener pySpark correctamente funcionando
- Tener disponible al menos 20% del tamaño de todos los documentos a leer en memoria

Una vez cumplidos los requisitos unicamente se debe ejecutar el siguiente comando:
```
$  spark-submit --master yarn --deploy-mode client code.py path clusters

path = Ubicación donde se encuentran los documentos a leer
clusters = Numero de clusters o conjuntos de documentos esperados
```

---------------

