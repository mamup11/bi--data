# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from Jaccard import *
from kmeans import *
from Dictionary import *


timeIni = time.time()
words = {}
index = 0
words_map = []

if len(sys.argv) != 3:
    print "Error!"
    print "Usage: " + sys.argv[0] + " PATH " + "NumberOfClusters"
    exit(-1)

documents = str(sys.argv[1])
clusters = str(sys.argv[2])

for filename in os.listdir(documents):
    print filename
    stopWords = getstopwords()
    content = open(documents + filename, 'r')
    txt = content.read().lower()
    text = txt.split()
    uniques = []
    unique_words = []
    for word in text:
        if word not in stopWords and word not in unique_words:
            unique_words.append(word)
            if word in words:
                uniques.append(words.get(word))
            else:
                words.update({word: index})
                uniques.append(index)
                index += 1
    words_map.append(uniques)
print len(words)
print "*"*20
read_time = time.time()
print "Tiempo de lectura: " + str(read_time - timeIni)
print "Tiempo de ejecucion finalizando lectura: " + str(read_time - timeIni)
print "*"*20

#print len(words_map)
amount = len(words_map)
jaccard_matrix = np.empty((amount, amount), dtype=float)
for i in range(amount):
    for j in range(amount):
        if j < i:
            jaccard_matrix[i][j] = jaccard_matrix[j][i]
        else:
            if j == i:
                jaccard_matrix[i][j] = 0
            else:
                jaccard_matrix[i][j] = 1 - jaccard_similarity(words_map[i], words_map[j])

print "*"*20
jaccard_time = time.time()
print "Tiempo de jaccard: " + str(jaccard_time - read_time)
print "Tiempo de ejecucion finalizando jaccard: " + str(jaccard_time - timeIni)
print "*"*20

cents, c = kMeans(jaccard_matrix, int(clusters))
books = os.listdir(str(sys.argv[1]))
centroides = []
for i in range(int(clusters)):
    cluster = []
    centroides.append(cluster)
for i in range(len(c)):
    centroides[c[i]].append(books[i])

print centroides
print

print "*"*20
kMeans_time = time.time()
print "Tiempo de kMeans: " + str(kMeans_time - jaccard_time)
print "Tiempo de ejecucion total: " + str(kMeans_time - timeIni)
print "*"*20