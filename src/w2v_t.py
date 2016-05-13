#!/usr/bin/python3
#encoding=utf-8

from gensim.models import Word2Vec
import os
import scipy
#from word2vec import Word2Vec

sentences = "today is a good day"
model=Word2Vec(sentences,min_count=0,size=100,workers=8)
print(model['a'])