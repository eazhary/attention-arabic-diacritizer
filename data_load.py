# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
#import regex
#import read
import glob

def load_vocab():
	characters = "E ًٌٍَُِْاإأآبتثجحخدذرزسشصضطظعغفقكلمنهويىؤءةئ." # Arabic character set
	char2idx = {char: idx for idx, char in enumerate(characters)}
	idx2char = {idx: char for idx, char in enumerate(characters)}
	return char2idx, idx2char

def create_data(): 
	char2idx, idx2char = load_vocab()
	characters = "E ًٌٍَُِْاإأآبتثجحخدذرزسشصضطظعغفقكلمنهويىؤءةئ." # Arabic character set
	files = glob.glob(hp.data_dir+'/*.txt',recursive=True)
	
	# Index
	x_list, y_list = [], []
	for file in files:
		print("Processing file ", file)
		with open(file,"r",encoding="utf-8") as f:
			for line in f:
				target,tlen,source,slen = line.split("|")
				#print(target,tlen)
				if (int(tlen)<hp.maxlen):
					#print("True")
					x = [char2idx[c] for c in (target+'E')]
					y = [char2idx[c] for c in (source+'E')]
					x_list.append(np.array(x))
					y_list.append(np.array(y))
				#break;	
		#break;	
	# Pad	   
	X = np.zeros([len(x_list), hp.maxlen], np.int32)
	Y = np.zeros([len(y_list), hp.maxlen], np.int32)
	print(X.shape)
	for i, (x, y) in enumerate(zip(x_list, y_list)):
		print(i)
		X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
		Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
	
	return X, Y

def load_train_data():
	#de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
	#en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
	
	X, Y=create_data()
	return X, Y
	
def load_test_data():
	def _refine(line):
		#line = regex.sub("<[^>]+>", "", line)
		#line = regex.sub("[^\s\p{Latin}']", "", line) 
		return line.strip()
	
	#de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
	#en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
		
	X, Y  = create_data()
	return X # (1064, 150)

def get_batch_data():
	# Load data
	#X, Y = load_train_data()
	X = np.load("source.npy")
	Y = np.load("target.npy")
	X=X[:2000]
	Y=Y[:2000]
	# calc total batch count
	num_batch = len(X) // hp.batch_size
	
	# Convert to tensor
	X = tf.convert_to_tensor(X, tf.int32)
	Y = tf.convert_to_tensor(Y, tf.int32)
	
	# Create Queues
	input_queues = tf.train.slice_input_producer([X, Y])
			
	# create batch queues
	x, y = tf.train.shuffle_batch(input_queues,
								num_threads=8,
								batch_size=hp.batch_size, 
								capacity=hp.batch_size*64,	 
								min_after_dequeue=hp.batch_size*32, 
								allow_smaller_final_batch=False)
	
	return x, y, num_batch # (N, T), (N, T), ()


#X,Y = load_train_data()
#print(X[0],Y[0])
#np.save("X.npy",X)
#np.save("Y.npy",Y)
