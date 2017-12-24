# -*- coding: utf-8 -*-

import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
#from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph, load_vocab
#from nltk.translate.bleu_score import corpus_bleu

def eval(): 
	# Load graph
	g = Graph(is_training=False)
	print("Graph loaded")
	
	# Load data
	#X, Sources, Targets = load_test_data()
	char2idx, idx2char = load_vocab()
	inp = "بسم الله الرحمن الرحيم"
	x = [char2idx[c] for c in inp+'E']
	x += [0]*(hp.maxlen-len(x))
	x = np.array(x)
	x = x.reshape(1,-1)
#	x = x.repeat(hp.batch_size,axis=0)
#	  X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
	 
	# Start session			
	with g.graph.as_default():	  
		sv = tf.train.Supervisor()
		with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			## Restore parameters
			sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
			print("Restored!")
			  
			## Get model name
			mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
			 
			## Inference

			#preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
			preds = np.zeros((1, hp.maxlen), np.int32)
			for j in range(hp.maxlen):
				_preds = sess.run(g.preds, {g.x: x, g.y: preds})
				preds[:, j] = _preds[:, j]
			got = "".join(idx2char[idx] for idx in preds[0]).split("E")[0].strip()
			print("Source: ",inp)
			print("got : ", got)
										  
if __name__ == '__main__':
	eval()
	print("Done")
	
	
