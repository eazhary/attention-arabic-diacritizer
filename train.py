# -*- coding: utf-8 -*-
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from modules import *
import os, codecs
import numpy as np

def load_vocab():
	characters = "PSEاإأآبتثجحخدذرزسشصضطظعغفقكلمنهويىؤءةئ ًٌٍَُِّْ،." # Arabic character set
	char2idx = {char: idx for idx, char in enumerate(characters)}
	idx2char = {idx: char for idx, char in enumerate(characters)}
	return char2idx, idx2char
	
def get_data():
	def mypyfunc(text):
		text = text.decode("utf-8")
		items = text.split("|")
		char2idx,_=load_vocab()
		source = [char2idx[c] for c in items[2]+'E']
		dest = [char2idx[c] for c in items[0]+'E']
		source += [0]*(hp.maxlen-len(source))
		dest += [0]*(hp.maxlen-len(dest))
		return np.array(source, dtype=np.int32),np.array(dest, dtype=np.int32)
	filenames = tf.gfile.Glob("data/*.txt")
	dataset = tf.data.TextLineDataset(filenames)
	dataset = dataset.map(lambda text: tuple(tf.py_func(mypyfunc, [text], [tf.int32, tf.int32])))
	dataset = dataset.filter(lambda x,y: tf.less_equal(tf.size(y),hp.maxlen))
	dataset = dataset.batch(hp.batch_size)
	dataset = dataset.repeat()
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	return(next_element)

class Graph():
	def __init__(self, is_training=True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			if is_training:
				self.x, self.y = get_data() # (N, T)
				self.x = tf.reshape(self.x,shape=[-1,hp.maxlen])
				self.y = tf.reshape(self.y,shape=[-1,hp.maxlen])
			else: # inference
				self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
				self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

			# define decoder inputs
			self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*1, self.y[:, :-1]), -1) # S+output (shifted)
			char2idx, idx2char = load_vocab()
			self.lookup_table = tf.get_variable('lookup_table',         # [50,512]
                                       dtype=tf.float32,
                                       shape=[len(char2idx), hp.hidden_units],
                                       initializer=tf.contrib.layers.xavier_initializer())

			# Load vocabulary	 

			# Encoder
			with tf.variable_scope("encoder"):
				## Embedding
				self.enc = embeddings(self.x, 
									  vocab_size=len(char2idx), 
									  num_units=hp.hidden_units, 
									  scale=True,
									  lookup = self.lookup_table,
									  scope="embed")
				
				## Positional Encoding
				if hp.sinusoid:
					self.enc += positional_encoding(self.x,
									  num_units=hp.hidden_units, 
									  zero_pad=False, 
									  scale=False,
									  scope="enc_pe")
				else:
					self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
									  vocab_size=hp.maxlen, 
									  num_units=hp.hidden_units, 
									  zero_pad=False, 
									  scale=False,
									  scope="enc_pe")
					
				 
				## Dropout
				self.enc = tf.layers.dropout(self.enc, 
											rate=hp.dropout_rate, 
											training=tf.convert_to_tensor(is_training))
				## Blocks
				for i in range(hp.num_blocks):
					with tf.variable_scope("num_blocks_{}".format(i)):
						### Multihead Attention
						self.enc = multihead_attention(queries=self.enc, 
														keys=self.enc, 
														num_units=hp.hidden_units, 
														num_heads=hp.num_heads, 
														dropout_rate=hp.dropout_rate,
														is_training=is_training,
														causality=False)
						
						### Feed Forward
						self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])
			
			# Decoder
			with tf.variable_scope("decoder"):
				## Embedding
				self.dec = embeddings(self.decoder_inputs, 
									  vocab_size=len(char2idx), 
									  num_units=hp.hidden_units,
									  scale=True,
									  lookup = self.lookup_table,
									  #reuse=True, 
									  scope="embed")
				
				## Positional Encoding
				if hp.sinusoid:
					self.dec += positional_encoding(self.decoder_inputs,
									  vocab_size=hp.maxlen, 
									  num_units=hp.hidden_units, 
									  zero_pad=False, 
									  scale=False,
									  scope="dec_pe")
				else:
					self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
									  vocab_size=hp.maxlen, 
									  num_units=hp.hidden_units, 
									  zero_pad=False, 
									  scale=False,
									  scope="dec_pe")
				
				## Dropout
				self.dec = tf.layers.dropout(self.dec, 
											rate=hp.dropout_rate, 
											training=tf.convert_to_tensor(is_training))
				
				## Blocks
				for i in range(hp.num_blocks):
					with tf.variable_scope("num_blocks_{}".format(i)):
						## Multihead Attention ( self-attention)
						self.dec = multihead_attention(queries=self.dec, 
														keys=self.dec, 
														num_units=hp.hidden_units, 
														num_heads=hp.num_heads, 
														dropout_rate=hp.dropout_rate,
														is_training=is_training,
														causality=True, 
														scope="self_attention")
						
						## Multihead Attention ( vanilla attention)
						self.dec = multihead_attention(queries=self.dec, 
														keys=self.enc, 
														num_units=hp.hidden_units, 
														num_heads=hp.num_heads,
														dropout_rate=hp.dropout_rate,
														is_training=is_training, 
														causality=False,
														scope="vanilla_attention")
						
						## Feed Forward
						self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
				
			# Final linear projection
			#self.dec = (batch,180,512)
			self.logits = tf.layers.dense(self.dec, len(char2idx)) # (batch,180,50)
			self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1)) # (batch,180)
			self.istarget = tf.to_float(tf.not_equal(self.y, 0)) # (batch,180) (1,1,1,1,0,0,0,0,0,0,0,0,0,0)
			self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
			tf.summary.scalar('acc', self.acc)
				
			if is_training:	 
				# Loss
				self.global_step = tf.Variable(0, name='global_step', trainable=False)
				self.learning_rate = _learning_rate_decay(self.global_step)
				self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(char2idx))) # (batch,180,50)
				self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed) # (batch,180)
				self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget)) # scalar
			   
				# Training Scheme
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
				self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
				   
				# Summary 
				tf.summary.scalar('mean_loss', self.mean_loss)
				tf.summary.scalar('learning_rate', self.learning_rate)
				self.merged = tf.summary.merge_all()

				

def _learning_rate_decay(global_step):
  # Noam scheme from tensor2tensor:
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return hp.hidden_units**-0.5 * tf.minimum(step * hp.warmup_steps**-1.5, step**-0.5)
				
if __name__ == '__main__':				  
	# Load vocabulary	 
	#de2idx, idx2de = load_de_vocab()
	#en2idx, idx2en = load_en_vocab()
	
	# Construct graph
	g = Graph("train"); print("Graph loaded Suc")
	
	# Start session
	sv = tf.train.Supervisor(graph=g.graph, 
							 logdir=hp.logdir,)
							 #save_model_secs=0)
	with sv.managed_session() as sess:
		while not sv.should_stop(): 
#			for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
			gs,acc,mean,ops=sess.run([g.global_step,g.acc,g.mean_loss,g.train_op])
			message = "Step %-7d : acc=%.05f mean=%.05f" % (gs,acc,mean)
			print(message)
			#gs = sess.run(g.global_step)   
			#sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
	
	print("Done")	 
	

