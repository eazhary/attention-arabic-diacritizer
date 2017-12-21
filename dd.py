import tensorflow as tf
from hyperparams import Hyperparams as hp
import numpy as np


def load_vocab():
	characters = "PSE ًٌٍَُِْاإأآبتثجحخدذرزسشصضطظعغفقكلمنهويىؤءةئ." # Arabic character set
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
		#source = source[:180]
		#dest = dest[:180]
		return np.array(source, dtype=np.int32),np.array(dest, dtype=np.int32)
#	filenames = tf.gfile.Glob("data/*.txt")
	filenames = tf.gfile.Glob("data/emad.txt")
	dataset = tf.data.TextLineDataset(filenames)
	#dataset = dataset.shuffle(buffer_size=10000)
	dataset = dataset.map(lambda text: tuple(tf.py_func(mypyfunc, [text], [tf.int32, tf.int32])))
	dataset = dataset.filter(lambda x,y: tf.less_equal(tf.size(y),hp.maxlen))
	print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
	print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
	dataset = dataset.padded_batch(32,padded_shapes=([None],[None]))
	dataset = dataset.repeat()
	print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
	print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	return(next_element)

	
next_element = get_data()
with tf.Session() as sess:
	for i in range(3):
		x,y = sess.run(next_element)
		print("Source=",x[:,:40])
		print("Dest = ",y[:,:40])

