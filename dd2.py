"""
This script is used to test how to implement
a string -> list of ids using tensorflow,
instead of using python code to be able to
embedd the decoding into  the model
"""

import tensorflow as tf


def test():
	
	# First build the model

	# An input to take the entire sentence
	input_sentence = tf.placeholder(tf.string, name='user_input')

	input_sentence_array = tf.reshape(input_sentence, [1])
	

	# Split the input into tokens
	input_tokens = tf.string_split(input_sentence_array, delimiter='')

	mapping_strings = tf.constant(["P","E", "ا","ب"])
	#map = tf.reshape(mapping_strings, [1])
#	maps = tf.string_split(mapping_strings,delimiter='')
	
	table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, num_oov_buckets=0, default_value=-1)
	
	# map the tokens into IDs
	input_ids = table.lookup(input_tokens.values)


	# Now test the model:
	with tf.Session() as session:
		text_input = 'ابE'
		tf.tables_initializer().run(session=session)

		result = session.run(input_ids, feed_dict={input_sentence: text_input})

		print("Result:", result)

if __name__ == '__main__':
	test()