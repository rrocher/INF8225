import requests
import os.path
from argparse import ArgumentParser
from tempfile import mkstemp
from io import StringIO, BytesIO
from lxml import etree
import os
import re
import stat
import fileinput
import subprocess
import sys, traceback,logging
import json
import collections
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logs_path = './rnn_words'
writer = tf.summary.FileWriter(logs_path)

class LogLine(object):
	def __init__(self, msg_type=None, line_id = None, params = None):
		self.msg_type = msg_type
		self.line_id = line_id
		self.params = params

	def __repr__(self):
		if not self.params:
			msg2 = ''
		else:
			ll = [str(x) for x in self.params]
			msg2 = ' '.join(ll)
		return '('+str(self.line_id)+'-args['+msg2+'])'

class LCSObject(object):
	def __init__(self):
		self.lcs_seq = []
		self.log_lines = []

	def __repr__(self):
		msg1 = ' '.join(self.lcs_seq)
		ll = [str(x) for x in self.log_lines]
		msg2 = ','.join(ll)
		return '{"'+msg1+'"}\n'

def lcs(a,b):
	lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
	for i, x in enumerate(a):
		for j, y in enumerate(b):
			if x == y:
				lengths[i+1][j+1] = lengths[i][j] + 1
			else:
				lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
	result = []
	rlen = 0
	x,y = len(a),len(b)
	params_for_a = []
	params_for_b = []
	resx = True
	if x < y:
		resx = False
	while x != 0 and y != 0:
		if lengths[x][y] == lengths[x-1][y]:
			x -= 1
			if resx:
				result.insert(0,'*')
			params_for_a.insert(0 , a[x])
			if lengths[x][y] == 0:
				params_for_b.insert(0, b[y-1])
		elif lengths[x][y] == lengths[x][y-1]:
			y -= 1
			if resx is False:
				result.insert(0,'*')
			params_for_b.insert(0,b[y])
			if lengths[x][y] == 0:
				params-for_a.insert(0, b[x-1])
		else:
			assert a[x-1] == b[y-1]
			result.insert(0, a[x-1])
			rlen = rlen + 1
			x -= 1
			y -= 1
	return rlen, result, params_for_a, params_for_b
		

def checkParams(paramsL):
	for param in paramsL:
		if param == '*':
			return False
	return True

def search(lmap, seq, seqLine):
	maxLen = 0
	maxLiPos = -1
	lcsObjFound = None
	finalLcs = None
	finalParamsA = None
	finalParamsB = None
	if len(lmap) == 0:
		return None

	for pos, lcsObj in enumerate(lmap):
		tmpLen, tmp, params_for_a, params_for_b = lcs(lcsObj.lcs_seq, seq)
		if tmpLen > maxLen:
			maxLen = tmpLen
			maxLiPos= pos
			lcsObjFound = lcsObj
			finalLcs = tmp
			finalParamsA = params_for_a
			finalParamsB = params_for_b
	if maxLen > (len(seq)/2):
		lmap[maxLiPos].lcs_seq = finalLcs
		if checkParams(finalParamsA):
			for n, t in enumerate(lmap[maxLiPos].log_lines):
				if t.params is None or len(t.params) == 0:
					t.params = finalParamsA
		lmap[maxLiPos].log_lines.append(LogLine(finalLcs, seqLine+1, finalParamsB))
		return lcsObjFound
	return None

def processMsg(msg, lmap, pos):
    seq = msg.split()
    rc = search(lmap, seq, pos)
    if rc is not None:
        return rc
    lcsObj = LCSObject()
    lcsObj.lcs_seq = seq
    lcsObj.log_lines.append(LogLine(seq, pos+1))
    lmap.append(lcsObj)
    return lcsObj


def readFile(filename):
	with open(filename, 'r', encoding = 'utf-8') as f:
		content = f.readlines()
	return content

def runXmlLcs(lmap, content, msgattrib):
    sequences = []
    for pos, line in enumerate(content):
        root = etree.fromstring(line)
        msg = root.attrib[msgattrib]
        seq = processMsg(msg, lmap, pos)
        sequences.append(seq)
    return lmap, sequences

def runTextLogLcs(lmap, content, offset):
    sequences = []
    for pos, line in enumerate(content):
        ll = line.split()
        msg = ' '.join(ll[offset:]) 
        seq = processMsg(msg, lmap, pos)
        sequences.append(seq)
    return lmap, sequences

def mainProcess(filename, logtype, offset=None, msgattrib=None):
    lcsmap = []
    filecontent = readFile(filename)
    if logtype == 'text':
        lmap, sequences = runTextLogLcs(lcsmap, filecontent, offset)
    elif logtype == 'xml':
         lmap, sequences = runXmlLcs(lcsmap, filecontent, msgattrib)
    return lcsmap, sequences


def testMainProcess(lcsmap, filename, logtype, offset=None, msgattrib=None):
    filecontent = readFile(filename)
    if logtype == 'text':
        lmap, sequences = runTextLogLcs(lcsmap, filecontent, offset)
    elif logtype == 'xml':
         lmap, sequences = runXmlLcs(lcsmap, filecontent, msgattrib)
    return lcsmap, sequences

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

# number of units in RNN cell
n_hidden = 512

def RNN(x, weights, biases):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    x = tf.split(x, n_input,1)

    # 3-layer LSTM with n_hidden units.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

#tf Graph input
def train(x, y,training_data, dictionary, reverse_dictionary):
    vocab_size = len(dictionary)
    
    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]))
    }
    pred = RNN(x, weights, biases)

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    avg_losses = []
    avg_accuracies = []
    # Initializing the variables
    init = tf.global_variables_initializer()
    # Launch the graph
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0, n_input+1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)

        while step < training_iters:
            # Generate a minibatch. Add some randomness on selection process.
            if offset > (len(training_data)-end_offset):
                offset = random.randint(0, n_input+1)
            symbols_in_keys = [ [dictionary[str(training_data[i])]] for i in range(offset, offset+n_input) ]
            
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
            
            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
            
            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                    feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
            loss_total += loss
            acc_total += acc
            if (step+1) % display_step == 0:
                avg_losses.append(loss_total)
                avg_accuracies.append(acc_total)
                acc_total = 0
                loss_total = 0
                symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                symbols_out = training_data[offset + n_input]
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            step += 1
            offset += (n_input+1)
        plt.figure("Training_1")
        plt.subplot(211)
        plt.plot(avg_losses)
        plt.title('Losses')

        plt.subplot(212)
        plt.plot(avg_accuracies)
        plt.title('Accuracies')
        plt.savefig("./Training_1")
        print("Optimization Finished!")
        return pred
    
def test(x, pred, loginputs, dictionary, reverse_dictionary):
    # print("Test***")
    # print(x)
    # print(y)
    possible_logs = []
    # Initializing the variables
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        if len(loginputs) != n_input:
            print("Invalid")
            return
        symbols_in_keys = []
        for i in range(len(loginputs)):
            if str(loginputs[i]) in dictionary:
                symbols_in_keys.append(dictionary[str(loginputs[i])]) 
            else:
                symbols_in_keys.append(0)
                 
        keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        onehot_pred = session.run(pred, feed_dict={x: keys})
        tops = tf.nn.top_k(onehot_pred,10).indices.eval()[0]
        for idx in tops:
            onehot_pred_index = int(idx)
            if onehot_pred_index in reverse_dictionary:
                possible_logs.append(reverse_dictionary[onehot_pred_index])
        return possible_logs
        
        
if __name__ == '__main__':
    lcsmap, sequences = mainProcess('hadoop.log', 'text',offset=2 )
    lcsseqs  = [ ' '.join(x.lcs_seq) for x in sequences]
    dictionary, reverse_dictionary = build_dataset(lcsseqs)
    x = tf.placeholder("float", [None, n_input, 1])

    y = tf.placeholder("float", [None, len(dictionary)])

    pred = train(x,y,lcsseqs, dictionary, reverse_dictionary)
    lcsmap, testsequences = testMainProcess(lcsmap, 'hadoop.test.log', 'text',offset=2 )
    
            
    lcstestseqs  = [ ' '.join(x.lcs_seq) for x in testsequences]
    logs = []
    anomalies = 0
    successes = 0
    for l1,l2,l3,totest in zip(*[iter(lcstestseqs)]*4):
        logs = []
        logs.append(l1)
        logs.append(l2)
        logs.append(l3)
        possible_logs = test(x, pred, logs, dictionary, reverse_dictionary)
        print("Possible logs:", possible_logs)
        print("To test:", totest)
        if possible_logs is None or len(possible_logs) == 0:
            continue
        if totest not in possible_logs:
            print("Anomaly:",totest)
            anomalies = anomalies +1
        else:
            successes = successes +1
    print("Anomalies:", anomalies)
    print("Successes:", successes)
