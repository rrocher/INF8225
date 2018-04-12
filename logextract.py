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
		return
	lcsObj = LCSObject()
	lcsObj.lcs_seq = seq
	lcsObj.log_lines.append(LogLine(seq, pos+1))
	lmap.append(lcsObj)


def readFile(filename):
	with open(filename, 'r', encoding = 'utf-8') as f:
		content = f.readlines()
	return content

def runXmlLcs(lmap, content, msgattrib):
	for pos, line in enumerate(content):
		root = etree.fromstring(line)
		msg = root.attrib[msgattrib]
		processMsg(msg, lmap, pos)
	return lmap

def runTextLogLcs(lmap, content, offset):
	for pos, line in enumerate(content):
		ll = line.split()
		msg = ' '.join(ll[offset:]) 
		processMsg(msg, lmap, pos)
	return lmap


def main(filename, logtype, offset=None, msgattrib=None):
	lcsmap = []
	filecontent = readFile(filename)
	if logtype == 'text':
		runTextLogLcs(lcsmap, filecontent, offset)
	elif logtype == 'xml':
		runXmlLcs(lcsmap, filecontent, msgattrib)
	return lcsmap

if __name__ == '__main__':
	lcsmap = main('hadoop.log', 'text',offset=2 )
	print(lcsmap)


