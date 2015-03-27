import csv, exceptions, os, glob, copy
from scipy.special import erfc
import networkx as nx
import numpy as np
import math

ENABLE_DEL_OUTLIERS = False
THRESHOLD_OUTLIERS = 1000.0

#WSMD2013
#trueAnswers = [951000, 52.59, 10.35, 3796000]
#EXPERIMENT_PATH = 'wsdm/'

#SocialDIAG - Experiment 1
#trueAnswers = [58734670, 580, 69, 42.86]
#EXPERIMENT_PATH = 'diag/experiment_1/'

#SocialDIAG - Experiment 2
#trueAnswers = [951000, 792, 80, 21.01]
#EXPERIMENT_PATH = 'diag/experiment_2/'

#International Festival
#trueAnswers = [1.185806613, 471, 50, 21.01]
#EXPERIMENT_PATH = 'festival/'

#Priverno
#trueAnswers = [27.657346, 40, 16, 600]
#EXPERIMENT_PATH = 'priverno/'

#SocialDIAG2
trueAnswers = [36.5, 69, 100, 450]
EXPERIMENT_PATH = 'diag2/'


answersPre = {}
answersPost = {}
answersPreNorm = {}
answersPostNorm = {}
ANSWERS_PRE_FILE = 'answers_raw_pre.csv'
ANSWERS_POST_FILE = 'answers_raw_post.csv'

GRAPHS_DIR = EXPERIMENT_PATH + 'graphs/singles/'
OUTCOME_PATH = 'outcomes/'

def checkDir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
	return directory

def num(s):
	try:
		return int(s)
	except exceptions.ValueError:
		return float(s)

def division(x, y):
    if x % y != 0:
        return float(x) / float(y)
    else:
        return x / y


def chauvenet(data):				 # http://en.wikipedia.org/wiki/Chauvenet%27s_criterion, http://www.astro.rug.nl/software/kapteyn-beta/EXAMPLES/kmpfit_chauvenet.py
	mean = data.mean()				# Mean of incoming array y
	stdv = data.std()				# Its standard deviation
	N = len(data)					# Lenght of incoming arrays
	criterion = 1.0/(2*N)				# Chauvenet's criterion
	d = abs(data-mean)/stdv			# Distance of a value to mean in stdv's
	d /= 2.0**0.5					# The left and right tail threshold values
	prob = erfc(d)					# Area normal dist.    
	filter = prob >= criterion			# The 'accept' filter array with booleans
	return filter

def delOutliers():
	answersPreMatrix = np.concatenate((np.array([answersPre.keys()]).T, np.array(answersPre.values())), axis=1)
	answersPostMatrix = np.concatenate((np.array([answersPost.keys()]).T, np.array(answersPost.values())), axis=1)
	answerPre1Filter = chauvenet(answersPreMatrix[:, 1])
	answerPre2Filter = chauvenet(answersPreMatrix[:, 2])
	answerPre3Filter = chauvenet(answersPreMatrix[:, 3])
	answerPre4Filter = chauvenet(answersPreMatrix[:, 4])
	answerPost1Filter = chauvenet(answersPostMatrix[:, 1])
	answerPost2Filter = chauvenet(answersPostMatrix[:, 2])
	answerPost3Filter = chauvenet(answersPostMatrix[:, 3])
	answerPost4Filter = chauvenet(answersPostMatrix[:, 4])
	outliersIndexes = elementWiseAnd(answerPre1Filter, answerPre2Filter, answerPre3Filter, answerPre4Filter, answerPost1Filter, answerPost2Filter, answerPost3Filter, answerPost4Filter)

	count_outliers = 0
	for index, item in enumerate(outliersIndexes):
		if item == False:
			count_outliers += 1
			del answersPre[answersPreMatrix[index, 0]]
			del answersPost[answersPostMatrix[index, 0]]
			
	print 'Outliers rejected: ' + str(count_outliers) + ', remaining users: ' + str(len(answersPre))
"""

def outliersCriterion(data, correct_answer):
	filter = [x < (correct_answer * THRESHOLD_OUTLIERS) and x > (correct_answer / THRESHOLD_OUTLIERS) for x in data]
	return filter

def delOutliers():
	answersPreMatrix = np.concatenate((np.array([answersPre.keys()]).T, np.array(answersPre.values())), axis=1)
	answersPostMatrix = np.concatenate((np.array([answersPost.keys()]).T, np.array(answersPost.values())), axis=1)
#	print answersPreMatrix
#	print answersPostMatrix
	answerPre1Filter = outliersCriterion(answersPreMatrix[:, 1], trueAnswers[0])
	answerPre2Filter = outliersCriterion(answersPreMatrix[:, 2], trueAnswers[1])
	answerPre3Filter = outliersCriterion(answersPreMatrix[:, 3], trueAnswers[2])
	answerPre4Filter = outliersCriterion(answersPreMatrix[:, 4], trueAnswers[3])
	answerPost1Filter = outliersCriterion(answersPostMatrix[:, 1], trueAnswers[0])
	answerPost2Filter = outliersCriterion(answersPostMatrix[:, 2], trueAnswers[1])
	answerPost3Filter = outliersCriterion(answersPostMatrix[:, 3], trueAnswers[2])
	answerPost4Filter = outliersCriterion(answersPostMatrix[:, 4], trueAnswers[3])
	outliersIndexes = elementWiseAnd(answerPre1Filter, answerPre2Filter, answerPre3Filter, answerPre4Filter, answerPost1Filter, answerPost2Filter, answerPost3Filter, answerPost4Filter)

	count_outliers = 0
	for index, item in enumerate(outliersIndexes):
		if item == False:
			count_outliers += 1
			del answersPre[answersPreMatrix[index, 0]]
			del answersPost[answersPostMatrix[index, 0]]
	print 'Outliers rejected: ' + str(count_outliers) + ', remaining users: ' + str(len(answersPre))
"""

def elementWiseAnd(*args):
    return np.array([all(tuple) for tuple in zip(*args)])

def readAnswers(answerFilename):
	answers = {}
	answersFile = csv.reader(open(answerFilename, 'r'), delimiter=';')
	for row in answersFile:
		answer = []
		for element in row:
			answer.append(num(element))
		answers[answer[0]] = answer[1:]
	return answers

def normalizeAnswers():
	for nodeID in answersPre.iterkeys():
		answersPreNorm[nodeID] = [0, 0, 0, 0]
		answersPreNorm[nodeID][0] = division(answersPre[nodeID][0], trueAnswers[0])
		answersPreNorm[nodeID][1] = division(answersPre[nodeID][1], trueAnswers[1])
		answersPreNorm[nodeID][2] = division(answersPre[nodeID][2], trueAnswers[2])
		answersPreNorm[nodeID][3] = division(answersPre[nodeID][3], trueAnswers[3])
	for nodeID in answersPost.iterkeys():
		answersPostNorm[nodeID] = [0, 0, 0, 0]
		answersPostNorm[nodeID][0] = division(answersPost[nodeID][0], trueAnswers[0])
		answersPostNorm[nodeID][1] = division(answersPost[nodeID][1], trueAnswers[1])
		answersPostNorm[nodeID][2] = division(answersPost[nodeID][2], trueAnswers[2])
		answersPostNorm[nodeID][3] = division(answersPost[nodeID][3], trueAnswers[3])

def answersProcessing():
	global answersPre, answersPost, answersPreNorm, answersPostNorm, pearsonPreAvg
	answersPre = readAnswers(EXPERIMENT_PATH + ANSWERS_PRE_FILE)
	answersPost = readAnswers(EXPERIMENT_PATH + ANSWERS_POST_FILE)
	if len(answersPre) != len(answersPost):
		print "ERROR: Length of answersPre is not equal to length of answersPost... exit!"
		return
	if ENABLE_DEL_OUTLIERS:
		delOutliers()
	normalizeAnswers()
	

def addNodesToGraph(G, graph_file):
	f = open(graph_file, 'r')
	lines = f.readlines()
	f.close()
	
	for line in lines:
		if not(line.startswith('#')):
			(node1, node2, relationship) = line.strip().split('\t')
			node1 = num(node1)
			node2 = num(node2)
			if not G.has_node(node1) and node1 in answersPre and node1 in answersPost:
				G.add_node(node1)
			if not G.has_node(node2) and node2 in answersPre and node2 in answersPost:
				G.add_node(node2)
	

def buildGraph(G, graph_file):
	for edge in G.edges():
		G.remove_edge(*edge)
	
	f = open(graph_file, 'r')
	lines = f.readlines()
	f.close()
	
	for line in lines:
		if not(line.startswith('#')):
			(node1, node2, relationship) = line.strip().split('\t')
			node1 = num(node1)
			node2 = num(node2)
			if node1 != node2 and node1 in G.nodes() and node2 in G.nodes():
				G.add_edge(node1, node2)
	
def graphToMatrix(G):
	M = []
	for src in G.nodes():
		row = []
		for dst in G.nodes():
			if (src,dst) in G.edges():
				row.append(1)
			else:
				row.append(0)
		M.append(row)
	return M

def stochasticMatrix(M):
	W = []
	for i in range(len(M)):
		sum_i = sum(M[i])
		row_i = []
		if sum_i > 0:	
			for j in range(len(M[i])):
				row_i.append(float(M[i][j])/sum_i)
		else:
			for j in range(len(M[i])):
				row_i.append(1.0/len(M[i]))
		W.append(row_i)
	return np.array(W)
	
def degroot(data_files):
	savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'DeGroot_Evolution/'
	f = open(checkDir(savingPath) + 'degroot_evolution.csv', 'w')
	for i in range(4):			
		G = nx.Graph()
		for graph_file in data_files:
			addNodesToGraph(G, graph_file)
			
		p0 = []
		p1 = []
		for node in G.nodes():
			p0.append([answersPreNorm[node][i]]) # it will be a column-array
			p1.append([answersPostNorm[node][i]]) # it will be a column-array
		
		p0 = np.array(p0)
		p1 = np.array(p1)
		
		pt = p0.copy()
		
		for iteration, graph_file in enumerate(data_files):
			#print "\n\nITERATION: " + str(iteration) + ", processing " + graph_file
			buildGraph(G, graph_file)
			#print "Nodes:", len(G)
			#print "Edges:", len(G.edges())
			M = graphToMatrix(G)	
			W = stochasticMatrix(M)
			
			ppt = pt.copy()
			pt = np.dot(W, pt)
			
		#print 'ppt:', ppt
		#print 'pt:', pt
		
		if all(abs(pt[0] - x) < 0.0001 for x in pt): # check if all elements of pt are equal (i.e., consensus)
			mean = np.mean(pt)
			err_truth = np.mean([abs(x-1) for x in pt])
			err_p1 = np.mean(abs(pt-p1))
	
			print str(i+1) + ', Mean: ' + str(np.mean(pt)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1)
			f.write(str(i+1) + ', Mean: ' + str(np.mean(pt)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1) + '\n')
		else:
			print 'ERROR: Consensus not reached!!!'
	f.close()
		

def main():
	# Answers
	answersProcessing()
	
	data_files = sorted(glob.glob(GRAPHS_DIR + "*.txt"))
	if len(data_files) == 0:
		return
	
	degroot(data_files)


if __name__ == '__main__':
	main()
