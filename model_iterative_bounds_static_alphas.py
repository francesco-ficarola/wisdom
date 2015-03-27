import csv, exceptions, glob, os
import numpy as np
import networkx as nx
from scipy.special import erfc
from copy import deepcopy
from leastsq_bounds import leastsq_bounds

ENABLE_DEL_OUTLIERS = False
THRESHOLD_OUTLIERS = 100.0
ITERATIONS_MODEL = 100

#WSMD2013
trueAnswers = [951000, 52.59, 10.35, 3796000]
EXPERIMENT_PATH = 'wsdm/'

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
#trueAnswers = [36.5, 69, 100, 450]
#EXPERIMENT_PATH = 'diag2/'


answersPre = {}
answersPost = {}
answersPreNorm = {}
answersPostNorm = {}
ANSWERS_PRE_FILE = 'answers_raw_pre.csv'
ANSWERS_POST_FILE = 'answers_raw_post.csv'

GRAPHS_TYPE = 'graphs/*.csv'
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

"""
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

def readGraphFile(graphFileName):
	G = nx.Graph()
	nodesIds = []
	
	graphFile = csv.reader(open(graphFileName, 'r'), delimiter=';')
	firstRow = True
	for row in graphFile:
		if firstRow:
			nodesIds = map(int, row[1:])
			for node in nodesIds:
				if node in answersPreNorm and node in answersPostNorm:
					G.add_node(node)
			firstRow = False
		else:
			sourceNodeID = int(row[0])
			if sourceNodeID in G.nodes():
				weights = map(float, row[1:])
				for index, weight in enumerate(weights):
					targetNodeID = nodesIds[index]
					if weight > 0 and targetNodeID in G.nodes():
						edge = (sourceNodeID, targetNodeID)
						if edge not in G.edges():
							G.add_edge(*edge, weight=weight)
	return G

def residualsModel(alpha0, a_2, a_1, w_ij_pj_list, w_ij_list):
	alpha = alpha0
	a_2 = np.array(a_2)
	a_1 = np.array(a_1)
	w_ij_pj = np.array(w_ij_pj_list)
	w_ij = np.array(w_ij_list)
	
	est = (alpha * a_1 + w_ij_pj) / (alpha + w_ij)
	err = np.linalg.norm(est - a_2)
	return err
	
def fitModel(G, pt_dict):
	answers_params = {}
	for node in G.nodes():
		answers_pre_node = []
		answers_post_node = []
		w_ij_pj_list = []
		w_ij_list = []
		for i in range(4):
			w_ij_pj = 0.0
			w_ij = 0.0
			for neighbor in G.neighbors(node):
				w_ij += G[node][neighbor]['weight']
				w_ij_pj += (G[node][neighbor]['weight'] * pt_dict[neighbor][i])
			answers_pre_node.append(answersPreNorm[node][i])
			answers_post_node.append(answersPostNorm[node][i])
			w_ij_pj_list.append(w_ij_pj)
			w_ij_list.append(w_ij)
						
		alpha0 = 1.0
		least_square = leastsq_bounds(residualsModel, alpha0, [[0., float('Inf')]], boundsweight=10000, args=(answers_post_node, answers_pre_node, w_ij_pj_list, w_ij_list))
		alpha = least_square[0][0]
		
		pt_dict[node][i] = (alpha * answersPreNorm[node][i] + w_ij_pj) / (alpha + w_ij)
		
		answers_params[node] = alpha
	return (pt_dict, answers_params)
	
def model(G, pt_dict, answers_params):
	for i in range(4):
		for node in G.nodes():
			alpha = answers_params[node]
			w_ij_pj = 0.0
			w_ij = 0.0
			for neighbor in G.neighbors(node):
				w_ij += G[node][neighbor]['weight']
				w_ij_pj += (G[node][neighbor]['weight'] * pt_dict[neighbor][i])

			est = (alpha * answersPreNorm[node][i] + w_ij_pj) / (alpha + w_ij)
			pt_dict[node][i] = est
	return pt_dict
	
def exportResults(G, pt_dict, ppt_dict, fileName):
	savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Model/'
	f = open(checkDir(savingPath) + 'stats_' + fileName + '.csv', 'w')	
	for i in range(4):
		p1 = []
		pt = []
		ppt = []
		for node in G.nodes():
			p1.append(answersPostNorm[node][i])
			pt.append(pt_dict[node][i])
			ppt.append(ppt_dict[node][i])
		
		p1 = np.array(p1)
		pt = np.array(pt)
		ppt = np.array(ppt)
		
		if np.mean(abs(pt - ppt)) < 0.01:
			mean = np.mean(pt)
			err_truth = np.mean([abs(x-1) for x in pt])
			err_p1 = np.mean(abs(pt-p1))
	
			print str(i+1) + ', Mean: ' + str(np.mean(pt)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1)
			f.write(str(i+1) + ', Mean: ' + str(np.mean(pt)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1) + '\n')
		else:
			print 'ERROR: Convergence is not reached!!!'
	f.close()

def main():
	answersProcessing()
	
	csvFiles = glob.glob(EXPERIMENT_PATH + GRAPHS_TYPE)
	for graphFileName in csvFiles:
		head, tail = os.path.split(graphFileName)
		fileName, fileExtension = os.path.splitext(tail)
		
		print '\n\n' + graphFileName
		G = readGraphFile(graphFileName)
		
		pt_dict = deepcopy(answersPreNorm)
		counter = 0
		(pt_dict, answers_params) = fitModel(G, pt_dict)
		alphas = np.array(answers_params.values())
		alphas_prec = np.array([-1] * len(alphas))
		while np.mean(abs(alphas_prec - alphas)) > 0.0001:
			alphas_prec = list(alphas)
			(pt_dict, answers_params) = fitModel(G, pt_dict)
			alphas = np.array(answers_params.values())
			#print "t-1",alphas_prec
			#print "t",alphas
			#print np.mean(abs(alphas_prec - alphas))
			counter += 1
			if counter > ITERATIONS_MODEL:
				break

		pt_dict = deepcopy(answersPreNorm)
		for iteration in range(ITERATIONS_MODEL):
			ppt_dict = deepcopy(pt_dict)
			pt_dict = model(G, pt_dict, answers_params)
		#print 'ppt:', ppt_dict
		#print 'pt:', pt_dict
		exportResults(G, pt_dict, ppt_dict, fileName)

if __name__ == '__main__':
	main()
