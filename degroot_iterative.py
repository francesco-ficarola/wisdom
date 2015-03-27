import csv, exceptions, glob, os
import numpy as np
from scipy.special import erfc

ENABLE_DEL_OUTLIERS = False
THRESHOLD_OUTLIERS = 100.0

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
	with open(graphFileName, 'rb') as f:
		ncols = len(next(f).split(';'))
	M = np.genfromtxt(graphFileName, delimiter=';', dtype=np.float64, names=True, usecols=range(1,ncols))
	labels = M.dtype.names
	
	nodes = []
	X = []
	for i in range(len(M)):
		if int(labels[i]) in answersPreNorm and int(labels[i]) in answersPostNorm:
			row_i = []
			nodes.append(int(labels[i]))
			for j in range(len(M[i])):
				if int(labels[j]) in answersPreNorm and int(labels[j]) in answersPostNorm:
					row_i.append(M[i][j])
			X.append(row_i)
	return (np.array(X), np.array(nodes))
	
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
			
	
def deGroot(W, nodes, fileName):
	savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'DeGroot_Iterative/'
	f = open(checkDir(savingPath) + 'stats_' + fileName + '.csv', 'w')
	for i in range(4):
		p0 = []
		p1 = []
		for node in nodes:
			p0.append([answersPreNorm[node][i]]) # it will be a column-array
			p1.append([answersPostNorm[node][i]]) # it will be a column-array
		
		p0 = np.array(p0)
		p1 = np.array(p1)
		
		#pt = p0.copy()
		#for j in range(1000):
		#	pt = np.dot(W, pt)	
		#print 'pt:', pt
		
		W_inf = np.linalg.matrix_power(W, 10000)
		pt = np.dot(W_inf, p0)
	
		#print W_inf
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
	answersProcessing()
	
	csvFiles = glob.glob(EXPERIMENT_PATH + GRAPHS_TYPE)
	for graphFileName in csvFiles:
		head, tail = os.path.split(graphFileName)
		fileName, fileExtension = os.path.splitext(tail)
		
		print '\n\n' + graphFileName
		(M, nodes) = readGraphFile(graphFileName)
		
		W = stochasticMatrix(M)
		deGroot(W, nodes, fileName)
		

if __name__ == '__main__':
	main()
