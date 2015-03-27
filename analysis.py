import csv, sys, exceptions, itertools, collections, os, glob, re
from scipy import stats
from collections import defaultdict
from scipy.special import erfc
from scipy.optimize import leastsq
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf
import networkx as nx
import random as rnd
import numpy as np
import math

ENABLE_DEL_OUTLIERS = False
THRESHOLD_OUTLIERS = 10.0

#WSMD2013
#trueAnswers = [951000, 52.59, 10.35, 3796000]
#EXPERIMENT_PATH = 'wsdm/'

#SocialDIAG - Experiment 1
#trueAnswers = [58734670, 580, 69, 42.86]
#EXPERIMENT_PATH = 'diag/experiment_1/'

#SocialDIAG - Experiment 2
trueAnswers = [951000, 792, 80, 21.01]
EXPERIMENT_PATH = 'diag/experiment_2/'

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
pearsonPreAvg = 0
ANSWERS_PRE_FILE = 'answers_raw_pre.csv'
ANSWERS_POST_FILE = 'answers_raw_post.csv'

GRAPHS_TYPE = 'graphs/*.csv'
#GRAPHS_TYPE = 'graphs_interv/*.csv'
#GRAPHS_TYPE = 'graphs_interv_sub/*.csv'
#GRAPHS_TYPE = 'graphs_cont/*.csv'
#GRAPHS_TYPE = 'graphs_cont_sub/*.csv'

OUTCOME_PATH = 'outcomes/'
OUTPUT_HEADER = 'filename;pearson_pre;pearson_post;num_correlations\n'
OUTPUT_FILENAME = 'pearson_correlations_graphs.csv'

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
#	print answersPreMatrix
#	print answersPostMatrix
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
#			print answersPreMatrix[index, 0], item
#			print answersPre[answersPreMatrix[index, 0]]
#			print answersPost[answersPostMatrix[index, 0]]
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
#			print answersPreMatrix[index, 0], item
#			print answersPre[answersPreMatrix[index, 0]]
#			print answersPost[answersPostMatrix[index, 0]]
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

def plotAnswers():
	savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'AnswersDistribution/'
	answersPreMatrix = np.array(answersPreNorm.values())
	answersPostMatrix = np.array(answersPostNorm.values())
	
	filename = "answers_pre.png"
	plt.figure(figsize=(18, 3))
	for i in range(4):
		plt.subplot(141+i)
		max_value = max(answersPreMatrix[:, i])
		x = []
		y = []
		scale = 0.0
		offset = 0.1
		while scale < max_value:
			x.append(scale)
			n = sum(v > scale and v <= scale+offset for v in answersPreMatrix[:, i])
			y.append(n)
			scale += offset
		
		plt.title('Q' + str(i+1))
		plt.xlabel('Answers')
		plt.ylabel('Frequency')
		plt.xlim(xmin=0,xmax=(scale-offset))
		plt.ylim(ymin=0,ymax=max(y))
		plt.plot(x, y, 'r-', linewidth=1)
	plt.subplots_adjust(left=0.04, bottom=0.18, right=0.98, top=0.9, wspace=0.28)
	plt.savefig(checkDir(savingPath) + filename, format="png")
	plt.close()
	
	filename = "answers_post.png"
	plt.figure(figsize=(18, 3))
	for i in range(4):
		plt.subplot(141+i)
		max_value = max(answersPostMatrix[:, i])
		x = []
		y = []
		scale = 0.0
		offset = 0.1
		while scale < max_value:
			x.append(scale)
			n = sum(v > scale and v <= scale+offset for v in answersPostMatrix[:, i])
			y.append(n)
			scale += offset
		
		plt.title('Q' + str(i+1))
		plt.xlabel('Answers')
		plt.ylabel('Frequency')
		plt.xlim(xmin=0,xmax=(scale-offset))
		plt.ylim(ymin=0,ymax=max(y))
		plt.plot(x, y, 'r-', linewidth=1)
	plt.subplots_adjust(left=0.04, bottom=0.18, right=0.98, top=0.9, wspace=0.28)
	plt.savefig(checkDir(savingPath) + filename, format="png")
	plt.close()

def findAnswersMax(answers):
	answersMatrix = np.array(answers.values())
	return [num(max(answersMatrix[:, 0])), num(max(answersMatrix[:, 1])), num(max(answersMatrix[:, 2])), num(max(answersMatrix[:, 3]))]

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

def pearsonAnswersPreNorm():
	# Avg Pearson correlation of answers for all combinations of nodes (pre-experiment)
	pearsonCorrelationsPre = []
	alreadyProcessedID = []
	for i in answersPreNorm.keys():
		for j in [k for k in answersPreNorm.keys() if k not in alreadyProcessedID]:
			if i != j:
#					print answersPreNorm[i], answersPreNorm[j]
				pearsonPre = stats.mstats.pearsonr(answersPreNorm[i], answersPreNorm[j])
				pearsonCorrelationsPre.append(pearsonPre)
		alreadyProcessedID.append(i)
	return np.mean(pearsonCorrelationsPre)

def plotErrorDistance():
	savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Answers_ErrorPre_DistancePrePost/'
	plt.figure(figsize=(20, 4.5))
	marker = 'ro' if ENABLE_DEL_OUTLIERS else 'bo'
	for i in range(4):
		answersPreArray = np.array(answersPreNorm.values())[:, i]
		answersPostArray = np.array(answersPostNorm.values())[:, i]
		errorPre = answersPreArray - 1 # 1 is the ground truth normalized
		distancePrePost = (answersPostArray - answersPreArray)
		pearsonErrorDistancePrePost = stats.mstats.pearsonr(errorPre, distancePrePost)[0]
		m, b = np.polyfit(errorPre, distancePrePost, 1)
		title = 'Answer ' + str(i+1)  + ', ' + r'$\rho=' + str(round(pearsonErrorDistancePrePost, 3)) + '$'
		plt.subplot(141+i)
		plt.title(title, fontsize=20)
		plt.xlabel(r'$\epsilon_{R_1}$', fontsize=24)
		plt.ylabel(r'$\Delta_{R_2, R_1}$', fontsize=24)
		plt.xlim(min(min(errorPre), min(distancePrePost)) * 1.1, max(max(errorPre), max(distancePrePost)) * 1.1)
		plt.ylim(min(min(errorPre), min(distancePrePost)) * 1.1, max(max(errorPre), max(distancePrePost)) * 1.1)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.plot(errorPre, distancePrePost, marker, errorPre, m*errorPre+b, '--k')
	plt.subplots_adjust(left=0.07, bottom=0.18, right=0.98, top=0.88, wspace=0.35, hspace=0)
	plt.savefig(checkDir(savingPath) + 'answers_error_distance' + '.png', format='png')
	plt.close()

def plotErrorRounds():
	savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Answers_ErrorPre_ErrorPost/'
	plt.figure(figsize=(20,18))
	for i in range(4):
		correctAnswer = [trueAnswers[i]] * len(answersPre)
		answersPreArray = np.array(answersPre.values())[:, i]
		answersPostArray = np.array(answersPost.values())[:, i]
#		print answersPreArray, correctAnswer
		errorPre = abs(answersPreArray - correctAnswer)
		errorPost = abs(answersPostArray - correctAnswer)
		maxError = max(max(errorPre), max(errorPost))
		title = 'Answer ' + str(i+1)
		plt.subplot(221+i)
		plt.title(title)
		plt.xlabel('Error in the pre-experiment with respect to the correct answer')
		plt.ylabel('Error in the post-experiment with respect to the correct answer')
		plt.plot(errorPre, errorPost, 'ro')
		plt.xlim(xmax=maxError)
		plt.ylim(ymax=maxError)
	plt.savefig(checkDir(savingPath) + 'answers_error_rounds' + '.png', format='png')
	plt.close()
	
# Correlations between nodes' answers and mean
def plotDistanceAvgAnswers():
	answersPreMatrix = np.concatenate((np.array([answersPre.keys()]).T, np.array(answersPre.values())), axis=1)
	answersPostMatrix = np.concatenate((np.array([answersPost.keys()]).T, np.array(answersPost.values())), axis=1)
	savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_DistanceAvgAnswers/'
	plt.figure(figsize=(20,18))
	for i in range(4):
		meanRound1 = np.mean(answersPreMatrix[:, i+1])
		meanRound2 = np.mean(answersPostMatrix[:, i+1])
		distanceAvgAnswersRound1 = {}
		distanceAvgAnswersRound2 = {}
		for nodeID in answersPre and answersPost:
			distanceAvgAnswersRound1[nodeID] = answersPre[nodeID][i] - meanRound1
			distanceAvgAnswersRound2[nodeID] = answersPost[nodeID][i] - meanRound2
		distanceAvgAnswersRound2 = collections.OrderedDict(sorted(distanceAvgAnswersRound2.items(), key=lambda t: t[1], reverse=False))
		distanceAvgAnswersRound1List = []
		distanceAvgAnswersRound2List = []
		for nodeID in distanceAvgAnswersRound2:
			distanceAvgAnswersRound1List.append(distanceAvgAnswersRound1[nodeID])
			distanceAvgAnswersRound2List.append(distanceAvgAnswersRound2[nodeID])
		x = np.array(distanceAvgAnswersRound1List)
		y = np.array(distanceAvgAnswersRound2List)
		title = 'Answer ' + str(i+1)
		plt.subplot(221+i)
		plt.title(title)
		plt.xlabel('answers_round1[i] - avg(answers_round1)')
		plt.ylabel('answers_round2[i] - avg(answers_round2)')
		plt.plot(x, y, 'ro')
	plt.savefig(checkDir(savingPath) + 'diff_singleAnswer_average' + '.png', format='png')
	plt.close()

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
	plotAnswers()
	plotErrorDistance()
	plotErrorRounds()
	plotDistanceAvgAnswers()
	pearsonPreAvg = pearsonAnswersPreNorm()


class GraphAnalyzer():
	def __init__(self):
		self.nodesIds = []
		self.edgesDict = {} # key: [edge], value: [weight, pearson]

	def readAggregateGraph(self, graphFileName):
		graphFile = csv.reader(open(graphFileName, 'r'), delimiter=';')
		firstRow = True
		for row in graphFile:
			if firstRow:
				self.nodesIds = map(int, row[1:])
				firstRow = False
			else:
				sourceNodeID = int(row[0])
				weights = map(float, row[1:])
				for index, weight in enumerate(weights):
					if weight > 0:
						targetNodeID = self.nodesIds[index]
						edge = [min(sourceNodeID, targetNodeID), max(sourceNodeID, targetNodeID)]
						edgeTuple = tuple(edge)
						if edgeTuple not in self.edgesDict:
							self.edgesDict[edgeTuple] = [weight]
		return len(self.edgesDict)

	def buildNXGraph(self):
		nxGraph = nx.Graph()
		for node in self.nodesIds:
			nxGraph.add_node(node)
		for edge in self.edgesDict.iterkeys():
			nxGraph.add_edge(edge[0], edge[1], weight=self.edgesDict[edge][0])
		#print self.edgesDict.keys()
		#print nxGraph.edges()
		#print str(nxGraph.nodes())
		#print str(nxGraph.edges(data=True))
		return nxGraph

	# Avg Pearson correlation of answers for all nodes in graphs (post-experiment)
	def pearsonAnswersPostNorm(self):
		countCorrelations = 0
		pearsonCorrelationsPost = []
		for edge in self.edgesDict:
			if(answersPostNorm.has_key(edge[0]) and answersPostNorm.has_key(edge[1])):
				countCorrelations += 1
				pearsonPost = stats.mstats.pearsonr(answersPostNorm[edge[0]], answersPostNorm[edge[1]])[0]
#				print edge, pearsonPost
				pearsonCorrelationsPost.append(pearsonPost)
				self.edgesDict[edge].append(pearsonPost)
		return [np.mean(pearsonCorrelationsPost), countCorrelations] if countCorrelations > 0 else [0, 0]

	# Correlation between edge pearson coefficients and edge weights
	def plotAllAnswersPearsonWeight(self, fileName):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_AllAnswersPearson_Weight/'
		weightsList = []
		correlationsList = []
		for values in self.edgesDict.values():
			if len(values) == 2: # check if edgesDict[node] has the correlation field (i.e. a correspondence of node in answers)
				weightsList.append(values[0])
				correlationsList.append(values[1])
		weights = np.array(weightsList)
		correlations = np.array(correlationsList)
		pearsonWeightCor = stats.mstats.pearsonr(weights, correlations)[0]
		m, b = np.polyfit(weights, correlations, 1)
		title = 'Graph ' + fileName + '\nPearson coefficient ' + str(pearsonWeightCor) + '\nm=' + str(m) + ' b=' + str(b)
		plt.figure(figsize=(10,8))
		plt.title(title)
		plt.xlabel('Weight of the edge')
		plt.ylabel('Pearson correlation for each edge')
		plt.plot(weights, correlations, 'ro', weights, m*weights+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
	
	################################################
	################## STATISTICS ##################
	################################################
	
	def computeStats(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Stats/'
		checkDir(savingPath)
		f = open(savingPath + fileName + '.txt', 'w')
		for i in range(4):
			allValidAnswersPre = []
			allValidAnswersPost = []
			allValidAnswersPreNorm = []
			allValidAnswersPostNorm = []
			degMeanPreArray = []
			degMeanPostArray = []
			wDegMeanPreArray = []
			wDegMeanPostArray = []
			totalDegree = 0
			totalWeightDegree = 0
			for nodeID in nxGraph:
				degreeNodeID = nxGraph.degree(nodeID)
				wDegreeNodeID = nxGraph.degree(nodeID, weight='weight')
				totalDegree += degreeNodeID
				totalWeightDegree += wDegreeNodeID
				if answersPre.has_key(nodeID):
					allValidAnswersPre.append(answersPre[nodeID][i])
					allValidAnswersPreNorm.append(answersPreNorm[nodeID][i])
					degMeanPreArray.append(answersPre[nodeID][i] * degreeNodeID)
					wDegMeanPreArray.append(answersPre[nodeID][i] * wDegreeNodeID)
				if answersPost.has_key(nodeID):
					allValidAnswersPost.append(answersPost[nodeID][i])
					allValidAnswersPostNorm.append(answersPostNorm[nodeID][i])
					degMeanPostArray.append(answersPost[nodeID][i] * degreeNodeID)
					wDegMeanPostArray.append(answersPost[nodeID][i] * wDegreeNodeID)
			meanPreNorm = np.mean(allValidAnswersPreNorm)
			meanPostNorm = np.mean(allValidAnswersPostNorm)
			gMeanPre = stats.mstats.gmean(allValidAnswersPre)
			gMeanPost = stats.mstats.gmean(allValidAnswersPost)
			degMeanPre = sum(degMeanPreArray)/float(totalDegree)
			degMeanPost = sum(degMeanPostArray)/float(totalDegree)
			wDegMeanPre = sum(wDegMeanPreArray)/float(totalWeightDegree)
			wDegMeanPost = sum(wDegMeanPostArray)/float(totalWeightDegree)
			stDevPreNorm = np.std(allValidAnswersPreNorm)
			stDevPostNorm = np.std(allValidAnswersPostNorm)
			
			# Wisdom indicator
			allValidAnswersPre.sort()
			centralPre = len(allValidAnswersPre) / 2
			indicatorPre = 0
			foundPre = False
			if len(allValidAnswersPre) % 2 == 0:
				while foundPre == False:
					if indicatorPre + 1 > centralPre:
						indicatorPre = -1
						break
					if trueAnswers[i] >= allValidAnswersPre[centralPre - indicatorPre - 1] and trueAnswers[i] <= allValidAnswersPre[centralPre + indicatorPre]:
						foundPre = True
					else:
						indicatorPre += 1
			else:
				while foundPre == False:
					if indicatorPre + 1 > centralPre:
						indicatorPre = -1
						break
					if trueAnswers[i] >= allValidAnswersPre[centralPre - indicatorPre - 1] and trueAnswers[i] <= allValidAnswersPre[centralPre + indicatorPre + 1]:
						foundPre = True
					else:
						indicatorPre += 1
			
			allValidAnswersPost.sort()
			centralPost = len(allValidAnswersPost) / 2
			indicatorPost = 0
			foundPost = False
			if len(allValidAnswersPost) % 2 == 0:
				while foundPost == False:
					if indicatorPost + 1 > centralPost:
						indicatorPost = -1
						break
					if trueAnswers[i] >= allValidAnswersPost[centralPost - indicatorPost - 1] and trueAnswers[i] <= allValidAnswersPost[centralPost + indicatorPost]:
						foundPost = True
					else:
						indicatorPost += 1
			else:
				while foundPost == False:
					if indicatorPost + 1 > centralPost:
						indicatorPost = -1
						break
					if trueAnswers[i] >= allValidAnswersPost[centralPost - indicatorPost - 1] and trueAnswers[i] <= allValidAnswersPost[centralPost + indicatorPost + 1]:
						foundPost = True
					else:
						indicatorPost += 1
			
			f.write('\n*** Graph ' + fileName + ' - Answer '  + str(i+1) + ' ***\n')
			f.write('Truth: ' + str(trueAnswers[i])+ '\n')
			f.write('Norm Mean [PRE]: ' + str(meanPreNorm)+ '\n')
			f.write('Norm Mean [POST]: ' + str(meanPostNorm)+ '\n')
			f.write('Wisdom Indicator [PRE]: ' + str(indicatorPre) + '\n')
			f.write('Wisdom Indicator [POST]: ' + str(indicatorPost) + '\n')
			f.write('Geometric Mean [PRE]: ' + str(gMeanPre) + '\n')
			f.write('Geometric Mean [POST]: ' + str(gMeanPost) + '\n')
			f.write('Deg-Mean [PRE]: ' + str(degMeanPre)+ '\n')
			f.write('Deg-Mean [POST]: ' + str(degMeanPost)+ '\n')
			f.write('wDeg-Mean [PRE]: ' + str(wDegMeanPre)+ '\n')
			f.write('wDeg-Mean [POST]: ' + str(wDegMeanPost)+ '\n')
			f.write('Norm StDev [PRE]: ' + str(stDevPreNorm)+ '\n')
			f.write('Norm StDev [POST]: ' + str(stDevPostNorm)+ '\n')
			f.flush()
		f.close()
			
	
	###############################################################
	################## CORRELATION WITH DISTANCE ##################
	###############################################################
	
	# Correlations between answers distance (difference) and weights
	def plotDistanceNodes(self, fileName):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_DistanceR2R1AnswersNodes_Weight/'
		weightsList = []
		for values in self.edgesDict.values():
			if len(values) == 2: # check if edgesDict[node] has the correlation field (i.e. a correspondence of node in answers)
				weightsList.append(values[0])
		weights = np.array(weightsList)
		plt.figure(figsize=(20,18))
		for i in range(4):
			distanceNodesRound1 = []
			distanceNodesRound2 = []
			for edgeTuple in self.edgesDict.iterkeys():
				sourceNode = edgeTuple[0]
				targetNode = edgeTuple[1]
				if sourceNode in answersPre and sourceNode in answersPost and targetNode in answersPre and targetNode in answersPost:
					distanceNodesRound1.append(answersPre[sourceNode][i] - answersPre[targetNode][i])
					distanceNodesRound2.append(answersPost[sourceNode][i] - answersPost[targetNode][i])
			differenceRound2Round1 = [a - b for a, b in zip(distanceNodesRound1, distanceNodesRound2)]
			arrayRound2Round1 = np.array(differenceRound2Round1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Distance round 2 - Distance round 1')
			plt.ylabel('Weight of the edge')
			plt.plot(arrayRound2Round1, weights, 'ro')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
	
	# Correlations between answers distance (ratio) and weights
	def plotDistanceRatioNodes(self, fileName):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_DistanceRationR2R1AnswersNodes_Weight/'
		weightsList = []
		for values in self.edgesDict.values():
			if len(values) == 2: # check if edgesDict[node] has the correlation field (i.e. a correspondence of node in answers)
				weightsList.append(values[0])
		weights = np.array(weightsList)
		plt.figure(figsize=(20,18))
		for i in range(4):
			distanceNodesRound1 = []
			distanceNodesRound2 = []
			for edgeTuple in self.edgesDict.iterkeys():
				sourceNode = edgeTuple[0]
				targetNode = edgeTuple[1]
				if sourceNode in answersPre and sourceNode in answersPost and targetNode in answersPre and targetNode in answersPost:
					distanceNodesRound1.append(answersPre[sourceNode][i] - answersPre[targetNode][i])
					distanceNodesRound2.append(answersPost[sourceNode][i] - answersPost[targetNode][i])
			ratioRound2Round1 = [float(b)/float(a) if a != 0 else 0 for a, b in zip(distanceNodesRound1, distanceNodesRound2)]
			arrayRound2Round1 = np.array(ratioRound2Round1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Ratio round 2 / round 1')
			plt.ylabel('Weight of the edge')
			plt.plot(arrayRound2Round1, weights, 'ro')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
			
	
	# Correlations between node answers distance and weights
	def plotDistanceNodesPreWeight(self, fileName):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_DistanceAnswersNodesPre_Weight/'
		weightsList = []
		for values in self.edgesDict.values():
			if len(values) == 2: # check if edgesDict[node] has the correlation field (i.e. a correspondence of node in answers)
				weightsList.append(values[0])
		weights = np.array(weightsList)
		plt.figure(figsize=(20,18))
		for i in range(4):
			distanceNodeAnswersList = []
			for edgeTuple in self.edgesDict.iterkeys():
				sourceNode = edgeTuple[0]
				targetNode = edgeTuple[1]
				if sourceNode in answersPre and targetNode in answersPre:
					distanceNodeAnswersList.append(answersPre[sourceNode][i] - answersPre[targetNode][i])
			distanceNodeAnswers = np.array(distanceNodeAnswersList)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Answer distance between nodes of each edge')
			plt.ylabel('Weight of the edge')
			plt.plot(distanceNodeAnswers, weights, 'ro')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	# Correlations between node answers distance and weights
	def plotDistanceNodesPostWeight(self, fileName):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_DistanceAnswersNodesPost_Weight/'
		weightsList = []
		for values in self.edgesDict.values():
			if len(values) == 2: # check if edgesDict[node] has the correlation field (i.e. a correspondence of node in answers)
				weightsList.append(values[0])
		weights = np.array(weightsList)
		plt.figure(figsize=(20,18))
		for i in range(4):
			distanceNodeAnswersList = []
			for edgeTuple in self.edgesDict.iterkeys():
				sourceNode = edgeTuple[0]
				targetNode = edgeTuple[1]
				if sourceNode in answersPost and targetNode in answersPost:
					distanceNodeAnswersList.append(answersPost[sourceNode][i] - answersPost[targetNode][i])
			distanceNodeAnswers = np.array(distanceNodeAnswersList)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Answer distance between nodes of each edge')
			plt.ylabel('Weight of the edge')
			plt.plot(distanceNodeAnswers, weights, 'ro')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between answer distances and edge weights
	def plotDistancePrePostWeight(self, fileName):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_AvgDistancePrePost_Weight/'
		weightsList = []
		for values in self.edgesDict.values():
			if len(values) == 2: # check if edgesDict[node] has the correlation field (i.e. a correspondence of node in answers)
				weightsList.append(values[0])
		weights = np.array(weightsList)
		plt.figure(figsize=(20,18))
		for i in range(4):
			distancePrePostEdgesList = []
			for edgeTuple in self.edgesDict.iterkeys():
				sourceNode = edgeTuple[0]
				targetNode = edgeTuple[1]
				if answersPre.has_key(sourceNode) and answersPre.has_key(targetNode) and answersPost.has_key(sourceNode) and answersPost.has_key(targetNode):
					distancePrePostSourceNode = answersPost[sourceNode][i] - answersPre[sourceNode][i]
					distancePrePostTargetNode = answersPost[targetNode][i] - answersPre[targetNode][i]
					avgDistancePrePostEdge = np.mean([distancePrePostSourceNode, distancePrePostTargetNode])
					distancePrePostEdgesList.append(avgDistancePrePostEdge)
			distancePrePostEdges = np.array(distancePrePostEdgesList)
			pearsonWeightsDistancePrePost = stats.mstats.pearsonr(weights, distancePrePostEdges)[0]
			m, b = np.polyfit(weights, distancePrePostEdges, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)  + '\nPearson coefficient ' + str(pearsonWeightsDistancePrePost) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Weight of the edge')
			plt.ylabel('Avg answer distance pre- and post-interaction for each edge')
			plt.plot(weights, distancePrePostEdges, 'ro', weights, m*weights+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between answer distances and node degrees
	def plotDistancePrePostDegree(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_DistancePrePost_Degree/'
		plt.figure(figsize=(20,18))
		for i in range(4):
			distancePrePostNodesList = []
			degreesList = []
			for nodeID in nxGraph:
				if answersPre.has_key(nodeID) and answersPost.has_key(nodeID):
					distancePrePostNode = answersPost[nodeID][i] - answersPre[nodeID][i]
					degreesList.append(nxGraph.degree(nodeID))
					distancePrePostNodesList.append(distancePrePostNode)
			degrees = np.array(degreesList)
			distancePrePostNodes = np.array(distancePrePostNodesList)
			pearsonDistanceDegree = stats.mstats.pearsonr(degrees, distancePrePostNodes)[0]
			m, b = np.polyfit(degrees, distancePrePostNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)  + '\nPearson coefficient ' + str(pearsonDistanceDegree) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Degree')
			plt.ylabel('Answer distance between pre- and post-interaction')
			plt.plot(degrees, distancePrePostNodes, 'ro', degrees, m*degrees+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between answer distances and node weighted degrees
	def plotDistancePrePostWeightedDegree(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_DistancePrePost_WeightedDegree/'
		plt.figure(figsize=(20,18))
		for i in range(4):
			distancePrePostNodesList = []
			weightedDegreesList = []
			for nodeID in nxGraph:
				if answersPre.has_key(nodeID) and answersPost.has_key(nodeID):
					distancePrePostNode = answersPost[nodeID][i] - answersPre[nodeID][i]
					weightedDegreesList.append(nxGraph.degree(nodeID, weight='weight'))
					distancePrePostNodesList.append(distancePrePostNode)
			weightedDegrees = np.array(weightedDegreesList)
			distancePrePostNodes = np.array(distancePrePostNodesList)
			pearsonDistanceDegree = stats.mstats.pearsonr(weightedDegrees, distancePrePostNodes)[0]
			m, b = np.polyfit(weightedDegrees, distancePrePostNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)  + '\nPearson coefficient ' + str(pearsonDistanceDegree) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Weighted Degree')
			plt.ylabel('Answer distance between pre- and post-interaction')
			plt.plot(weightedDegrees, distancePrePostNodes, 'ro', weightedDegrees, m*weightedDegrees+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
	
	# Correlation between answer distances and node pageranks
	def plotDistancePrePostPagerank(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_DistancePrePost_Pagerank/'
		pageranksDict = nx.pagerank(nxGraph)
		diff = set(pageranksDict) - set(answersPre)
		for element in diff:
			del(pageranksDict[element])
		pageranksList = pageranksDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			distancePrePostNodesList = []
			for nodeID in pageranksDict.iterkeys():
				if answersPre.has_key(nodeID) and answersPost.has_key(nodeID):
					distancePrePostNode = answersPost[nodeID][i] - answersPre[nodeID][i]
					distancePrePostNodesList.append(distancePrePostNode)
			pageranks = np.array(pageranksList)
			distancePrePostNodes = np.array(distancePrePostNodesList)
			pearsonDistancePagerank = stats.mstats.pearsonr(pageranks, distancePrePostNodes)[0]
			m, b = np.polyfit(pageranks, distancePrePostNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)  + '\nPearson coefficient ' + str(pearsonDistancePagerank) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Pagerank')
			plt.ylabel('Answer distance between pre- and post-interaction')
			plt.plot(pageranks, distancePrePostNodes, 'ro', pageranks, m*pageranks+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	# Correlation between answer distances and node betweenness
	def plotDistancePrePostBetweenness(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_DistancePrePost_Betweenness/'
		betweennessDict = nx.betweenness_centrality(nxGraph)
		diff = set(betweennessDict) - set(answersPre)
		for element in diff:
			del(betweennessDict[element])
		betweennessList = betweennessDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			distancePrePostNodesList = []
			for nodeID in betweennessDict.iterkeys():
				if answersPre.has_key(nodeID) and answersPost.has_key(nodeID):
					distancePrePostNode = answersPost[nodeID][i] - answersPre[nodeID][i]
					distancePrePostNodesList.append(distancePrePostNode)
			betweenness = np.array(betweennessList)
			distancePrePostNodes = np.array(distancePrePostNodesList)
			pearsonDistanceBetweenness = stats.mstats.pearsonr(betweenness, distancePrePostNodes)[0]
			m, b = np.polyfit(betweenness, distancePrePostNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)  + '\nPearson coefficient ' + str(pearsonDistanceBetweenness) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Betweenness')
			plt.ylabel('Answer distance between pre- and post-interaction')
			plt.plot(betweenness, distancePrePostNodes, 'ro', betweenness, m*betweenness+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	# Correlation between answer distances and node closeness
	def plotDistancePrePostCloseness(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_DistancePrePost_Closeness/'
		closenessDict = nx.closeness_centrality(nxGraph)
		diff = set(closenessDict) - set(answersPre)
		for element in diff:
			del(closenessDict[element])
		closenessList = closenessDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			distancePrePostNodesList = []
			for nodeID in closenessDict.iterkeys():
				if answersPre.has_key(nodeID) and answersPost.has_key(nodeID):
					distancePrePostNode = answersPost[nodeID][i] - answersPre[nodeID][i]
					distancePrePostNodesList.append(distancePrePostNode)
			closeness = np.array(closenessList)
			distancePrePostNodes = np.array(distancePrePostNodesList)
			pearsonDistanceCloseness = stats.mstats.pearsonr(closeness, distancePrePostNodes)[0]
			m, b = np.polyfit(closeness, distancePrePostNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)  + '\nPearson coefficient ' + str(pearsonDistanceCloseness) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Closeness')
			plt.ylabel('Answer distance between pre- and post-interaction')
			plt.plot(closeness, distancePrePostNodes, 'ro', closeness, m*closeness+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	# Correlation between answer distances and node communicability
	def plotDistancePrePostCommunicability(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_DistancePrePost_Communicability/'
		communicabilityDict = nx.communicability_centrality(nxGraph)
		diff = set(communicabilityDict) - set(answersPre)
		for element in diff:
			del(communicabilityDict[element])
		communicabilityList = communicabilityDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			distancePrePostNodesList = []
			for nodeID in communicabilityDict.iterkeys():
				if answersPre.has_key(nodeID) and answersPost.has_key(nodeID):
					distancePrePostNode = answersPost[nodeID][i] - answersPre[nodeID][i]
					distancePrePostNodesList.append(distancePrePostNode)
			communicability = np.array(communicabilityList)
			distancePrePostNodes = np.array(distancePrePostNodesList)
			pearsonDistanceCommunicability = stats.mstats.pearsonr(communicability, distancePrePostNodes)[0]
			m, b = np.polyfit(communicability, distancePrePostNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)  + '\nPearson coefficient ' + str(pearsonDistanceCommunicability) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Communicability')
			plt.ylabel('Answer distance between pre- and post-interaction')
			plt.plot(communicability, distancePrePostNodes, 'ro', communicability, m*communicability+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	# Correlation between answer distances and node eigenvector
	def plotDistancePrePostEigenvector(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_DistancePrePost_Eigenvector/'
		eigenvectorDict = {}
		try:
			eigenvectorDict = nx.eigenvector_centrality(nxGraph)
		except nx.exception.NetworkXError:
			print fileName + " plotDistancePrePostEigenvector exception... Skipping!"
			return
		diff = set(eigenvectorDict) - set(answersPre)
		for element in diff:
			del(eigenvectorDict[element])
		eigenvectorList = eigenvectorDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			distancePrePostNodesList = []
			for nodeID in eigenvectorDict.iterkeys():
				if answersPre.has_key(nodeID) and answersPost.has_key(nodeID):
					distancePrePostNode = answersPost[nodeID][i] - answersPre[nodeID][i]
					distancePrePostNodesList.append(distancePrePostNode)
			eigenvector = np.array(eigenvectorList)
			distancePrePostNodes = np.array(distancePrePostNodesList)
			pearsonDistanceEigenvector = stats.mstats.pearsonr(eigenvector, distancePrePostNodes)[0]
			m, b = np.polyfit(eigenvector, distancePrePostNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)  + '\nPearson coefficient ' + str(pearsonDistanceEigenvector) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Eigenvector')
			plt.ylabel('Answer distance between pre- and post-interaction')
			plt.plot(eigenvector, distancePrePostNodes, 'ro', eigenvector, m*eigenvector+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
		
	#############################################################
	################## CORRELATION WITH ERRORS ##################
	#############################################################
	
	# Correlation between node answer errors and average degree of the neighborhood of each node
	def plotErrorPreAvgNeighborDegree(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPre_AvgNeighborDegree/'
		degreeDict = nx.average_neighbor_degree(nxGraph, weight='weight')
		diff = set(degreeDict) - set(answersPre)
		for element in diff:
			del(degreeDict[element])
		degreeList = degreeDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in degreeDict.iterkeys():
				if answersPre.has_key(nodeID):
					errorNode = answersPre[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			degrees = np.array(degreeList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorDegree = stats.mstats.pearsonr(errorNodes, degrees)[0]
			m, b = np.polyfit(errorNodes, degrees, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorDegree) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Answers error in the pre-interaction')
			plt.ylabel('Average degree of the neighborhood')
			plt.plot(errorNodes, degrees, 'ro', errorNodes, m*errorNodes+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
	
	# Correlation between node answer errors and average degree of the neighborhood of each node
	def plotErrorPostAvgNeighborDegree(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPost_AvgNeighborDegree/'
		degreeDict = nx.average_neighbor_degree(nxGraph, weight='weight')
		diff = set(degreeDict) - set(answersPost)
		for element in diff:
			del(degreeDict[element])
		degreeList = degreeDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in degreeDict.iterkeys():
				if answersPost.has_key(nodeID):
					errorNode = answersPost[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			degrees = np.array(degreeList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorDegree = stats.mstats.pearsonr(errorNodes, degrees)[0]
			m, b = np.polyfit(errorNodes, degrees, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorDegree) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Answers error in the post-interaction')
			plt.ylabel('Average degree of the neighborhood')
			plt.plot(errorNodes, degrees, 'ro', errorNodes, m*errorNodes+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between node answer errors and node degrees
	def plotErrorPreDegree(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPre_Degree/'
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			degreesList = []
			for nodeID in nxGraph:
				if answersPre.has_key(nodeID):
					errorNode = answersPre[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					degreesList.append(nxGraph.degree(nodeID))
					errorNodesList.append(errorNode)
			degrees = np.array(degreesList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorDegree = stats.mstats.pearsonr(degrees, errorNodes)[0]
			m, b = np.polyfit(degrees, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorDegree) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Degree')
			plt.ylabel('Answers error in the pre-interaction')
			plt.plot(degrees, errorNodes, 'ro', degrees, m*degrees+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between node answer errors and node degrees
	def plotErrorPostDegree(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPost_Degree/'
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			degreesList = []
			for nodeID in nxGraph:
				if answersPost.has_key(nodeID):
					errorNode = answersPost[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					degreesList.append(nxGraph.degree(nodeID))
					errorNodesList.append(errorNode)
			degrees = np.array(degreesList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorDegree = stats.mstats.pearsonr(degrees, errorNodes)[0]
			m, b = np.polyfit(degrees, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorDegree) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Degree')
			plt.ylabel('Answers error in the post-interaction')
			plt.plot(degrees, errorNodes, 'ro', degrees, m*degrees+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	def plotErrorPrePostDegree(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPrePost_Degree/'
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			degreesList = []
			for nodeID in nxGraph:
				if answersPost.has_key(nodeID):
					errorNode = ((answersPost[nodeID][i] - trueAnswers[i]) - (answersPre[nodeID][i] - trueAnswers[i]))
#					print nodeID, errorNode
					degreesList.append(nxGraph.degree(nodeID))
					errorNodesList.append(errorNode)
			degrees = np.array(degreesList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorDegree = stats.mstats.pearsonr(degrees, errorNodes)[0]
			m, b = np.polyfit(degrees, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorDegree) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Degree')
			plt.ylabel('Distance answers error between pre- and post-interaction')
			plt.plot(degrees, errorNodes, 'ro', degrees, m*degrees+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	def plotErrorPrePostWeightedDegree(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPrePost_WeightedDegree/'
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			degreesList = []
			for nodeID in nxGraph:
				if answersPost.has_key(nodeID):
					errorNode = ((answersPost[nodeID][i] - trueAnswers[i] - (answersPre[nodeID][i] - trueAnswers[i])))
#					print nodeID, errorNode
					degreesList.append(nxGraph.degree(nodeID, weight='weight'))
					errorNodesList.append(errorNode)
			degrees = np.array(degreesList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorDegree = stats.mstats.pearsonr(degrees, errorNodes)[1]
			m, b = np.polyfit(degrees, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorDegree) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Weighted Degree')
			plt.ylabel('Distance answers error between pre- and post-interaction')
			plt.plot(degrees, errorNodes, 'ro', degrees, m*degrees+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	def plotErrorPrePostPagerank(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPrePost_Pagerank/'
		pageranksDict = nx.pagerank(nxGraph)
		diff = set(pageranksDict) - set(answersPost)
		for element in diff:
			del(pageranksDict[element])
		pageranksList = pageranksDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in pageranksDict.iterkeys():
				if answersPost.has_key(nodeID):
					errorNode = ((answersPost[nodeID][i] - trueAnswers[i] - (answersPre[nodeID][i] - trueAnswers[i])))
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			pageranks = np.array(pageranksList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorDegree = stats.mstats.pearsonr(pageranks, errorNodes)[1]
			m, b = np.polyfit(pageranks, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorDegree) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Pagerank')
			plt.ylabel('Distance answers error between pre- and post-interaction')
			plt.plot(pageranks, errorNodes, 'ro', pageranks, m*pageranks+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
	
	def saveNXGraph(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + "Graphs/"
		nx.write_gexf(nxGraph, checkDir(savingPath) + fileName + '.gexf')
		plt.figure()
		nx.draw_networkx(nxGraph, pos=nx.spring_layout(nxGraph), font_size=8, node_size=400)
		plt.savefig(checkDir(EXPERIMENT_PATH + OUTCOME_PATH + "Graphs/") + fileName + '.png')
		plt.close()
		
	# Correlation between node node answer errors and node pageranks
	def plotErrorPostPagerank(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPost_Pagerank/'
		pageranksDict = nx.pagerank(nxGraph)
		diff = set(pageranksDict) - set(answersPost)
		for element in diff:
			del(pageranksDict[element])
		pageranksList = pageranksDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in pageranksDict.iterkeys():
				if answersPost.has_key(nodeID):
					errorNode = answersPost[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			pageranks = np.array(pageranksList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorPagerank = stats.mstats.pearsonr(pageranks, errorNodes)[0]
			m, b = np.polyfit(pageranks, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorPagerank) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Pagerank')
			plt.ylabel('Answers error in the post-interaction')
			plt.plot(pageranks, errorNodes, 'ro', pageranks, m*pageranks+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between node node answer errors and node pageranks
	def plotErrorPrePagerank(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPre_Pagerank/'
		pageranksDict = nx.pagerank(nxGraph)
		diff = set(pageranksDict) - set(answersPre)
		for element in diff:
			del(pageranksDict[element])
		pageranksList = pageranksDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in pageranksDict.iterkeys():
				if answersPre.has_key(nodeID):
					errorNode = answersPre[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			pageranks = np.array(pageranksList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorPagerank = stats.mstats.pearsonr(pageranks, errorNodes)[0]
			m, b = np.polyfit(pageranks, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorPagerank) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Pagerank')
			plt.ylabel('Answers error in the pre-interaction')
			plt.plot(pageranks, errorNodes, 'ro', pageranks, m*pageranks+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between node node answer errors and node betweenness
	def plotErrorPreBetweenness(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPre_Betweenness/'
		betweennessDict = nx.betweenness_centrality(nxGraph)
		diff = set(betweennessDict) - set(answersPre)
		for element in diff:
			del(betweennessDict[element])
		betweennessList = betweennessDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in betweennessDict.iterkeys():
				if answersPre.has_key(nodeID):
					errorNode = answersPre[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			betweenness = np.array(betweennessList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorbetweenness = stats.mstats.pearsonr(betweenness, errorNodes)[0]
			m, b = np.polyfit(betweenness, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorbetweenness) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Betweenness Centrality')
			plt.ylabel('Answers error in the pre-interaction')
			plt.plot(betweenness, errorNodes, 'ro', betweenness, m*betweenness+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between node node answer errors and node betweenness
	def plotErrorPostBetweenness(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPost_Betweenness/'
		betweennessDict = nx.betweenness_centrality(nxGraph)
		diff = set(betweennessDict) - set(answersPost)
		for element in diff:
			del(betweennessDict[element])
		betweennessList = betweennessDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in betweennessDict.iterkeys():
				if answersPost.has_key(nodeID):
					errorNode = answersPost[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			betweenness = np.array(betweennessList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorBetweenness = stats.mstats.pearsonr(betweenness, errorNodes)[0]
			m, b = np.polyfit(betweenness, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorBetweenness) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Betweenness Centrality')
			plt.ylabel('Answers error in the post-interaction')
			plt.plot(betweenness, errorNodes, 'ro', betweenness, m*betweenness+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between node node answer errors and node closeness
	def plotErrorPreCloseness(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPre_Closeness/'
		closenessDict = nx.closeness_centrality(nxGraph)
		diff = set(closenessDict) - set(answersPre)
		for element in diff:
			del(closenessDict[element])
		closenessList = closenessDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in closenessDict.iterkeys():
				if answersPre.has_key(nodeID):
					errorNode = answersPre[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			closeness = np.array(closenessList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorCloseness = stats.mstats.pearsonr(closeness, errorNodes)[0]
			m, b = np.polyfit(closeness, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorCloseness) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Closeness Centrality')
			plt.ylabel('Answers error in the pre-interaction')
			plt.plot(closeness, errorNodes, 'ro', closeness, m*closeness+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between node node answer errors and node closeness
	def plotErrorPostCloseness(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPost_Closeness/'
		closenessDict = nx.closeness_centrality(nxGraph)
		diff = set(closenessDict) - set(answersPost)
		for element in diff:
			del(closenessDict[element])
		closenessList = closenessDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in closenessDict.iterkeys():
				if answersPost.has_key(nodeID):
					errorNode = answersPost[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			closeness = np.array(closenessList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorCloseness = stats.mstats.pearsonr(closeness, errorNodes)[0]
			m, b = np.polyfit(closeness, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorCloseness) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Closeness Centrality')
			plt.ylabel('Answers error in the post-interaction')
			plt.plot(closeness, errorNodes, 'ro', closeness, m*closeness+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	# Correlation between node node answer errors and node communicability
	def plotErrorPreCommunicability(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPre_Communicability/'
		communicabilityDict = nx.communicability_centrality(nxGraph)
		diff = set(communicabilityDict) - set(answersPre)
		for element in diff:
			del(communicabilityDict[element])
		communicabilityList = communicabilityDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in communicabilityDict.iterkeys():
				if answersPre.has_key(nodeID):
					errorNode = answersPre[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			communicability = np.array(communicabilityList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorCommunicability = stats.mstats.pearsonr(communicability, errorNodes)[0]
			m, b = np.polyfit(communicability, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorCommunicability) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Communicability Centrality')
			plt.ylabel('Answers error in the pre-interaction')
			plt.plot(communicability, errorNodes, 'ro', communicability, m*communicability+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between node node answer errors and node communicability
	def plotErrorPostCommunicability(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPost_Communicability/'
		communicabilityDict = nx.communicability_centrality(nxGraph)
		diff = set(communicabilityDict) - set(answersPost)
		for element in diff:
			del(communicabilityDict[element])
		communicabilityList = communicabilityDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in communicabilityDict.iterkeys():
				if answersPost.has_key(nodeID):
					errorNode = answersPost[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			communicability = np.array(communicabilityList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorCommunicability = stats.mstats.pearsonr(communicability, errorNodes)[0]
			m, b = np.polyfit(communicability, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorCommunicability) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Communicability Centrality')
			plt.ylabel('Answers error in the post-interaction')
			plt.plot(communicability, errorNodes, 'ro', communicability, m*communicability+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	# Correlation between node node answer errors and node eigenvector
	def plotErrorPreEigenvector(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPre_Eigenvector/'
		eigenvectorDict = {}
		try:
			eigenvectorDict = nx.eigenvector_centrality(nxGraph)
		except nx.exception.NetworkXError:
			print fileName + " plotErrorPreEigenvector exception... Skipping!"
			return
		diff = set(eigenvectorDict) - set(answersPre)
		for element in diff:
			del(eigenvectorDict[element])
		eigenvectorList = eigenvectorDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in eigenvectorDict.iterkeys():
				if answersPre.has_key(nodeID):
					errorNode = answersPre[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			eigenvector = np.array(eigenvectorList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorEigenvector = stats.mstats.pearsonr(eigenvector, errorNodes)[0]
			m, b = np.polyfit(eigenvector, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorEigenvector) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Eigenvector Centrality')
			plt.ylabel('Answers error in the pre-interaction')
			plt.plot(eigenvector, errorNodes, 'ro', eigenvector, m*eigenvector+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()

	# Correlation between node node answer errors and node eigenvector
	def plotErrorPostEigenvector(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_ErrorPost_Eigenvector/'
		eigenvectorDict = {}
		try:
			eigenvectorDict = nx.eigenvector_centrality(nxGraph)
		except nx.exception.NetworkXError:
			print fileName + " plotErrorPostEigenvector exception... Skipping!"
			return
		diff = set(eigenvectorDict) - set(answersPost)
		for element in diff:
			del(eigenvectorDict[element])
		eigenvectorList = eigenvectorDict.values()
		plt.figure(figsize=(20,18))
		for i in range(4):
			errorNodesList = []
			for nodeID in eigenvectorDict.iterkeys():
				if answersPost.has_key(nodeID):
					errorNode = answersPost[nodeID][i] - trueAnswers[i]
#					print nodeID, errorNode
					errorNodesList.append(errorNode)
			eigenvector = np.array(eigenvectorList)
			errorNodes = np.array(errorNodesList)
			pearsonErrorEigenvector = stats.mstats.pearsonr(eigenvector, errorNodes)[0]
			m, b = np.polyfit(eigenvector, errorNodes, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1) + '\nPearson coefficient ' + str(pearsonErrorEigenvector) + '\nm=' + str(m) + ' b=' + str(b)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Nodes Eigenvector Centrality')
			plt.ylabel('Answers error in the post-interaction')
			plt.plot(eigenvector, errorNodes, 'ro', eigenvector, m*eigenvector+b, '--k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	##########################################################################################
	################## CORRELATION WITH AVG ANSWER AND AVG DEGREE/THRESHOLD ##################
	##########################################################################################

	# Correlation between answer average and node degrees
	def plotAvgAnswerAvgDegreeTh(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_AvgAnswersPost_DegreeThreshold/'
		plt.figure(figsize=(20,18))
		# Finding the maximum degree...
		maxDegree = 0
		for nodeID in nxGraph:
			if answersPost.has_key(nodeID):
				if nxGraph.degree(nodeID) > maxDegree:
					maxDegree = nxGraph.degree(nodeID)
		for i in range(4):
			# Finding the average answer for all nodes...
			avgAnswerPostAll = 0.0
			count = 0
			for nodeID in nxGraph:
				if answersPost.has_key(nodeID):
					avgAnswerPostAll += answersPost[nodeID][i]
					count += 1
			avgAnswerPostAll = float(avgAnswerPostAll) / float(count)
			# Main computation...
			diffAvgAllAvgTop = []
			diffAvgAllAvgRand = []
			thresholdDegree = []
			xlabels = []
			for j in range(0, maxDegree+1):
				thresholdDegree.append(j)
				# Finding top-degree nodes with degree >= j
				topDegreeNodes = []
				for nodeID in nxGraph:
					if answersPost.has_key(nodeID) and nxGraph.degree(nodeID) >= j:
						topDegreeNodes.append(nodeID)
				xlabels.append(str(len(topDegreeNodes)) + ';' + str(j))
				# Finding the average answer for top-degree nodes
				avgAnswerPostTop = 0.0
				for nodeID in topDegreeNodes:
					avgAnswerPostTop += float(answersPost[nodeID][i]) / float(len(topDegreeNodes))
				diffTop = abs(avgAnswerPostTop - avgAnswerPostAll)
				diffAvgAllAvgTop.append(diffTop)
				# Finding the average answer for n random nodes
				avgAnswerPostRand = 0.0
				num_simulations = 100
				for z in range(num_simulations): # Number of simulations
					for k in range(len(topDegreeNodes)):
						nodeID = rnd.choice(answersPost.keys())
						avgAnswerPostRand += float(answersPost[nodeID][i]) / float((len(topDegreeNodes) * num_simulations))
				diffRand = abs(avgAnswerPostRand - avgAnswerPostAll)
				diffAvgAllAvgRand.append(diffRand)
			x = np.array(thresholdDegree)
			y_real = np.array(diffAvgAllAvgTop)
			y_sim = np.array(diffAvgAllAvgRand)
			m_real, b_real = np.polyfit(x, y_real, 1)
			m_sim, b_sim = np.polyfit(x, y_sim, 1)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Number of nodes depending on the degree threshold')
			plt.ylabel('|avg(asnwers_top) - avg(answers_all)|')
			plt.xticks(plt.xticks()[0], xlabels)
			plt.plot(x, y_real, 'ro', x, m_real*x+b_real, '--r', x, y_sim, 'bo', x, m_sim*x+b_sim, '--b',)
			plt.xticks(range(len(x)), xlabels)
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	# Correlation between answer stdev and neighborhood degree
	def plotStdevAnswerDegreeTh(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_StdevAnswers_DegreeThreshold/'
		plt.figure(figsize=(20,18))
		# Finding the maximum degree...
		maxDegree = 0
		for nodeID in nxGraph:
			if answersPost.has_key(nodeID):
				if nxGraph.degree(nodeID) > maxDegree:
					maxDegree = nxGraph.degree(nodeID)
		for i in range(4):
			listStdevPre = []
			listStdevPost = []
			thresholdDegree = []
			xlabels = []
			for j in range(0, maxDegree+1):
				thresholdDegree.append(j)
				topDegreeNodes = []
				count = 0
				answersNeighborsPre = set()
				answersNeighborsPost = set()
				for nodeID in nxGraph:
					if nxGraph.degree(nodeID) >= j:
						topDegreeNodes.append(nodeID)
						neighbors = nxGraph.neighbors(nodeID)
						if len(neighbors) > 0:
							for neighbor in neighbors:
								if answersPre.has_key(neighbor) and answersPost.has_key(neighbor):
									answersNeighborsPre.add(answersPre[neighbor][i])
									answersNeighborsPost.add(answersPost[neighbor][i])
							count += 1
				xlabels.append(str(len(topDegreeNodes)) + ';' + str(j))
				meanStdevPre = np.std(list(answersNeighborsPre))
				meanStdevPost = np.std(list(answersNeighborsPost))
				listStdevPre.append(meanStdevPre)
				listStdevPost.append(meanStdevPost)
			maxPre = max(listStdevPre)
			maxPost = max(listStdevPost)
			maxPrePost = max(maxPre, maxPost)
			listStdevPre = [float(x) / float(maxPrePost) for x in listStdevPre]
			listStdevPost = [float(x) / float(maxPrePost) for x in listStdevPost]
			x = np.array(thresholdDegree)
			y_pre = np.array(listStdevPre)
			y_post = np.array(listStdevPost)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Number of nodes depending on the degree threshold')
			plt.ylabel('Standard Deviation N(i) Normalized')
			ymin, ymax = plt.ylim()
			plt.ylim(ymax=ymax*1.2)
			plt.plot(x, y_pre, 'r-', x, y_post, 'b-', linewidth=2.0)
			plt.xticks(range(len(x)), xlabels)
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
	
	
	def computeStatsEdgeAccuracy(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_AccuracyPost_TrueAnswer/'
		
		# PRE
		f_answers_in_pre = open(checkDir(savingPath) + fileName + '_answers_in_pre.csv', 'w')
		f_answers_out_pre = open(checkDir(savingPath) + fileName + '_answers_out_pre.csv', 'w')
		angle = 0
		first_value_in = 0
		first_value_out = 0
		cluster_in = {}
		cluster_out = {}
		for i in range(4):
			cluster_in[i] = {}
			cluster_out[i] = {}
			for edge in nxGraph.edges():
				sourceNode = edge[0]
				targetNode = edge[1]
				if sourceNode in answersPre and targetNode in answersPre:
					if (answersPre[sourceNode][i] <= trueAnswers[i] and answersPre[targetNode][i] >= trueAnswers[i]) or (answersPre[targetNode][i] <= trueAnswers[i] and answersPre[sourceNode][i] >= trueAnswers[i]):
						cluster_in[i][edge] = [answersPre[sourceNode][i], answersPre[targetNode][i]]
					else:
						cluster_out[i][edge] = [answersPre[sourceNode][i], answersPre[targetNode][i]]
			
			if len(cluster_in[i]) == 0:
				cluster_in[i] = None
			if len(cluster_out[i]) == 0:
				cluster_out[i] = None
				
			f = open(checkDir(savingPath) + fileName + '_a' + str(i+1) + '_pre.csv', 'w')
			f.write('IN OUT\n')
			accuracy_in = 0
			accuracy_out = 0
			if cluster_in[i] != None:
				for edge in cluster_in[i]:
					accuracy_in += (abs(cluster_in[i][edge][0] - trueAnswers[i]) + abs(cluster_in[i][edge][1] - trueAnswers[i])) / (2.0 * len(cluster_in[i]) * trueAnswers[i])
			else:
				accuracy_in = None
				
			if cluster_out[i] != None:
				for edge in cluster_out[i]:
					accuracy_out += (abs(cluster_out[i][edge][0] - trueAnswers[i]) + abs(cluster_out[i][edge][1] - trueAnswers[i])) / (2.0 * len(cluster_out[i]) * trueAnswers[i]) 
			else:
				accuracy_out = None
				
			f.write(str(accuracy_in) + ' ' + str(accuracy_out))
			f.close()
			
			if angle == 0:
				first_value_in = accuracy_in
				first_value_out = accuracy_out
				
			f_answers_in_pre.write(str(angle) + ' ' + str(accuracy_in) + '\n')
			f_answers_out_pre.write(str(angle) + ' ' + str(accuracy_out) + '\n')
			angle += 90
		
		f_answers_in_pre.write(str(angle) + ' ' + str(first_value_in) + '\n')
		f_answers_out_pre.write(str(angle) + ' ' + str(first_value_out) + '\n')
		f_answers_in_pre.close()
		f_answers_out_pre.close()
		
		# POST
		f_answers_in_post = open(checkDir(savingPath) + fileName + '_answers_in_post.csv', 'w')
		f_answers_out_post = open(checkDir(savingPath) + fileName + '_answers_out_post.csv', 'w')
		angle = 0
		first_value_in = 0
		first_value_out = 0
		for i in range(4):
			if cluster_in[i] != None:
				for edge in cluster_in[i]:
					sourceNode = edge[0]
					targetNode = edge[1]
					cluster_in[i][edge] = [answersPost[sourceNode][i], answersPost[targetNode][i]]
			if cluster_out[i] != None:
				for edge in cluster_out[i]:
					sourceNode = edge[0]
					targetNode = edge[1]
					cluster_out[i][edge] = [answersPost[sourceNode][i], answersPost[targetNode][i]]
			
			f = open(checkDir(savingPath) + fileName + '_a' + str(i+1) + '_post.csv', 'w')
			f.write('IN OUT\n')
			accuracy_in = 0
			accuracy_out = 0
			if cluster_in[i] != None:
				for edge in cluster_in[i]:
					accuracy_in += (abs(cluster_in[i][edge][0] - trueAnswers[i]) + abs(cluster_in[i][edge][1] - trueAnswers[i])) / (2.0 * len(cluster_in[i]) * trueAnswers[i])
			else:
				accuracy_in = None
				
			if cluster_out[i] != None:
				for edge in cluster_out[i]:
					accuracy_out += (abs(cluster_out[i][edge][0] - trueAnswers[i]) + abs(cluster_out[i][edge][1] - trueAnswers[i])) / (2.0 * len(cluster_out[i]) * trueAnswers[i]) 
			else:
				accuracy_out = None
				
			f.write(str(accuracy_in) + ' ' + str(accuracy_out))
			f.close()
			
			if angle == 0:
				first_value_in = accuracy_in
				first_value_out = accuracy_out
				
			f_answers_in_post.write(str(angle) + ' ' + str(accuracy_in) + '\n')
			f_answers_out_post.write(str(angle) + ' ' + str(accuracy_out) + '\n')
			angle += 90
		
		f_answers_in_post.write(str(angle) + ' ' + str(first_value_in) + '\n')
		f_answers_out_post.write(str(angle) + ' ' + str(first_value_out) + '\n')
		f_answers_in_post.close()
		f_answers_out_post.close()
		
	
	def plotEdgeAccuracyPre(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_AccuracyPre_TrueAnswer/'
		plt.figure(figsize=(20,18))
		
		for i in range(4):
			ids_values_out_list = []
			ids_values_in_list = []
			answers_values_out_list = []
			answers_values_in_list = []
			counter = 1
			for edge in nxGraph.edges():
				sourceNode = edge[0]
				targetNode = edge[1]
				if sourceNode in answersPre and targetNode in answersPre:
					
					if (answersPre[sourceNode][i] <= trueAnswers[i] and  answersPre[targetNode][i] >= trueAnswers[i]) or (answersPre[targetNode][i] <= trueAnswers[i] and answersPre[sourceNode][i] >= trueAnswers[i]):
						answers_values_in_list.append(answersPre[sourceNode][i])
						answers_values_in_list.append(answersPre[targetNode][i])
						ids_values_in_list.append(counter)
						ids_values_in_list.append(counter)
					else:
						answers_values_out_list.append(answersPre[sourceNode][i])
						answers_values_out_list.append(answersPre[targetNode][i])
						ids_values_out_list.append(counter)
						ids_values_out_list.append(counter)
					
					counter += 1
			
			ids_values_out = np.array(ids_values_out_list)
			answers_values_out = np.array(answers_values_out_list)
			ids_values_in = np.array(ids_values_in_list)
			answers_values_in = np.array(answers_values_in_list)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Edges')
			plt.ylabel('Answer value')
			plt.plot(ids_values_in, answers_values_in, 'bo', ids_values_out, answers_values_out, 'ro', range((len(ids_values_out) + len(ids_values_in))/2), [trueAnswers[i]] * ((len(ids_values_out) + len(ids_values_in))/2), '-k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
	
	
	def plotEdgeAccuracyPost(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_AccuracyPost_TrueAnswer/'
		plt.figure(figsize=(20,18))
		
		for i in range(4):
			ids_values_out_list = []
			ids_values_in_list = []
			answers_values_out_list = []
			answers_values_in_list = []
			counter = 1
			for edge in nxGraph.edges():
				sourceNode = edge[0]
				targetNode = edge[1]
				if sourceNode in answersPost and targetNode in answersPost:
					
					if (answersPost[sourceNode][i] <= trueAnswers[i] and  answersPost[targetNode][i] >= trueAnswers[i]) or (answersPost[targetNode][i] <= trueAnswers[i] and answersPost[sourceNode][i] >= trueAnswers[i]):
						answers_values_in_list.append(answersPost[sourceNode][i])
						answers_values_in_list.append(answersPost[targetNode][i])
						ids_values_in_list.append(counter)
						ids_values_in_list.append(counter)
					else:
						answers_values_out_list.append(answersPost[sourceNode][i])
						answers_values_out_list.append(answersPost[targetNode][i])
						ids_values_out_list.append(counter)
						ids_values_out_list.append(counter)
					
					counter += 1
			
			ids_values_out = np.array(ids_values_out_list)
			answers_values_out = np.array(answers_values_out_list)
			ids_values_in = np.array(ids_values_in_list)
			answers_values_in = np.array(answers_values_in_list)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Edges')
			plt.ylabel('Answer value')
			plt.plot(ids_values_in, answers_values_in, 'bo', ids_values_out, answers_values_out, 'ro', range((len(ids_values_out) + len(ids_values_in))/2), [trueAnswers[i]] * ((len(ids_values_out) + len(ids_values_in))/2), '-k')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
	
	def plotEdgeAccuracyPreAvg(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_AccuracyPreAvg_TrueAnswer/'
		plt.figure(figsize=(20,18))
		
		for i in range(4):
			ids_values_out_list = []
			ids_values_in_list = []
			answers_values_out_list = []
			answers_values_in_list = []
			weights_list = []
			counter = 1
			for edge in nxGraph.edges():
				sourceNode = edge[0]
				targetNode = edge[1]
				if sourceNode in answersPre and targetNode in answersPre:
					weights_list.append(nxGraph[sourceNode][targetNode]['weight'] * 10000)
					valueAnswer = (answersPre[sourceNode][i] + answersPre[targetNode][i]) / 2.0
					if (answersPre[sourceNode][i] <= trueAnswers[i] and answersPre[targetNode][i] >= trueAnswers[i]) or (answersPre[targetNode][i] <= trueAnswers[i] and answersPre[sourceNode][i] >= trueAnswers[i]):
						answers_values_in_list.append(valueAnswer)
						ids_values_in_list.append(counter)
					else:
						answers_values_out_list.append(valueAnswer)
						ids_values_out_list.append(counter)
					
					counter += 1
			
			weights = np.array(weights_list)
			ids_values_out = np.array(ids_values_out_list)
			answers_values_out = np.array(answers_values_out_list)
			ids_values_in = np.array(ids_values_in_list)
			answers_values_in = np.array(answers_values_in_list)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Edges')
			plt.ylabel('Avg(answer) for each edge')
			plt.plot(range(len(ids_values_out) + len(ids_values_in)), [trueAnswers[i]] * (len(ids_values_out) + len(ids_values_in)), '-k')
			plt.scatter(ids_values_in, answers_values_in, s=weights, c='blue')
			plt.scatter(ids_values_out, answers_values_out, s=weights, c='red')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
	
	
	def plotEdgeAccuracyPostAvg(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_AccuracyPostAvg_TrueAnswer/'
		plt.figure(figsize=(20,18))
		
		for i in range(4):
			ids_values_out_list = []
			ids_values_in_list = []
			answers_values_out_list = []
			answers_values_in_list = []
			weights_list = []
			counter = 1
			for edge in nxGraph.edges():
				sourceNode = edge[0]
				targetNode = edge[1]
				if sourceNode in answersPost and targetNode in answersPost:
					weights_list.append(nxGraph[sourceNode][targetNode]['weight'] * 10000)
					valueAnswer = (answersPost[sourceNode][i] + answersPost[targetNode][i]) / 2.0
					if (answersPost[sourceNode][i] <= trueAnswers[i] and answersPost[targetNode][i] >= trueAnswers[i]) or (answersPost[targetNode][i] <= trueAnswers[i] and answersPost[sourceNode][i] >= trueAnswers[i]):
						answers_values_in_list.append(valueAnswer)
						ids_values_in_list.append(counter)
					else:
						answers_values_out_list.append(valueAnswer)
						ids_values_out_list.append(counter)
					
					counter += 1
			
			weights = np.array(weights_list)
			ids_values_out = np.array(ids_values_out_list)
			answers_values_out = np.array(answers_values_out_list)
			ids_values_in = np.array(ids_values_in_list)
			answers_values_in = np.array(answers_values_in_list)
			title = 'Graph ' + fileName + ' - Answer ' + str(i+1)
			plt.subplot(221+i)
			plt.title(title)
			plt.xlabel('Edges')
			plt.ylabel('Avg(answer) for each edge')
			plt.plot(range(len(ids_values_out) + len(ids_values_in)), [trueAnswers[i]] * (len(ids_values_out) + len(ids_values_in)), '-k')
			plt.scatter(ids_values_in, answers_values_in, s=weights, c='blue')
			plt.scatter(ids_values_out, answers_values_out, s=weights, c='red')
		plt.savefig(checkDir(savingPath) + fileName + '.png', format='png')
		plt.close()
		
		
	def plotNodeAccuracyAll(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_AccuracyAll_TrueAnswer/'
		
		for i in range(4):
			e = open(savingPath + 'err' + str(i+1) + '.csv', 'w')
			f = open(savingPath + 'q' + str(i+1) + '.csv', 'w')
			g = open(savingPath + 'stdev' + str(i+1) + '.csv', 'w')
			answers_pre = []
			answers_post = []
			
			counter = 1
			err_pre = []
			err_post = []
			x_node_pre_list = []
			x_node_post_list = []
			N_x_node_list = []
			y_node_pre_list = []
			y_node_post_list = []
			N_y_node_list = []
			for node in nxGraph.nodes():
				if node in answersPre and node in answersPost:
					err_pre.append(abs(answersPreNorm[node][i] - 1))
					err_post.append(abs(answersPostNorm[node][i] - 1))
					
					answers_pre.append(answersPreNorm[node][i])
					answers_post.append(answersPostNorm[node][i])
					
					#PRE
					x_node_pre_list.append(counter)
					y_node_pre_list.append(answersPre[node][i])
					#POST
					x_node_post_list.append(counter)
					y_node_post_list.append(answersPost[node][i])
					#NEIGHBORS, POST
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPost:
							N_x_node_list.append(counter)
							N_y_node_list.append(answersPost[neighbor][i])
					
					counter += 1
			
			e.write('R1 R2\n')
			e.write(str(np.mean(err_pre)) + ' ' + str(np.mean(err_post)))
			e.close()
					
			f.write('R1 R2\n')
			f.write(str(np.mean(answers_pre)) + ' ' + str(np.mean(answers_post)))
			f.close()
			
			g.write('R1 R2\n')
			g.write(str(np.std(answers_pre)) + ' ' + str(np.std(answers_post)))
			g.close()
						
			x_node_pre = np.array(x_node_pre_list)
			x_node_post = np.array(x_node_post_list)
			N_x_src = np.array(N_x_node_list)
			y_node_pre = np.array(y_node_pre_list)
			y_node_post = np.array(y_node_post_list)
			N_y_src = np.array(N_y_node_list)
			
			plt.figure(figsize=(20,18))
			plt.xlabel('Nodes')
			plt.ylabel('Answer values')
			plt.plot(range(counter), [trueAnswers[i]] * counter, '-k', x_node_pre, y_node_pre, 'rD', x_node_post, y_node_post, 'bs', N_x_src, N_y_src, 'kx')
			max_node_pre = max(y_node_pre)
			max_node_post = max(y_node_post)
			max_y = max(max_node_pre, max_node_post) + (0.1 * max(max_node_pre, max_node_post))
			plt.ylim(0, max_y)
			plt.savefig(checkDir(savingPath) + fileName + '_a' + str(i+1) + '.png', format='png')
			plt.close()
		
		
	def plotEdgeAccuracyAll(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Edges_AccuracyAll_TrueAnswer/'
		
		for i in range(4):
			counter = 1
			x_src_pre_list = []
			x_dst_pre_list = []
			x_src_post_list = []
			x_dst_post_list = []
			N_x_src_list = []
			N_x_dst_list = []
			y_src_pre_list = []
			y_dst_pre_list = []
			y_src_post_list = []
			y_dst_post_list = []
			N_y_src_list = []
			N_y_dst_list = []
			for edge in nxGraph.edges():
				sourceNode = edge[0]
				targetNode = edge[1]
				if sourceNode in answersPre and targetNode in answersPre and sourceNode in answersPost and targetNode in answersPost:
					#PRE
					x_src_pre_list.append(counter)
					x_dst_pre_list.append(counter)
					y_src_pre_list.append(answersPre[sourceNode][i])
					y_dst_pre_list.append(answersPre[targetNode][i])
					#POST
					x_src_post_list.append(counter)
					x_dst_post_list.append(counter)
					y_src_post_list.append(answersPost[sourceNode][i])
					y_dst_post_list.append(answersPost[targetNode][i])
					#NEIGHBORS, POST
					for neighbor in nxGraph.neighbors(sourceNode):
						if neighbor in answersPost and neighbor != targetNode:
							N_x_src_list.append(counter)
							N_y_src_list.append(answersPost[neighbor][i])
					for neighbor in nxGraph.neighbors(targetNode):
						if neighbor in answersPost and neighbor != sourceNode:
							N_x_dst_list.append(counter)
							N_y_dst_list.append(answersPost[neighbor][i])
					
					counter += 1
			
			x_src_pre = np.array(x_src_pre_list)
			x_dst_pre = np.array(x_dst_pre_list)
			x_src_post = np.array(x_src_post_list)
			x_dst_post = np.array(x_dst_post_list)
			N_x_src = np.array(N_x_src_list)
			N_x_dst = np.array(N_x_dst_list)
			y_src_pre = np.array(y_src_pre_list)
			y_dst_pre = np.array(y_dst_pre_list)
			y_src_post = np.array(y_src_post_list)
			y_dst_post = np.array(y_dst_post_list)
			N_y_src = np.array(N_y_src_list)
			N_y_dst = np.array(N_y_dst_list)
			
			plt.figure(figsize=(20,18))
			plt.xlabel('Edges and Neighbors')
			plt.ylabel('Answer values')
			plt.plot(range(counter), [trueAnswers[i]] * counter, '-k', x_src_pre, y_src_pre, 'rD', x_dst_pre, y_dst_pre, 'rD', x_src_post, y_src_post, 'bs', x_dst_post, y_dst_post, 'bs', N_x_src, N_y_src, 'kx',  N_x_dst, N_y_dst, 'kx')
			max_src_pre = max(y_src_pre)
			max_dst_pre = max(y_dst_pre)
			max_src_post = max(y_src_post)
			max_dst_post = max(y_dst_post)
			max_y = max(max_src_pre, max_dst_pre, max_src_post, max_dst_post) + (0.1 * max(max_src_pre, max_dst_pre, max_src_post, max_dst_post))
			plt.ylim(0, max_y)
			plt.savefig(checkDir(savingPath) + fileName + '_a' + str(i+1) + '.png', format='png')
			plt.close()
			
			
	def residualsFirstModel(self, p, a_2, a_1, sum_N, d):
		alpha = p
		a_1 = np.array(a_1)
		a_2 = np.array(a_2)
		sum_N = np.array(sum_N)
		d = np.array(d)
		
		err = np.linalg.norm(((alpha * a_1 + sum_N) / (d + alpha)) - a_2)
		return err
			
	
	def fitModel1Answer(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_Estimation_Model1Answer/'
		answers_params = {}
		
		f = open(checkDir(savingPath) + 'distances_' + fileName + '.csv', 'w')
		f.write('answers, alpha, err, l2\n')
		
		for i in range(4):
			answers_pre_list = []
			answers_post_list = []
			sum_N_list = []
			num_neighbors_list = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					sum_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					answers_pre_list.append(answersPreNorm[node][i])
					answers_post_list.append(answersPostNorm[node][i])
					sum_N_list.append(sum_N)
					num_neighbors_list.append(num_neighbors)
			p0 = 1.0
			least_square = leastsq(self.residualsFirstModel, p0, args=(answers_post_list, answers_pre_list, sum_N_list, num_neighbors_list), maxfev=1000)
			alpha = least_square[0][0]
			l2 = self.residualsFirstModel(alpha, answers_post_list, answers_pre_list, sum_N_list, num_neighbors_list)
			err = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					sum_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					est = (alpha * answersPreNorm[node][i] + sum_N) / (num_neighbors + alpha)
					if answersPostNorm[node][i] > 0:
						err.append(abs(est - 1))
			#print str(i+1) + ', ' + str(alpha) + ', ' + str(np.mean(err)) + ', ' + str(l2)
			f.write(str(i+1) + ', ' + str(alpha) + ', ' + str(np.mean(err)) + ', ' + str(l2) + '\n')
			answers_params[i] = alpha
		f.close()
		return answers_params
			
	
	def fitModel1User(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_Estimation_Model1User/'
		answers_params = {}
		
		f = open(checkDir(savingPath) + 'distances_' + fileName + '.csv', 'w')
		f.write('user, err, l2\n')
		
		avg_error = {}
		total_l2 = {}
		
		for node in nxGraph.nodes():
			answers_pre_list = []
			answers_post_list = []
			sum_N_list = []
			num_neighbors_list = []
			if node in answersPreNorm and node in answersPostNorm:
				for i in range(4):
					sum_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					answers_pre_list.append(answersPreNorm[node][i])
					answers_post_list.append(answersPostNorm[node][i])
					sum_N_list.append(sum_N)
					num_neighbors_list.append(num_neighbors)
								
				p0 = 1.0
				least_square = leastsq(self.residualsFirstModel, p0, args=(answers_post_list, answers_pre_list, sum_N_list, num_neighbors_list), maxfev=1000)
				alpha = least_square[0][0]
				
				answers_params[node] = alpha
				
				for i in range(4):
					sum_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					est = (alpha * answersPreNorm[node][i] + sum_N) / (num_neighbors + alpha)				
					if not i in avg_error:
						avg_error[i] = []
					if not i in total_l2:
						total_l2[i] = 0.0
					if answersPostNorm[node][i] > 0:
						avg_error[i].append(abs(est - 1))
					total_l2[i] += math.pow(est - answersPostNorm[node][i], 2)
		for i in range(4):
			total_l2[i] = math.sqrt(total_l2[i])
			f.write(str(i+1) + ', ' + str(np.mean(avg_error[i])) + ', ' + str(total_l2[i]) + '\n')
		f.close()
		return answers_params
			
	
	def fitModel1UserTraining(self, fileName, nxGraph, training):
		answers_params = {}
		
		for node in nxGraph.nodes():
			answers_pre_list = []
			answers_post_list = []
			sum_N_list = []
			num_neighbors_list = []
			if node in answersPreNorm and node in answersPostNorm:
				for i in training:
					sum_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					answers_pre_list.append(answersPreNorm[node][i])
					answers_post_list.append(answersPostNorm[node][i])
					sum_N_list.append(sum_N)
					num_neighbors_list.append(num_neighbors)
								
				p0 = 1.0
				least_square = leastsq(self.residualsFirstModel, p0, args=(answers_post_list, answers_pre_list, sum_N_list, num_neighbors_list), maxfev=1000)
				answers_params[node] = least_square[0][0]
		return answers_params

	
	def model1AnswerEstimation(self, fileName, nxGraph, answers_params):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_Estimation_Model1Answer/'
		print '\n*** Model 1 - Answer ***'
		
		f = open(checkDir(savingPath) + 'stats_' + fileName + '.csv', 'w')
		
		for i in range(4):
			counter = 0
			alpha = answers_params[i]
			
			nodes_pre_answers_list = []
			nodes_post_answers_list = []
			nodes_est_answers_list = []
			neighbors_answers_x_list = []
			neighbors_answers_y_list = []
			neighbors_avg_answers_list = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					nodes_pre_answers_list.append(answersPreNorm[node][i])
					nodes_post_answers_list.append(answersPostNorm[node][i])
					sum_N = 0.0
					avg_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							neighbors_answers_x_list.append(counter)
							neighbors_answers_y_list.append(answersPreNorm[neighbor][i])
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					if num_neighbors > 0:
						avg_N = sum_N / num_neighbors
					neighbors_avg_answers_list.append(avg_N)
					est = (alpha * answersPreNorm[node][i] + sum_N) / (num_neighbors + alpha)
					nodes_est_answers_list.append(est)
					
					counter += 1
			
			nodes_pre_answers = np.array(nodes_pre_answers_list)
			nodes_post_answers = np.array(nodes_post_answers_list)
			nodes_est_answers = np.array(nodes_est_answers_list)
			neighbors_answers_x = np.array(neighbors_answers_x_list)
			neighbors_answers_y = np.array(neighbors_answers_y_list)
			neighbors_avg_answers = np.array(neighbors_avg_answers_list)
			
			err_truth = np.mean([abs(x-1) for x in nodes_est_answers])
			err_p1 = np.mean(abs(nodes_est_answers-nodes_post_answers))
			
			print str(i+1) + ', Alpha: ' + str(alpha) + ', Mean: ' + str(np.mean(nodes_est_answers)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1)
			f.write(str(i+1) + ', Alpha: ' + str(alpha) + ', Mean: ' + str(np.mean(nodes_est_answers)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1) + '\n')
			
			plt.figure(figsize=(20,18))
			plt.title('Model 1 (unweighted): alpha ' + str(alpha))
			plt.xlabel('Nodes')
			plt.ylabel('Answer values')
			plt.plot(range(counter), [trueAnswers[i]] * counter, '-k', range(counter), nodes_est_answers, 'kD', range(counter), nodes_pre_answers, 'bs', range(counter), nodes_post_answers, 'r^', range(counter), neighbors_avg_answers, 'go', neighbors_answers_x, neighbors_answers_y, 'kx')
			max_node_pre = max(nodes_pre_answers)
			max_node_post = max(nodes_post_answers)
			max_node_est = max(nodes_est_answers)
			max_neighbors = max(neighbors_avg_answers)
			max_y = max(max_node_pre, max_node_post, max_node_est, max_neighbors) + (0.1 * max(max_node_pre, max_node_post, max_node_est, max_neighbors))
			plt.ylim(0, max_y)
			plt.xticks(range(counter))
			ax = plt.gca()
			ax.xaxis.grid(b=True, which='both', color='#777777', linestyle='-')
			plt.savefig(checkDir(savingPath) + fileName + '_a' + str(i+1) + '_alpha' + str(alpha) + '.png', format='png')
			plt.close()
		f.close()

	
	def model1UserEstimation(self, fileName, nxGraph, answers_params):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_Estimation_Model1User/'
		print '\n*** Model 1 - User ***'
		
		f = open(checkDir(savingPath) + 'stats_' + fileName + '.csv', 'w')
		
		for i in range(4):
			counter = 0
			
			nodes_pre_answers_list = []
			nodes_post_answers_list = []
			nodes_est_answers_list = []
			neighbors_answers_x_list = []
			neighbors_answers_y_list = []
			neighbors_avg_answers_list = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					alpha = answers_params[node]
					nodes_pre_answers_list.append(answersPreNorm[node][i])
					nodes_post_answers_list.append(answersPostNorm[node][i])
					sum_N = 0.0
					avg_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							neighbors_answers_x_list.append(counter)
							neighbors_answers_y_list.append(answersPreNorm[neighbor][i])
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					if num_neighbors > 0:
						avg_N = sum_N / num_neighbors
					neighbors_avg_answers_list.append(avg_N)
					est = (alpha * answersPreNorm[node][i] + sum_N) / (num_neighbors + alpha)
					nodes_est_answers_list.append(est)
					
					counter += 1
			
			nodes_pre_answers = np.array(nodes_pre_answers_list)
			nodes_post_answers = np.array(nodes_post_answers_list)
			nodes_est_answers = np.array(nodes_est_answers_list)
			neighbors_answers_x = np.array(neighbors_answers_x_list)
			neighbors_answers_y = np.array(neighbors_answers_y_list)
			neighbors_avg_answers = np.array(neighbors_avg_answers_list)
			
			err_truth = np.mean([abs(x-1) for x in nodes_est_answers])
			err_p1 = np.mean(abs(nodes_est_answers-nodes_post_answers))
			
			print str(i+1) + ', Mean: ' + str(np.mean(nodes_est_answers)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1)
			f.write(str(i+1) + ', Mean: ' + str(np.mean(nodes_est_answers)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1) + '\n')
			
			plt.figure(figsize=(20,18))
			plt.title('Model 1 (unweighted): alpha ' + str(alpha))
			plt.xlabel('Nodes')
			plt.ylabel('Answer values')
			plt.plot(range(counter), [trueAnswers[i]] * counter, '-k', range(counter), nodes_est_answers, 'kD', range(counter), nodes_pre_answers, 'bs', range(counter), nodes_post_answers, 'r^', range(counter), neighbors_avg_answers, 'go', neighbors_answers_x, neighbors_answers_y, 'kx')
			max_node_pre = max(nodes_pre_answers)
			max_node_post = max(nodes_post_answers)
			max_node_est = max(nodes_est_answers)
			max_neighbors = max(neighbors_avg_answers)
			max_y = max(max_node_pre, max_node_post, max_node_est, max_neighbors) + (0.1 * max(max_node_pre, max_node_post, max_node_est, max_neighbors))
			plt.ylim(0, max_y)
			plt.xticks(range(counter))
			ax = plt.gca()
			ax.xaxis.grid(b=True, which='both', color='#777777', linestyle='-')
			plt.savefig(checkDir(savingPath) + fileName + '_a' + str(i+1) + '.png', format='png')
			plt.close()
		f.close()

	
	def model1UserEstimationTraining(self, fileName, nxGraph, answers_params, training):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_Estimation_Model1User_Training/'
		
		training_str = re.sub(r' ', ',', str(np.array(training)+1))
		print '\n*** Model 1 - User [Training: ' + str(training) + '] ***'
		
		f = open(checkDir(savingPath) + 'stats_' + fileName + '_' + training_str + '.csv', 'w')
		
		for i in range(4):
			counter = 0
			
			nodes_pre_answers_list = []
			nodes_post_answers_list = []
			nodes_est_answers_list = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					alpha = answers_params[node]
					nodes_pre_answers_list.append(answersPreNorm[node][i])
					nodes_post_answers_list.append(answersPostNorm[node][i])
					sum_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					est = (alpha * answersPreNorm[node][i] + sum_N) / (num_neighbors + alpha)
					nodes_est_answers_list.append(est)
					
					counter += 1
			
			nodes_pre_answers = np.array(nodes_pre_answers_list)
			nodes_post_answers = np.array(nodes_post_answers_list)
			nodes_est_answers = np.array(nodes_est_answers_list)
			
			err_truth = np.mean([abs(x-1) for x in nodes_est_answers])
			err_p1 = np.mean(abs(nodes_est_answers-nodes_post_answers))
			
			print str(i+1) + ', Mean: ' + str(np.mean(nodes_est_answers)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1)
			f.write(str(i+1) + ', Mean: ' + str(np.mean(nodes_est_answers)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1) + '\n')
		f.close()
			
	
	def fitModel2Answer(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_Estimation_Model2Answer/'
		answers_params = {}
		
		f = open(checkDir(savingPath) + 'distances_' + fileName + '.csv', 'w')
		f.write('answers, alpha, err, l2\n')
		
		for i in range(4):
			answers_pre_list = []
			answers_post_list = []
			sum_N_list = []
			num_neighbors_list = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					sum_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							weight = nxGraph[node][neighbor]['weight']
							num_neighbors += weight
							sum_N += (answersPreNorm[neighbor][i] * weight)
					answers_pre_list.append(answersPreNorm[node][i])
					answers_post_list.append(answersPostNorm[node][i])
					sum_N_list.append(sum_N)
					num_neighbors_list.append(num_neighbors)
			
			p0 = 1.0
			least_square = leastsq(self.residualsFirstModel, p0, args=(answers_post_list, answers_pre_list, sum_N_list, num_neighbors_list), maxfev=1000)
			alpha = least_square[0][0]
			l2 = self.residualsFirstModel(alpha, answers_post_list, answers_pre_list, sum_N_list, num_neighbors_list)
			err = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					sum_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					est = (alpha * answersPreNorm[node][i] + sum_N) / (num_neighbors + alpha)
					if answersPostNorm[node][i] > 0:
						err.append(abs(est - 1))
			#print str(i+1) + ', ' + str(alpha) + ', ' + str(np.mean(err)) + ', ' + str(l2)
			f.write(str(i+1) + ', ' + str(alpha) + ', ' + str(np.mean(err)) + ', ' + str(l2) + '\n')
			answers_params[i] = alpha
		f.close()
		return answers_params
		
	
	def model2AnswerEstimation(self, fileName, nxGraph, answers_params):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_Estimation_Model2Answer/'
		print '\n*** Model 2 - Answer ***'
		
		f = open(checkDir(savingPath) + 'stats_' + fileName + '.csv', 'w')
		
		for i in range(4):
			counter = 0
			alpha = answers_params[i]
			
			nodes_pre_answers_list = []
			nodes_post_answers_list = []
			nodes_est_answers_list = []
			neighbors_answers_x_list = []
			neighbors_answers_y_list = []
			neighbors_avg_answers_list = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					nodes_pre_answers_list.append(answersPreNorm[node][i])
					nodes_post_answers_list.append(answersPostNorm[node][i])
					sum_N = 0.0
					avg_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							neighbors_answers_x_list.append(counter)
							neighbors_answers_y_list.append(answersPreNorm[neighbor][i])
							weight = nxGraph[node][neighbor]['weight']
							num_neighbors += weight
							sum_N += (answersPreNorm[neighbor][i] * weight)
					if num_neighbors > 0:
						avg_N = sum_N / num_neighbors
					neighbors_avg_answers_list.append(avg_N)
					est = (alpha * answersPreNorm[node][i] + sum_N) / (num_neighbors + alpha)
					nodes_est_answers_list.append(est)
					
					counter += 1
			
			nodes_pre_answers = np.array(nodes_pre_answers_list)
			nodes_post_answers = np.array(nodes_post_answers_list)
			nodes_est_answers = np.array(nodes_est_answers_list)
			neighbors_answers_x = np.array(neighbors_answers_x_list)
			neighbors_answers_y = np.array(neighbors_answers_y_list)
			neighbors_avg_answers = np.array(neighbors_avg_answers_list)
			
			err_truth = np.mean([abs(x-1) for x in nodes_est_answers])
			err_p1 = np.mean(abs(nodes_est_answers-nodes_post_answers))
			
			print str(i+1) + ', Alpha: ' + str(alpha) + ', Mean: ' + str(np.mean(nodes_est_answers)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1)
			f.write(str(i+1) + ', Alpha: ' + str(alpha) + ', Mean: ' + str(np.mean(nodes_est_answers)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1) + '\n')
			
			plt.figure(figsize=(20,18))
			plt.title('Model 2 (weighted): alpha ' + str(alpha))
			plt.xlabel('Nodes')
			plt.ylabel('Answer values')
			plt.plot(range(counter), [trueAnswers[i]] * counter, '-k', range(counter), nodes_est_answers, 'kD', range(counter), nodes_pre_answers, 'bs', range(counter), nodes_post_answers, 'r^', range(counter), neighbors_avg_answers, 'go', neighbors_answers_x, neighbors_answers_y, 'kx')
			max_node_pre = max(nodes_pre_answers)
			max_node_post = max(nodes_post_answers)
			max_node_est = max(nodes_est_answers)
			max_neighbors = max(neighbors_avg_answers)
			max_y = max(max_node_pre, max_node_post, max_node_est, max_neighbors) + (0.1 * max(max_node_pre, max_node_post, max_node_est, max_neighbors))
			plt.ylim(0, max_y)
			plt.xticks(range(counter))
			ax = plt.gca()
			ax.xaxis.grid(b=True, which='both', color='#777777', linestyle='-')
			plt.savefig(checkDir(savingPath) + fileName + '_a' + str(i+1) + '_alpha' + str(alpha) + '.png', format='png')
			plt.close()
		f.close()
			
			
	def residualsSecondModel(self, p, a_2, a_1, sum_N, d, T):
		alpha, beta = p
		a_1 = np.array(a_1)
		a_2 = np.array(a_2)
		sum_N = np.array(sum_N)
		d = np.array(d)
		
		e = np.array([min(x, y) for x, y in zip(abs(a_1-float(T))/float(T), [1]*len(a_1))])
		m1 = (alpha * a_1 + sum_N) / (d + alpha)
		func = beta * m1 + (1 - beta) * ((1 - e) * a_1 + e * m1)
		err = np.linalg.norm(func - a_2)
		return (err, err)
	
	
	def fitModel3Answer(self, fileName, nxGraph):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_Estimation_Model3Answer/'
		answers_params = {}
		
		f = open(checkDir(savingPath) + 'distances_' + fileName + '.csv', 'w')
		f.write('answers, alpha, beta, distance\n')
		
		for i in range(4):
			answers_pre_list = []
			answers_post_list = []
			sum_N_list = []
			num_neighbors_list = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					sum_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					answers_pre_list.append(answersPreNorm[node][i])
					answers_post_list.append(answersPostNorm[node][i])
					sum_N_list.append(sum_N)
					num_neighbors_list.append(num_neighbors)
			
			p0 = [1.0, 1.0]
			least_square = leastsq(self.residualsSecondModel, p0, args=(answers_post_list, answers_pre_list, sum_N_list, num_neighbors_list, trueAnswers[i]), maxfev=1000)
			alpha = least_square[0][0]
			beta = least_square[0][1]
			err = self.residualsSecondModel([alpha, beta], answers_post_list, answers_pre_list, sum_N_list, num_neighbors_list, trueAnswers[i])
			#print str(i+1) + ', ' + str(alpha) + ', ' + str(beta) + ', ' + str(err[0])
			f.write(str(i+1) + ', ' + str(alpha) + ', ' + str(beta) + ', ' + str(err[0]) + '\n')
			answers_params[i] = (alpha, beta)
		f.close()
		return answers_params
		
	
	def model3AnswerEstimation(self, fileName, nxGraph, answers_params):
		savingPath = EXPERIMENT_PATH + OUTCOME_PATH + 'Nodes_Estimation_Model3Answer/'
		print '\n*** Model 3 - Answer ***'
		
		for i in range(4):
			counter = 0
			alpha = answers_params[i][0]
			beta = answers_params[i][1]
			
			nodes_pre_answers_list = []
			nodes_post_answers_list = []
			nodes_est_answers_list = []
			neighbors_answers_x_list = []
			neighbors_answers_y_list = []
			neighbors_avg_answers_list = []
			for node in nxGraph.nodes():
				if node in answersPreNorm and node in answersPostNorm:
					nodes_pre_answers_list.append(answersPreNorm[node][i])
					nodes_post_answers_list.append(answersPostNorm[node][i])
					sum_N = 0.0
					avg_N = 0.0
					num_neighbors = 0.0
					for neighbor in nxGraph.neighbors(node):
						if neighbor in answersPreNorm and neighbor in answersPostNorm:
							neighbors_answers_x_list.append(counter)
							neighbors_answers_y_list.append(answersPreNorm[neighbor][i])
							num_neighbors += 1
							sum_N += answersPreNorm[neighbor][i]
					if num_neighbors > 0:
						avg_N = sum_N / num_neighbors
					neighbors_avg_answers_list.append(avg_N)
					e = min(1, abs(answersPreNorm[node][i] - float(trueAnswers[i]))/float(trueAnswers[i]))
					m1 = (alpha * answersPreNorm[node][i] + sum_N) / (num_neighbors + alpha)
					est = beta * m1 + (1 - beta) * ((1 - e) * answersPreNorm[node][i] + e * m1)
					nodes_est_answers_list.append(est)
					
					counter += 1
			
			nodes_pre_answers = np.array(nodes_pre_answers_list)
			nodes_post_answers = np.array(nodes_post_answers_list)
			nodes_est_answers = np.array(nodes_est_answers_list)
			neighbors_answers_x = np.array(neighbors_answers_x_list)
			neighbors_answers_y = np.array(neighbors_answers_y_list)
			neighbors_avg_answers = np.array(neighbors_avg_answers_list)
			
			err_truth = np.mean([abs(x-1) for x in nodes_est_answers])
			err_p1 = np.mean(abs(nodes_est_answers-nodes_post_answers))
			
			print str(i+1) + ', Mean: ' + str(np.mean(nodes_est_answers)) + ', Err_GT: ' + str(err_truth) + ', Err_p1: ' + str(err_p1)
			
			plt.figure(figsize=(20,18))
			plt.title('Model 3: alpha ' + str(alpha) + ', beta ' + str(beta))
			plt.xlabel('Nodes')
			plt.ylabel('Answer values')
			plt.plot(range(counter), [trueAnswers[i]] * counter, '-k', range(counter), nodes_est_answers, 'kD', range(counter), nodes_pre_answers, 'bs', range(counter), nodes_post_answers, 'r^', range(counter), neighbors_avg_answers, 'go', neighbors_answers_x, neighbors_answers_y, 'kx')
			max_node_pre = max(nodes_pre_answers)
			max_node_post = max(nodes_post_answers)
			max_node_est = max(nodes_est_answers)
			max_neighbors = max(neighbors_avg_answers)
			max_y = max(max_node_pre, max_node_post, max_node_est, max_neighbors) + (0.1 * max(max_node_pre, max_node_post, max_node_est, max_neighbors))
			plt.ylim(0, max_y)
			plt.xticks(range(counter))
			ax = plt.gca()
			ax.xaxis.grid(b=True, which='both', color='#777777', linestyle='-')
			plt.savefig(checkDir(savingPath) + fileName + '_a' + str(i+1) + '_alpha' + str(alpha) + '_beta' + str(beta) + '.png', format='png')
			plt.close()
			

def main():
	# Answers
	answersProcessing()
	graphFilePearson = {}
	csvFiles = glob.glob(EXPERIMENT_PATH + GRAPHS_TYPE)
	for graphFileName in csvFiles:
		head, tail = os.path.split(graphFileName)
		fileName, fileExtension = os.path.splitext(tail)
		ga = GraphAnalyzer()
		numEdges = ga.readAggregateGraph(graphFileName)
		graphFilePearson[graphFileName] = ga.pearsonAnswersPostNorm()
		print '\n\n' + graphFileName + ", number of edges: " + str(numEdges) + ", number of correlations: " + str(graphFilePearson[graphFileName][1])
		nxGraph = ga.buildNXGraph()
		
		# Stats
		ga.saveNXGraph(fileName, nxGraph)
		ga.computeStats(fileName, nxGraph)

		# Consensus
		ga.plotAllAnswersPearsonWeight(fileName)
		ga.plotDistanceNodes(fileName)
		ga.plotDistanceRatioNodes(fileName)

		ga.plotAvgAnswerAvgDegreeTh(fileName, nxGraph)
		ga.plotStdevAnswerDegreeTh(fileName, nxGraph)

		# Precision
		ga.plotDistanceNodesPreWeight(fileName)
		ga.plotDistanceNodesPostWeight(fileName)
		
		# Improvement
		ga.plotDistancePrePostWeight(fileName)
		ga.plotDistancePrePostDegree(fileName, nxGraph)
		ga.plotDistancePrePostWeightedDegree(fileName, nxGraph)
		ga.plotDistancePrePostPagerank(fileName, nxGraph)
		ga.plotDistancePrePostBetweenness(fileName, nxGraph)
		ga.plotDistancePrePostCloseness(fileName, nxGraph)
		ga.plotDistancePrePostCommunicability(fileName, nxGraph)
		ga.plotDistancePrePostEigenvector(fileName, nxGraph)

		# Accuracy
		ga.plotErrorPreAvgNeighborDegree(fileName, nxGraph)
		ga.plotErrorPostAvgNeighborDegree(fileName, nxGraph)
		ga.plotErrorPreDegree(fileName, nxGraph)
		ga.plotErrorPostDegree(fileName, nxGraph)
		ga.plotErrorPrePostDegree(fileName, nxGraph)
		ga.plotErrorPrePostWeightedDegree(fileName, nxGraph)
		ga.plotErrorPrePostPagerank(fileName, nxGraph)
		ga.plotErrorPrePagerank(fileName, nxGraph)
		ga.plotErrorPostPagerank(fileName, nxGraph)
		ga.plotErrorPreBetweenness(fileName, nxGraph)
		ga.plotErrorPostBetweenness(fileName, nxGraph)
		ga.plotErrorPreCloseness(fileName, nxGraph)
		ga.plotErrorPostCloseness(fileName, nxGraph)
		ga.plotErrorPreCommunicability(fileName, nxGraph)
		ga.plotErrorPostCommunicability(fileName, nxGraph)
		ga.plotErrorPreEigenvector(fileName, nxGraph)
		ga.plotErrorPostEigenvector(fileName, nxGraph)

		ga.computeStatsEdgeAccuracy(fileName, nxGraph)
		ga.plotEdgeAccuracyPre(fileName, nxGraph)
		ga.plotEdgeAccuracyPost(fileName, nxGraph)
		ga.plotEdgeAccuracyPreAvg(fileName, nxGraph)
		ga.plotEdgeAccuracyPostAvg(fileName, nxGraph)
		ga.plotEdgeAccuracyAll(fileName, nxGraph)
		ga.plotNodeAccuracyAll(fileName, nxGraph)
		
		"""
		# OLD CODE ABOUT MODELS
		answers_params = ga.fitModel1Answer(fileName, nxGraph)
		ga.model1AnswerEstimation(fileName, nxGraph, answers_params)
		answers_params = ga.fitModel2Answer(fileName, nxGraph)
		ga.model2AnswerEstimation(fileName, nxGraph, answers_params)
		answers_params = ga.fitModel3Answer(fileName, nxGraph)
		ga.model3AnswerEstimation(fileName, nxGraph, answers_params)
		
		answers_params = ga.fitModel1User(fileName, nxGraph)
		ga.model1UserEstimation(fileName, nxGraph, answers_params)
		
		comb_answers = itertools.combinations([0,1,2,3], 3)
		for training in comb_answers:
			answers_params = ga.fitModel1UserTraining(fileName, nxGraph, training)
			ga.model1UserEstimationTraining(fileName, nxGraph, answers_params, training)
		"""
				
	# Order by Pearson coefficient of Answer-Post
	orderedGraphFilePearson = collections.OrderedDict(sorted(graphFilePearson.items(), key=lambda t: t[1][0], reverse=True))

	outputFile = open(checkDir(EXPERIMENT_PATH + OUTCOME_PATH) + OUTPUT_FILENAME,"w")
	outputFile.write(OUTPUT_HEADER)
	for key in orderedGraphFilePearson.iterkeys():
		outputFile.write(key + ";" + str(pearsonPreAvg) + ";" + str(orderedGraphFilePearson[key][0]) + ";" + str(orderedGraphFilePearson[key][1]) + "\n")


if __name__ == '__main__':
	main()
