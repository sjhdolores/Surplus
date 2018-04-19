import numpy as np
import math, random, sys, time, copy

class  CSMax(object):

	def __init__(self, dim=5, vlambda=0.01, biasLambda=0.01):
		
		self.users = []
		self.user_id = {}
		self.items = []
		self.item_id = {}
		# self.quantities = []
		# self.prices = []
		self.QuantityMap = {} 
		self.PriceMap = {}
		self.IMap = {} 

		self.testUsers = []
		self.testuser_id = {}
		self.testItems = []
		self.testitem_id = {}
		# self.testQuantities = [] 
		# self.testPrices = [] 

		self.testQuantityMap = {} #key:(user,item) value:quantity
		self.testPriceMap = {} #key:(user,item), value:totalPrice
		self.testIMap = {} #key:user, value:categories the user bought

		self.numDim = dim
		self.numUsers = 0
		self.numItems = 0
		self.numRows = 0

		self.uMat = None
		self.iMat = None
		self.uBias = [0]*self.numUsers
		self.iBias = [0]*self.numItems
		self.globalBias =0

		self.vlambda = vlambda
		self.biasLambda = biasLambda


	def readTrainingData(self, file):
		start_time = time.time()
		with open(file) as f:
			content = f.readlines()
			f.close()

		i,j,ilen = 0,0,len(content)

		item_set = set()
		while i<ilen:
			while ord(content[i][0])==10:
				data = content[j:i]
				user = data[0].split('\n')[0]

				for record in data[1:]:
					record = record.split(',')[:-1]
					quantity = int(record[0])
					item = ''
					for c in record[1:quantity*(-1)]:
						item += c
					if item not in item_set:
						item_set.add(item)

					prices = record[quantity*(-1):] if quantity>1 else [record[-1]]

					self.QuantityMap[(user,item)] = self.QuantityMap.get((user,item),0) + quantity

					for price in prices:
						self.PriceMap[(user,item)] = self.PriceMap.get((user,item),0) + float(price)

					if user in self.IMap:
						self.IMap[user].append(item)
					else:
						self.IMap[user] = [item]

				j = i + 1
				break
			i += 1

		self.users = self.IMap.keys()
		self.items = list(item_set)
		# self.quantities = self.QuantityMap.values()
		# self.prices = self.PriceMap.values()

		self.numUsers = len(self.users)
		self.numItems = len(self.items)
		self.numRows = len(self.users)

		for i in range(self.numUsers):
			self.user_id[self.users[i]] = i
		for j in range(self.numItems):
			self.item_id[self.items[j]] = j

		print '==============data summary=============='
		print 'number of users: ', self.numUsers
		print 'number of items: ', self.numItems
		print 'number of rows: ', self.numRows

		self.uMat = np.random.rand(self.numDim, self.numUsers)
		self.iMat = np.random.rand(self.numDim, self.numItems)
		self.uBias = np.zeros(self.numUsers)
		self.iBias = np.zeros(self.numItems)
		self.globalBias = 0

		end_time = time.time()
		delta_time = end_time - start_time

		print 'Read training data: Done.'
		print 'Time it takes: ', delta_time, '\n\n'


	def readTestingData(self, file):
		with open(file) as f:
			content = f.readlines()
			f.close()

		i,j,ilen = 0,0,len(content)

		item_set = set()
		while i<ilen:
			while ord(content[i][0])==10:
				data = content[j:i]
				user = data[0].split('\n')[0]

				for record in data[1:]:
					record = record.split(',')[:-1]
					quantity = int(record[0])
					item = ''
					for c in record[1:quantity*(-1)]:
						item += c
					if item not in item_set:
						item_set.add(item)

					prices = record[quantity*(-1):] if quantity>1 else [record[-1]]

					self.testQuantityMap[(user,item)] = self.testQuantityMap.get((user,item),0) + quantity

					for price in prices:
						self.testPriceMap[(user,item)] = self.testPriceMap.get((user,item),0) + float(price)

					if user in self.testIMap:
						self.testIMap[user].append(item)
					else:
						self.testIMap[user] = [item]

				j = i + 1
				break
			i += 1

		self.testUsers = self.testIMap.keys()
		self.testItems = list(item_set)
		# self.testQuantities = self.testQuantityMap.values()
		# self.testPrices = self.testPriceMap.values()

		for i in range(len(self.testUsers)):
			self.testuser_id[self.testUsers[i]] = i
		for j in range(self.testItems):
			self.testitem_id[self.testItems[j]] = j


	def kprDeltaCs(self, a, q, p):
		return a * (math.log(q+1) - math.log(q)) - p

	def lrProb(self, deltaCs):
		try:
			res = 1.0/(1 + math.exp(-deltaCs))
		except Exception:
			res = 0
		return res

	# def lrProbVec(self, deltaCs):
	# 	ones = np.ones(len(deltaCs))
	# 	return ones/(1 + math.exp(-deltaCs))

	def logLrProb(self, x):
		if x<-10:
			return x
		return math.log(1.0/(1+math.exp(-x)))


	def sgd(self, maxEpochs=50):
		M, N, D = self.numUsers, self.numItems, self.numDim

		t = 0
		learningRate = 0

		rowIndices = [0]*self.numRows
		for i in range(self.numRows):
			rowIndices[i] = i

		batchSize = 20
		batchUsers = set()
		batchItems = set()
		uMatGrad = np.zeros((D,M))
		iMatGrad = np.zeros((D,M))
		uBiasGrad = np.zeros(M)
		iBiasGrad = np.zeros(N)
		globalBiasGrad = 0

		start_time = time.time()

		for epoch in range(maxEpochs):
			epoch_start_time = time.time()

			shuffledRowIndices = copy.copy(rowIndices)
			random.shuffle(shuffledRowIndices)
			fVal = 0
			
			for k in range(self.numRows):
				r = shuffledRowIndices[k]
				t += 1
				i = self.user_id[self.users[r]]
				for item in self.IMap[self.users[r]]:
					j = self.item_id[item]
					p = self.PriceMap[(self.users[r],item)]
					q = self.QuantityMap[(self.users[r],item)]

					xi = self.uMat[:,i]
					yj = self.iMat[:,j]
					bi = self.uBias[i]
					bj = self.iBias[j]
					aij = self.userItemUtility(xi, yj, bi, bj, self.globalBias)

					deltaCsQ = self.kprDeltaCs(aij, q, p)
					deltaCsQ1 = self.kprDeltaCs(aij, q+1, p)
					probQ = self.lrProb(deltaCsQ)
					probQ1 = self.lrProb(deltaCsQ1)

					#negative log-likelihood???
					llval = -(self.logLrProb(deltaCsQ) + self.logLrProb(-deltaCsQ1))
					regVal = 0.5*(self.vlambda*np.linalg.norm(xi)**2 + self.vlambda*np.linalg.norm(yj)**2 + self.biasLambda*bi*bi + self.biasLambda*bj*bj + self.biasLambda*self.globalBias*self.globalBias)
					fVal += (llval + regVal)

					#calculate the gradient
					commonTerm = -((1-probQ)*(math.log(q+1)-math.log(q)) - probQ1*(math.log(q+2)-math.log(q+1)))
					#update the user and item latent vector
					deltaXi = yj*commonTerm + xi*self.vlambda
					uMatGrad[:,i] += deltaXi
					deltaYj = xi*commonTerm + yj*self.vlambda
					iMatGrad[:,j] += deltaYj

					#update bias
					uBiasGrad[i] += (commonTerm + self.biasLambda*self.uBias[i])
					iBiasGrad[j] += (commonTerm + self.biasLambda*self.iBias[j])
					globalBiasGrad += (commonTerm + self.biasLambda*self.globalBias)
					batchUsers.add(i)
					batchItems.add(j)

					#update gradient vector
					learningRate = 0.1/(1+epoch*0.1)

					#apply the aggregated gradient
					if (t%batchSize==0 and t>0) or (t%self.numRows==0):
						for m in range(len(batchUsers)):
							tmpUser = batchUsers.pop()
							self.uMat[:,tmpUser] -= (uMatGrad[:,tmpUser]*learningRate)
							uMatGrad[:,tmpUser] = np.zeros(D)
							self.uBias[tmpUser] -= (learningRate*uBiasGrad[tmpUser])
							uBiasGrad[tmpUser] = 0

						for n in range(len(batchItems)):
							tmpItem = batchItems.pop()
							self.iMat[:,tmpItem] -= (iMatGrad[:,tmpItem]*learningRate)
							iMatGrad[:,tmpItem] = np.zeros(D)
							self.iBias[tmpItem] -= (learningRate*iBiasGrad[tmpItem])
							iBiasGrad[tmpItem] = 0

						self.globalBias -= (learningRate*globalBiasGrad)
						globalBiasGrad = 0

			rmse = self.trainRmse()
			# rmse1 = self.testRmse()

			epoch_end_time = time.time()
			print 'epoch: ', epoch, ', function value: ', fVal, ', learningRate: ', learningRate, ', training rmse:', rmse  #, ',test rmse: ', rmse1
			print 'Time it takes: ', (epoch_end_time - epoch_start_time), '\n'

		rmse = self.trainRmse()
		# rmse1 = self.testRmse()
		print 'final training rmse: ', rmse  #, ', test rmse: ', rmse1

		end_time = time.time()
		print 'Time it takes: ', (end_time - start_time), '\n\n'



	def saveModel(self):
		print 'save model parameters'
		np.savetxt('data/CSMax_umat.csv',self.uMat.transpose())
		np.savetxt('data/CSMax_imat.csv',self.iMat.transpose())
		np.savetxt('data/CSMax_ubias.csv',self.uBias)
		np.savetxt('data/CSMax_ibias.csv',self.iBias)
		np.savetxt('data/CSMax_globalbias.csv',[self.globalBias])


	def trainRmse(self):
		rmse = 0
		infCnt = 0
		for k in range(self.numRows):
			i = self.user_id[self.users[k]]
			for item in self.IMap[self.users[k]]:
				j = self.item_id[item]
				p = self.PriceMap[(self.users[k],item)]
				q = self.QuantityMap[(self.users[k],item)]

				xi = self.uMat[:,i]
				yj = self.iMat[:,j]
				aij = self.userItemUtility(xi, yj, self.uBias[i], self.iBias[j], self.globalBias)
				predQ = self.kprPredQuantity(aij, p)
				if predQ==float('inf'):
					infCnt += 1
				else:
					rmse += ((predQ - q)*(predQ - q))

		return math.sqrt(rmse/(self.numRows - infCnt))


	def saveTestPrediction(self):
		file = '/data/CSMax_test_prediction.csv'
		numTestings = len(self.testUsers)
		signMatch = 0
		infCnt = 0
		for k in range(numTestings):
			i = self.testuser_id[self.testUsers[k]]
			for item in self.testIMap[self.testUsers[k]]:
				j = self.testitem_id[item]
				p = self.testPriceMap[(self.testUsers[k],item)]
				q = self.testQuantityMap[(self.testUsers[k],item)]

				xi = self.uMat[:,i]
				yj = self.iMat[:,j]
				aij = self.userItemUtility(xi, yj, self.uBias[i], self.iBias[j], self.globalBias)
				predQ = self.kprPredQuantity(aij, p)
				record = self.testUsers[k] + '\t' + item + '\t' + str(p) + '\t' + str(q) + '\t' + predQ + '\n'
				with open(file,'a') as f:
					f.write(s)
					f.close()

				if predQ!=float('inf'):
					deltaCsQ = self.kprDeltaCs(aij, int(PredQ), p)
					deltaCsQ1 = self.kprDeltaCs(aij, int(predQ)+1, p)
					if deltaCsQ>0 and deltaCsQ1<0:
						signMatch += 1
					else:
						infCnt += 1

		print signMatch, ' out of ', (numTestings-infCnt), ' match'


	def testRmse(self):
		rmse = 0
		numTestings = len(self.testUsers)
		infCnt = 0
		for k in range(numTestings):
			i = self.testuser_id[self.testUsers[k]]
			for item in self.testIMap[self.testUsers[k]]:
				j = self.testitem_id[item]
				p = self.testPriceMap[(self.testUsers[k],item)]
				q = self.testQuantityMap[(self.testUsers[k],item)]

				xi = self.uMat[:,i]
				yj = self.iMat[:,j]
				aij = self.userItemUtility(xi, yj, self.uBias[i], self.iBias[j], self.globalBias)
				predQ = self.kprPredQuantity(aij, p)
				if predQ==float('inf'):
					infCnt += 1
				else:
					rmse += ((predQ-q)*(predQ-q))

		return math.sqrt(rmse/(numTestings-infCnt))


	def kprPredQuantity(self, a, price):
		try:
			res = 1.0/(math.exp(price/a)-1)
		except Exception:
			res = 0
		return res


	def userItemUtility(self, xi, yj, bi, bj, b0):
		return sum(xi*yj)+bi+bj+b0


if __name__=='__main__':
	trainFile = sys.argv[1]
	# testFile = sys.argv[2]
	dim = 20
	vlambda = 0.05
	biasLambda = vlambda*0.2
	maxEpochs = 10
	csm = CSMax(dim, vlambda, biasLambda)
	csm.readTrainingData(trainFile)
	# csm.readTestingData(testFile)
	csm.sgd(maxEpochs)
	print 'sgd parameters: dim - ', dim, ', lambda - ', vlambda, ', bias lambda - ', biasLambda, ', max epochs - ', maxEpochs
	csm.saveModel()
	# csm.saveTestPrediction()
	# return 0




