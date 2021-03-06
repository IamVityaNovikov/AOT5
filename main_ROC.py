import csv
import random
from utils import toTokens
from my_NBC import BayesClassifierModel

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SEED = 8
TRAIN_SIZE = 0.8
BETA = 1.0

table = []

with open('spam.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    table= list(spamreader)

table = table[1:]

random.Random(SEED).shuffle(table)

SPAM =1
HAM = 0

x = [row[1] for row in table]
y = [SPAM if row[0] == 'spam' else HAM for row in table]

x = [toTokens(text) for text in x]

x_train = x[:int(len(x)*TRAIN_SIZE)]
x_test = x[int(len(x)*TRAIN_SIZE):]

y_train = y[:int(len(y)*TRAIN_SIZE)]
y_test = y[int(len(y)*TRAIN_SIZE):]

classifier = BayesClassifierModel()
classifier.trainModel(x_train, y_train, SPAM, HAM)


TP, FP, TN, FN = 0, 0, 0, 0


results = []

for i in range(len(x_test)):
	class_result = classifier.classifyData(x_test[i])

	if class_result['class'] == SPAM:
		results.append((class_result['probability'], y_test[i]))
		if y_test[i] == SPAM:
			TP += 1
		else:
			FP += 1
	else:
		results.append((1 - class_result['probability'], y_test[i]))
		if y_test[i] != SPAM:
			TN += 1
		else:
			FN += 1


precision = TP / (TP+FP)
recall = TP / (TP+FN)

#F-measure
F_beta = (1 + BETA**2) * ((precision * recall)/(BETA**2 * precision + recall))

#ROC curve
TP_rate = []
FP_rate = []

#area under curve
AUC_ROC = 0

for i in range(101):

	TP, FP, TN, FN = 0, 0, 0, 0

	threshold = 0.01 * (i)
	for res in results:
		y = int(res[0] > threshold)

		if y and res[1]:
			TP += 1
		elif y and not res[1]:
			FP += 1
		elif not y and not res[1]:
			TN += 1
		elif not y and res[1]:
			FN += 1
	TP_rate.append(TP / (TP+FN))
	FP_rate.append(FP / (FP+TN))

	AUC_ROC += TP_rate[-1] * 0.01


fig, ax = plt.subplots()
ax.plot(FP_rate, TP_rate)
ax.set(xlabel='FP rate', ylabel='TP rate', title='AUC-ROC = ' + str(AUC_ROC) + 
		'\n Precision = ' + str(precision) + 
		' Recall = ' + str (recall) + 
		'\n Fβ = ' + str(F_beta) + ' (β = ' + str(BETA) + ')')
ax.grid()

#fig.savefig("ROC_CURVE.png")
plt.show()