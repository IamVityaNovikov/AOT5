from math import log, exp

class BayesClassifierModel:
	def __init__(self):


		self.class1 = None
		self.class2 = None


		self.data = {}
		self.class1_count = 0
		self.class2_count = 0

		self.class1_words = 0
		self.class2_words = 0

		self.lnP_class1 = 0
		self.lnP_class2 = 0

		self.W_class1 = 0
		self.W_class2 = 0 

		self.is_trained = False

	def trainModel(self, x_data, y_data, class1, class2):

		self.__init__()

		self.class1 = class1
		self.class2 = class2

		for i in range(len(x_data)):

			self.class1_count  += int(y_data[i] == class1)
			self.class2_count  += int(y_data[i] == class2)

			for word in x_data[i]:

				if word not in self.data:
					self.data[word] = {class1: 0, class2: 0}

				if y_data[i] == class1:
					self.data[word][class1] += 1
					self.class1_words += 1

				elif y_data[i] == class2:
					self.data[word][class2] += 1
					self.class2_words += 1

		self.lnP_class1 = log(self.class1_count / (self.class1_count + self.class2_count))
		self.lnP_class2 = log(self.class2_count / (self.class1_count + self.class2_count))

		self.W_class1 = len(self.data) + self.class1_words
		self.W_class2 = len(self.data) + self.class2_words

		self.is_trained = True

	def classifyData(self, new_x):

		if not self.is_trained:
			return None

		W_x_class1 = [self.data[word][self.class1] + 1 if word in self.data else 1 for word in new_x]
		W_x_class2 = [self.data[word][self.class2] + 1 if word in self.data else 1 for word in new_x]

		y_class1 = self.lnP_class1 + sum([log(w / self.W_class1) for w in W_x_class1])
		y_class2 = self.lnP_class2 + sum([log(w / self.W_class2) for w in W_x_class2])

		if y_class1 > y_class2:
			result_class = self.class1
		else:
			result_class = self.class2


		not_class_y, yes_class_y = min(y_class1, y_class2), max(y_class1, y_class2)
		yes_class_probability = 1/(1 + exp(not_class_y - yes_class_y))

		return {'class': result_class, 'probability': yes_class_probability}