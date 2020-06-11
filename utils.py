import re
from nltk.stem import  WordNetLemmatizer


stop_words = ['the', 'a', 'and', 'of', 'to', 'in', 'i', 'that', 'is', 'it', 'you',
              'he', 'for', 'wa', 'with', 's', 'my', 'his', 'on', 'not', 'this', 'be',
              'from', 'me', 'we', 'text', 'at', 'your', 'him', 'they', 'all', 'or',
              'her', 'what', 'can', 'no', 'one', 'will', 'had', 'so', 'an', 'do',
              'ha', 'their', 'if', 'when', 'she', 'there', 'which', 'would', 'were',
              'more', 'n', 'then', 'like', 'who', 'out', 'count', 'our', 'up', 'now',
              'them', 'these', 'some', 'been', 'about', 'could', 'may', 'how', 'd',
              'into', 'such', 'only', 'make', 'an', 'where', 'yes', 'most', 'must',
              'take', 'very', 'just', 'should', 'any', 'was', 'have', 't', 'isn',
              'by', 'but', 'after' , 'two', 'many', 'are', 'use', 'get', 'other',
              'keep', 'also', 'first'
              ]


def toTokens(text):
	lemmatizer = WordNetLemmatizer()

	words = re.findall(r"[a-zA-Z]+", text)
	words = [word.lower() for  word in words]
	words = [lemmatizer.lemmatize(word) for word in words]
	words = list(filter(lambda word: word not in stop_words, words))

	return words
