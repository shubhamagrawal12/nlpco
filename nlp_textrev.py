import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize

short_pos = open("../short_reviews/positive.txt","r").read()
short_neg = open("../short_reviews/negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
	documents.append((r,'pos'))

for r in short_neg.split('\n'):
	documents.append((r,'neg'))
	
all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
	all_words.append(w.lower())

for w in short_neg_words:
	all_words.append(w.lower())
	
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))

word_features = list(all_words.keys())[:5000]

def find_features(document):
#	words = set(document)
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
		
	return features
	

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# posterior - prior occurences x likelihood / evidence

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close() 

print("Original: Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#save_classifier = open("naivebayes.pickle","wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

# Multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive Bayes Classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

## Gaussian Naive Bayes
#GNB_classifier = SklearnClassifier(GaussianNB())
#GNB_classifier.train(training_set)
#print("GNB_classifier accuracy percent: ", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

# Bernoulli Naive Bayes
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("Bernoulli Naive Bayes Classifier accuracy percent: ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

## LogisticRegression, SGDClassifier
## SVC, LinearSVC, NuSVC

# LogisticRegression
LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
print("Logistic Regression Classifier accuracy percent: ", (nltk.classify.accuracy(LR_classifier, testing_set))*100)

# SGDClassifier
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD Classifier accuracy percent: ", (nltk.classify.accuracy(SGD_classifier, testing_set))*100)

# SVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC Classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# LinearSVC
LSVC_classifier = SklearnClassifier(LinearSVC())
LSVC_classifier.train(training_set)
print("Linear SVC Classifier accuracy percent: ", (nltk.classify.accuracy(LSVC_classifier, testing_set))*100)

# NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)
		
	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
			
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf
	
voted_classifier = VoteClassifier(classifier,MNB_classifier,BNB_classifier,LR_classifier,SGD_classifier,LSVC_classifier,NuSVC_classifier)
print("Voted Classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

#print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)



