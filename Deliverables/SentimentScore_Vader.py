from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import os
import numpy as np
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer

TRAIN_DATA_PATH = "C:\CMPSCI585\Project\OurProject\\aclImdb\\train"
TEST_DATA_PATH = "C:\CMPSCI585\Project\OurProject\\aclImdb\\test"

POS_LABEL = 'pos'
NEG_LABEL = 'neg'

pos_train = os.path.join(TRAIN_DATA_PATH, "pos")
neg_train = os.path.join(TRAIN_DATA_PATH, "neg")
pos_test = os.path.join(TEST_DATA_PATH, "pos")
neg_test = os.path.join(TEST_DATA_PATH, "neg")

iteration = 1
vader_scores = defaultdict(float)

def generate_sentment_score(filename_score, dataset):
    iteration = 0
    files = os.listdir(dataset)
    f2 = open(filename_score,"a")
    vader_scores_train_neg = defaultdict(float)
    for filename in files:
        file = open(os.path.join(dataset, filename), 'r')
        review = file.read()
        file.close()
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(review)
        vader_scores_train_neg[filename] = ss['compound']  # store the compound score into the vader score dictionary
        iteration += 1

    for key, value in vader_scores_train_neg.iteritems():
        f2.write(str(key) + " : " + str(value) + "\n" )
    f2.close()

generate_sentment_score("movie_reviews_polarity_score_train_neg.txt", neg_train)
generate_sentment_score("movie_reviews_polarity_score_train_pos.txt", pos_train)
generate_sentment_score("movie_reviews_polarity_score_test_neg.txt", neg_test)
generate_sentment_score("movie_reviews_polarity_score_test_pos.txt", pos_test)
