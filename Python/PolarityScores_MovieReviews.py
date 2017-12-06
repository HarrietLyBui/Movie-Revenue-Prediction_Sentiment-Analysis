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

iteration = 1
vader_scores = defaultdict(float)
#===========================Scores generated for the positive review files============================
for filename in os.listdir(pos_train):
    file = open(os.path.join(pos_train, filename), 'r')
    review = file.read()
    file.close()
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(review)
    vader_scores[filename] = ss['compound']  # store the compound score into the vader score dictionary
    # for k in sorted(ss):
    #     # print ('{0}: {1}, '.format(k, ss[k]), end='')
    #     #print('{0}: {1}, '.format(k, ss[k]))
    #     if k == 'compound':
    #         vader_scores[filename] = ss[k]     #store the compound score into the vader score dictionary

    if iteration == 5:
        break
    iteration += 1
#===========================Scores generated for the negative review files============================
iteration = 0
files = os.listdir(neg_train)
for filename in files:
    file = open(os.path.join(neg_train, filename), 'r')
    review = file.read()
    file.close()
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(review)
    vader_scores[filename] = ss['compound']  # store the compound score into the vader score dictionary
    # for k in sorted(ss):
    #     # print ('{0}: {1}, '.format(k, ss[k]), end='')
    #     #print('{0}: {1}, '.format(k, ss[k]))
    #     if k == 'compound':
    #         vader_scores[filename] = ss[k]     #store the compound score into the vader score dictionary

    if iteration == 5:
        break
    iteration += 1


print vader_scores
f = open("movie_reviews_polarity_score.txt","w")
f.write(str(vader_scores))
