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
#files = os.listdir(pos_train)
# f1 = open("movie_reviews_polarity_score_train_pos.txt","a")
# #===========================Scores generated for the positive review files============================
# for filename in files:
#     file = open(os.path.join(pos_train, filename), 'r')
#     review = file.read()
#     file.close()
#     sid = SentimentIntensityAnalyzer()
#     ss = sid.polarity_scores(review)
#     vader_scores[filename] = ss['compound']  # store the compound score into the vader score dictionary
#
# for key, value in vader_scores.iteritems():
#     f1.write(str(key) + " : " + str(value) + "\n" )
# f1.close()
#===========================Scores generated for the negative review files============================
iteration = 0
files = os.listdir(neg_train)
f2 = open("movie_reviews_polarity_score_train_neg.txt","a")
vader_scores_train_neg = defaultdict(float)
for filename in files:
    file = open(os.path.join(neg_train, filename), 'r')
    review = file.read()
    file.close()
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(review)
    vader_scores_train_neg[filename] = ss['compound']  # store the compound score into the vader score dictionary
    iteration += 1

for key, value in vader_scores_train_neg.iteritems():
    f2.write(str(key) + " : " + str(value) + "\n" )
f2.close()
# f = open("movie_reviews_polarity_score.txt","w")
# f.write(str(vader_scores))

# pos_test = os.path.join(TEST_DATA_PATH, "pos")
# neg_test = os.path.join(TEST_DATA_PATH, "neg")
#
# iteration = 1
# vader_scores = defaultdict(float)
# files = os.listdir(pos_test)
# f = open("movie_reviews_polarity_score_test_pos.txt","a")
# #===========================Scores generated for the positive review files============================
# for filename in files:
#     file = open(os.path.join(pos_test, filename), 'r')
#     review = file.read()
#     file.close()
#     sid = SentimentIntensityAnalyzer()
#     ss = sid.polarity_scores(review)
#     vader_scores[filename] = ss['compound']  # store the compound score into the vader score dictionary
#     #f.writerow([filename] + vader_scores[filename])
#     #f.write(str(filename)+ ":" + str(vader_scores))
#
#     # if iteration == 5:
#     #     break
#     iteration += 1
# for key, value in vader_scores.iteritems():
#     f.write(str(key) + " : " + str(value) + "\n")
# f.close()
# #===========================Scores generated for the negative review files============================
# iteration = 0
# files = os.listdir(neg_test)
# vader_scores_neg = defaultdict(float)
# #f = open("movie_reviews_polarity_score.txt","w")
# f_neg = open("movie_reviews_polarity_score_test_neg.txt","a")
# for filename in files:
#     file = open(os.path.join(neg_test, filename), 'r')
#     review = file.read()
#     file.close()
#     sid = SentimentIntensityAnalyzer()
#     ss = sid.polarity_scores(review)
#     vader_scores_neg[filename] = ss['compound']  # store the compound score into the vader score dictionary
#     iteration += 1
#     # if iteration == 5:
#     #     break
#
#     #f.writerow([filename] + vader_scores[filename])
#     #f.write(str(filename) + ":" + str(vader_scores))
# for key, value in vader_scores_neg.iteritems():
#     f_neg.write(str(key) + " : " + str(value) + "\n" )
#
# f_neg.close()
# f.write(str(vader_scores))
#
# writer.writerow([key] + value)
