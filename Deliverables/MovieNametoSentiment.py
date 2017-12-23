import requests
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from sets import Set
import os
import ast
import csv


def movieNametoSentiment():

	reviewsTrainNeg = {}
	reviewsTrainPos = {}
	reviewsTestNeg = {}
	reviewsTestPos = {}
	index = 0

	#GET NAMES OF MOVIE REV FILE (WITHOUT RATING) AND STORE THEM IN A LIST INDEXED BY THE DOC NAME
	for reviewFile in os.listdir('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/train/neg'):
		index = int(reviewFile.split("_")[0])
		reviewsTrainNeg[index] = reviewFile

	for reviewFile in os.listdir('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/train/pos'):
		index = int(reviewFile.split("_")[0])
		reviewsTrainPos[index] = reviewFile

	for reviewFile in os.listdir('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/test/neg'):
		index = int(reviewFile.split("_")[0])
		reviewsTestNeg[index] = reviewFile

	for reviewFile in os.listdir('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/test/pos'):
		index = int(reviewFile.split("_")[0])
		reviewsTestPos[index] = reviewFile

	#OPEN URL FILE AND STORE EACH URL IN A MOVIES LIST
	movie_url = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/train/urls_neg.txt','r')
	movies = movie_url.read().split('\n')
	movie_url.close()
	negTrainURLFile = {}

	for key in reviewsTrainNeg:
		if movies[key] in negTrainURLFile:
			negTrainURLFile[movies[key]].append(reviewsTrainNeg[key])
		else:
			negTrainURLFile[movies[key]] = [reviewsTrainNeg[key]]


	movie_url = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/train/urls_pos.txt','r')
	movies = movie_url.read().split('\n')
	movie_url.close()
	posTrainURLFile = {}

	for key in reviewsTrainPos:
		if movies[key] in posTrainURLFile:
			posTrainURLFile[movies[key]].append(reviewsTrainPos[key])
		else:
			posTrainURLFile[movies[key]] = [reviewsTrainPos[key]]

	movie_url = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/test/urls_pos.txt','r')
	movies = movie_url.read().split('\n')
	movie_url.close()
	posTestURLFile = {}

	for key in reviewsTestPos:
		if movies[key] in posTestURLFile:
			posTestURLFile[movies[key]].append(reviewsTestPos[key])
		else:
			posTestURLFile[movies[key]] = [reviewsTestPos[key]]

	movie_url = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/test/urls_neg.txt','r')
	movies = movie_url.read().split('\n')
	movie_url.close()
	negTestURLFile = {}

	for key in reviewsTestNeg:
		if movies[key] in negTestURLFile:
			negTestURLFile[movies[key]].append(reviewsTestNeg[key])
		else:
			negTestURLFile[movies[key]] = [reviewsTestNeg[key]]

	negTestUrlName = {}
	movie_name = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/Text_file/_urls_neg_test_movie_name_and_url_stop_time.txt','r')
	movies = movie_name.read().split('\n')
	movie_name.close()
	for movie in movies:
		movieUrlName = movie.split(" : ")
		if len(movieUrlName)<2:
			continue
		negTestUrlName[movieUrlName[0]] = movieUrlName[1]
	negTestNameReviewFile = {}
	for key in negTestUrlName:
		if key == '':
			continue
		negTestNameReviewFile[negTestUrlName[key]] = negTestURLFile[key]


	posTestUrlName = {}
	movie_name = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/Text_file/_urls_pos_test_movie_name_and_url_stop_time.txt','r')
	movies = movie_name.read().split('\n')
	movie_name.close()
	for movie in movies:
		movieUrlName = movie.split(" : ")
		if len(movieUrlName)<2:
			continue
		posTestUrlName[movieUrlName[0]] = movieUrlName[1]
	posTestNameReviewFile = {}
	for key in posTestUrlName:
		if key == '':
			continue
		posTestNameReviewFile[posTestUrlName[key]] = posTestURLFile[key]

	posTrainUrlName = {}
	movie_name = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/Text_file/_urls_pos_train_movie_name_and_url_stop_time.txt','r')
	movies = movie_name.read().split('\n')
	movie_name.close()
	for movie in movies:
		movieUrlName = movie.split(" : ")
		if len(movieUrlName)<2:
			continue
		posTrainUrlName[movieUrlName[0]] = movieUrlName[1]
	posTrainNameReviewFile = {}
	for key in posTrainUrlName:
		if key == '':
			continue
		posTrainNameReviewFile[posTrainUrlName[key]] = posTrainURLFile[key]


	negTrainUrlName = {}
	movie_name = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/Text_file/_urls_neg_train_movie_name_and_url_stop_time.txt','r')
	movies = movie_name.read().split('\n')
	movie_name.close()
	for movie in movies:
		movieUrlName = movie.split(" : ")
		if len(movieUrlName)<2:
			continue
		negTrainUrlName[movieUrlName[0]] = movieUrlName[1]
	negTrainNameReviewFile = {}
	for key in negTrainUrlName:
		if key == '':
			continue
		negTrainNameReviewFile[negTrainUrlName[key]] = negTrainURLFile[key]


	trainMovieNameReviewFile = {}

	for key in negTrainNameReviewFile:
		trainMovieNameReviewFile[key] = negTrainNameReviewFile[key]

	for key in posTrainNameReviewFile:
		if key in trainMovieNameReviewFile:
			trainMovieNameReviewFile[key].append(posTrainNameReviewFile[key])
		else:
			trainMovieNameReviewFile[key] = (posTrainNameReviewFile[key])

	testMovieNameReviewFile = {}

	for key in negTestNameReviewFile:
		testMovieNameReviewFile[key] = negTestNameReviewFile[key]

	for key in posTestNameReviewFile:
		if key in testMovieNameReviewFile:
			testMovieNameReviewFile[key].append(posTestNameReviewFile[key])
		else:
			testMovieNameReviewFile[key] = (posTestNameReviewFile[key])


	#OPEN FILES WITH SENTIMENT SCORES AND REVIEWS, STORE THEM IN A DICTIONARY WITH KEY IS THE FILE NAME AND VALUE IS THE SCORE
	trainFilePolScore = {}
	testFilePolScore = {}
	# with open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/movie_reviews_polarity_score.txt','r') as inf:
	# 	dict_from_file = ast.literal_eval(inf.read())
	filePolScore = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/Text_file/Naive_Bayes_Sentiment_Score/Subtraction_method/nb_train_pos_namefile_sentiment_score.txt','r')
	polarityScores = filePolScore.read().split('\n')
	for score in polarityScores:
		fileScore = score.split(" : ")
		if len(fileScore)<2:
			continue
		trainFilePolScore[fileScore[0]] = fileScore[1]
	filePolScore.close()

	filePolScore = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/Text_file/Naive_Bayes_Sentiment_Score/Subtraction_method/nb_train_neg_namefile_sentiment_score.txt','r')
	polarityScores = filePolScore.read().split('\n')
	for score in polarityScores:
		fileScore = score.split(" : ")
		if len(fileScore)<2:
			continue
		trainFilePolScore[fileScore[0]] = fileScore[1]
	filePolScore.close()

	filePolScore = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/Text_file/Naive_Bayes_Sentiment_Score/Subtraction_method/nb_test_pos_namefile_sentiment_score.txt','r')
	polarityScores = filePolScore.read().split('\n')
	for score in polarityScores:
		fileScore = score.split(" : ")
		if len(fileScore)<2:
			continue
		testFilePolScore[fileScore[0]] = fileScore[1]
	filePolScore.close()

	filePolScore = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/Text_file/Naive_Bayes_Sentiment_Score/Subtraction_method/nb_test_neg_namefile_sentiment_score.txt','r')
	polarityScores = filePolScore.read().split('\n')
	for score in polarityScores:
		fileScore = score.split(" : ")
		if len(fileScore)<2:
			continue
		testFilePolScore[fileScore[0]] = fileScore[1]
	filePolScore.close()


	
	trainMovieSentiment = {}
	testMovieSentiment = {}
	for movie in trainMovieNameReviewFile:
		reviewFiles = trainMovieNameReviewFile[movie]
		sentiPolScore = 0
		for file in reviewFiles:
			if isinstance(file, list):
				continue
			if file not in trainFilePolScore:
				continue
			sentiPolScore += float(trainFilePolScore[file])
		sentiPolScore /= len(reviewFiles)
		trainMovieSentiment[movie.strip()] = sentiPolScore

	for movie in testMovieNameReviewFile:
		reviewFiles = testMovieNameReviewFile[movie]
		sentiPolScore = 0
		for file in reviewFiles:
			if isinstance(file, list):
				continue
			if file not in testFilePolScore:
				continue
			sentiPolScore += float(testFilePolScore[file])
		sentiPolScore /= len(reviewFiles)
		testMovieSentiment[movie.strip()] = sentiPolScore

	movieSentiment = {}

	for movie in trainMovieSentiment:
		movieSentiment[movie] = trainMovieSentiment[movie]
	for movie in testMovieSentiment:
		if movie in movieSentiment:
			movieSentiment[movie] = (movieSentiment[movie]+testMovieSentiment[movie])/2
		else:
			movieSentiment[movie] = testMovieSentiment[movie]
	movieData = []
	path = '/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/CSV_file/dataset3.csv'
	dataset = open(path,'rb')
	reader = list(csv.reader(dataset))


	for movieInfo in reader:
		movieName = movieInfo[0]
		#print movieName, trainMovieSentiment[movieName]
		if movieName in movieSentiment:
			movieInfo.append(movieSentiment[movieName])
		movieData.append(movieInfo)

	with open('dataset_nb.csv', 'a') as csv_file:
		writer = csv.writer(csv_file)
		for movie in movieData:
			writer.writerow(movie)
movieNametoSentiment()
