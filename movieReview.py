import requests
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from sets import Set
import os


PATH_TO_URL = '/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/train/urls_neg.txt'

def mapURLStoReview(filename):

	movie_review = {}

	reviews = []
	index = 0
	for reviewFile in os.listdir('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/train/neg'):
		index = int(reviewFile.split("_")[0])
		reviews.insert(index, reviewFile)

	print reviews
	
	file = open(filename, 'r')
	lines = file.read()
	lines = lines.split('\n')
	file.close()


	allR = {}
	sz = len(reviews)
	for i in range(sz):
		reviewPath = '/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/train/neg/' + reviews[i]
		file = open(reviewPath, 'r')
		rev = file.read()
		allR[i] = rev
		file.close()

	for ct in range(len(allR)):
		if lines[ct] in movie_review:
			movie_review[lines[ct]] = movie_review[lines[ct]] + [allR[ct]]
		else:
			movie_review[lines[ct]] = [allR[ct]]
		
	movie_url = open('/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/train/urls_neg_train_movie_name.txt','r')
	movies = movie_url.read().split('\n')
	movie_url.close()
	movieUrlName = {}
	for movie in movies[:-1]:
		
		details = movie.split(" : ")
		movieUrlName[details[0]] = details[1]
	
	#print movieUrlName
	#print movie_review
	movieNameReview = {}
	for url in movieUrlName:
		url = url.strip()

		movieNameReview[movieUrlName[url].strip()] = movie_review[url]

	f = open("movie_name_review.txt","a")
	for name in movieNameReview:
		entry = name + ":" + str(movieNameReview[name])

		f.write(entry)
mapURLStoReview(PATH_TO_URL)