import requests
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from sets import Set
import os
import tmdbsimple as tmdb
from time import sleep
import csv


def scrapeFromTMDB(start):
	tmdb.API_KEY = 'f0dfd6aa643d9c600f57991473f2eaf3'
	search = tmdb.Search()

	movieNamePath = '/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/Data/_urls_neg_test_movie_name_and_url_stop_time.txt'
	file = open(movieNamePath,'r')
	lines = file.read()
	lines = lines.split('\n')
	file.close()
	movieNames = []
	for line in lines:
		name = line.split(" : ")[-1]
		if name=='':
			continue
		movieNames.append(name.strip())

	#movieDetails = defaultdict()
	movieDetails = []

	if start>len(movieNames):
		return
	for name in movieNames[start:start+40]:
		response = search.movie(query = name, include_adult = True)
		movie = []
		if len(response['results'])<=0:
			continue
		try:
			if str(response['results'][0]['title']) == str(name):
				movie.append(name)
				movie.append(response['results'][0]['id'])
				movie.append(response['results'][0]['genre_ids'])
				movie.append(response['results'][0]['popularity'])
				movie.append(response['results'][0]['release_date'])
				movie.append(response['results'][0]['adult'])
				movie.append(response['results'][0]['original_language'])
				movieDetails.append(movie)


		except Exception:
			print "continue"
				#movieDetails[name].append(result['genre_ids'])
		'''
		if str(result['title']) == str(name):
			if name not in movieDetails:
				movieDetails[name] = []
			movieDetails[name].append(int(result['id']))
			movieDetails[name].append(result['genre_ids'])

				# movie_id = int(result['id'])
				# movie = tmdb.Movies(movie_id)
				# res = movie.info()
				
				# print movie.title, movie.runtime, movie.budget, movie.revenue, movie.release_date, movie.popularity, movie.adult, movie.genres
				# print "\n"
		'''

	# for movie in movieDetails:
	# 	movie_id = movie[1]

	# 	movieInfo = tmdb.Movies(movie_id)
	# 	res = movieInfo.info()

	# 	print res
	

	with open('dataset2.csv', 'a') as csv_file:
		writer = csv.writer(csv_file)
		for movie in movieDetails:
			writer.writerow(movie)	
		
def scrapeUsingId(start):
	tmdb.API_KEY = 'f0dfd6aa643d9c600f57991473f2eaf3'
	search = tmdb.Search()
	moviesData = []
	path = '/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/dataset2.csv'
	with open(path, 'rb') as csv_file:
		reader = list(csv.reader(csv_file))
		count = 0
		if start>len(reader):
			return
		for movieInfo in reader[start:start+40]:
			movieId = movieInfo[1]
			movie_Info = tmdb.Movies(movieId)
			res = movie_Info.info()
			movieInfo.append(res['vote_count'])
			movieInfo.append(res['vote_average'])
			movieInfo.append(res['runtime'])
			movieInfo.append(res['budget'])
			movieInfo.append(res['revenue'])
			moviesData.append(movieInfo)
			count += 1
	#movieDetails = defaultdict()
	#movieDetails = []

	# if start>len(movieNames):
	# 	return
	# for name in movieNames[start:start+40]:
		
	# 	try:
			


	# 	except Exception:
	# 		print "lalalala"
				#movieDetails[name].append(result['genre_ids'])
		'''
		if str(result['title']) == str(name):
			if name not in movieDetails:
				movieDetails[name] = []
			movieDetails[name].append(int(result['id']))
			movieDetails[name].append(result['genre_ids'])

				# movie_id = int(result['id'])
				# movie = tmdb.Movies(movie_id)
				# res = movie.info()
				
				# print movie.title, movie.runtime, movie.budget, movie.revenue, movie.release_date, movie.popularity, movie.adult, movie.genres
				# print "\n"
		'''

	# for movie in movieDetails:
	# 	movie_id = movie[1]


	

	with open('dataset3.csv', 'a') as csv_file:
		writer = csv.writer(csv_file)
		for movie in moviesData:
			writer.writerow(movie)	

for i in range(160):
	#scrapeFromTMDB(i*40)
	scrapeUsingId(i*40)
	sleep(10)



