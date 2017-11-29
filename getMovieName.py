import requests
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from sets import Set

'''- get movies' names of review
- store names in a dataframe
- calculate sentiment score
- store sentiment score in a dataframe
- read in csv file from movies -> compare which movie is already in the list?
- or scrape data from the website'''

movie_sentiment_score = defaultdict(list)
PATH_TO_URL = '/Users/Rishi/Desktop/Study/Fall2017/NLP/Project/Movie-Revenue-Prediction_Sentiment-Analysis/aclImdb/train/urls_neg.txt'

def getMovieName(filename, start, begin):
    i=0
    f= open("urls_neg_train_movie_name.txt","a")

    review_ids = Set()
    for line in open(filename, 'r'):
    	review_id = line.split("/")[4]
        review_ids.add(review_id)
    
    for id in list(review_ids)[start:begin]:
    	line = "http://www.imdb.com/title/" + id + "/reviews"
        i+=1
        #print("loading ", i)
        page = requests.get(line)
        soup = BeautifulSoup(page.content, 'html.parser')
        title = soup.find('title').get_text()
        title_list = title.split('Reviews')
        print title_list[0]
        f.write(' %s\n' % title_list[0].encode("utf-8"))
        # if i==8740:
        #     break
        #movie_sentiment_score['movie'].append(title_list[0])
    f.close()
    print("finish")
 

for i in range(1, 34):
	start = i*80
	getMovieName(PATH_TO_URL, start, start+ 81)
#print(movie_sentiment_score['movie'][:,50])
