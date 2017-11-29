import requests
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict

'''- get movies' names of review
- store names in a dataframe
- calculate sentiment score
- store sentiment score in a dataframe
- read in csv file from movies -> compare which movie is already in the list?
- or scrape data from the website'''

movie_sentiment_score = defaultdict(list)
PATH_TO_URL = 'C:\\Users\\buihu\Documents\\2017_2018 Umass Amherst\\Intro to NLP\\Final project\\Data\\aclImdb_v1\\aclImdb\\train\\urls_neg.txt'

def getMovieName(filename):
    i=0
    f= open("urls_neg_train_movie_name.txt","w+")
    for line in open(filename, 'r'):
        i+=1
        print("loading ", i)
        page = requests.get(line)
        soup = BeautifulSoup(page.content, 'html.parser')
        title = soup.find('title').get_text()
        title_list = title.split('Reviews')
        f.write(' %s\n' % title_list[0])
        if i==8740:
            break
        #movie_sentiment_score['movie'].append(title_list[0])
    f.close()
    print("finish")


getMovieName(PATH_TO_URL)
print(movie_sentiment_score['movie'][:,50])
