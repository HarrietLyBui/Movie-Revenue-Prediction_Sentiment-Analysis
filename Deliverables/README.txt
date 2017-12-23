* Name:
Lopamudra Pal: lpal@cs.umass.edu
Ly (Harriet) Bui: bui23L@mtholyoke.edu
Rishi Mody: rmody@cs.umass.edu

* CS585: Final Project
* Title: Predicting Box Office Success:
Do Critical Reviews Really Matter?

- SentimentScore_NaiveBayes.py: 
	+ Tokenize a document, process entire training set and update the model.
	+ Calculate log likelihood, log prior, unormalized log posterior, log normalizer
	+ Calculate probability for a document to be positive and negative
	+ Calculate sentiment score for a document and write them to *txt file
	+ Evaluate the accuracy of the classification model and sentiment score generation model
- SentimentScore_Vader.py:
	+ Calculate sentiment score from Vader for four dataset positive train, negative train, positive test, negative test

- MovieNametoSentiment.py: 
	+ average sentiment score of multiple reviews for the same movies
	+ match the sentiment score with the correct movie name
	+ write the the score to csv file

- RevenuePrediction_Regression.py:
	+ Predict Revenue for Regression 

- RevenuePrediction_Classification.py:
	+ Predict Revenue for Classification

*How to run the file:
- Install Anaconda and Python 2.7
- In command line: python filename





