from __future__ import division

import matplotlib.pyplot as plt
import math
import os
import time

from collections import defaultdict


# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'


###### DO NOT MODIFY THIS FUNCTION #####
def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)
###### END FUNCTION #####


def n_word_types(word_counts):
    '''
    return a count of all word types in the corpus
    using information from word_counts
    '''
    word_types_count = 0

    if word_counts != {}:

        for key in word_counts.values():
             word_types_count += 1
    else:
        print('dic word_counts is empty')

    return word_types_count


def n_word_tokens(word_counts):
    '''
    return a count of all word tokens in the corpus
    using information from word_counts
    '''
    tokens_count = 0

    if word_counts != {}:

        for value in word_counts.values():
             tokens_count += value
    else:
        print('dic word_counts is empty')

    return tokens_count



class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self, path_to_data, tokenizer):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()
        self.path_to_data = path_to_data
        self.tokenize_doc = tokenizer
        self.train_dir = os.path.join(path_to_data, "train")
        self.test_dir = os.path.join(path_to_data, "test")
        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }

    def train_model(self):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.
        """

        pos_path = os.path.join(self.train_dir, POS_LABEL)
        neg_path = os.path.join(self.train_dir, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            for f in os.listdir(p):
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print "REPORTING CORPUS STATISTICS"
        print "NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL]
        print "NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL]
        print "NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL]
        print "NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL]
        print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)

    def update_model(self, bow, label):
        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each label (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each label (self.class_total_doc_counts)
        """
        #update times each word was seen


        for (key, value) in bow.items():

            self.class_word_counts[label][key] += value

            self.class_total_word_counts[label] += value

        #update total tokens in the corresponding label


        for values in bow.keys():
            self.vocab.add(values)

        self.class_total_doc_counts[label] += 1

        pass

    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = self.tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """
        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.class_word_counts[label].items(), key=lambda (w,c): -c)[:n]

    def p_word_given_label(self, word, label):
        """

        Returns the probability of word given label
        according to this NB model.

        """
        #calculate frequency of the word in the corresponding label
        count_word_in_label = self.class_word_counts[label][word]

        #calculate the total tokens in the label
        tokens_in_label = self.class_total_word_counts[label]

        #probability of word given label
        prob_word = count_word_in_label/tokens_in_label

        return prob_word

    def p_word_given_label_and_pseudocount(self, word, label, alpha):
        """
        Returns the probability of word given label wrt psuedo counts.
        alpha - pseudocount parameter
        """
        #calculate frequency of the word in the corresponding label
        count_word_in_label = self.class_word_counts[label][word] + alpha

        #calculate all the tokens in the label
        tokens_in_label = float(self.class_total_word_counts[label] + alpha*len(self.vocab))

        prob_word = count_word_in_label/tokens_in_label
        return prob_word

    def log_likelihood(self, bow, label, alpha):

        """
        Computes the log likelihood of a set of words give a label and pseudocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; pseudocount parameter
        """
        first =  bow.keys()[0]
        likelihood = math.log(self.p_word_given_label_and_pseudocount(first, label, alpha))

        for key in bow.keys():
            if key != first:
                log_word_prop = math.log(self.p_word_given_label_and_pseudocount( key, label, alpha))
                likelihood +=  log_word_prop
        return  likelihood

    def log_prior(self, label):
        """
        Returns the log prior of a document having the class 'label'.
        """

        numdoc_in_label = self.class_total_doc_counts[label]

        if label == POS_LABEL:
            total_doc = numdoc_in_label + self.class_total_doc_counts[NEG_LABEL]
        else:
            total_doc = numdoc_in_label + self.class_total_doc_counts[POS_LABEL]

        lo_prior = math.log(numdoc_in_label/ total_doc)

        return lo_prior

    def unnormalized_log_posterior(self, bow, label, alpha):

        """
        Computes the unnormalized log posterior (of doc being of class 'label').
        bow - a bag of words (i.e., a tokenized document)
                """
        log_pos = self.log_prior( label) + self.log_likelihood( bow, label, alpha)

        return log_pos

    def calculate_log_normalizer(self, bow,alpha):

        log_posterior_pos = self.unnormalized_log_posterior( bow, POS_LABEL, alpha)
        log_psterior_neg = self.unnormalized_log_posterior( bow, NEG_LABEL, alpha)

        log_normalizer = log_posterior_pos * log_psterior_neg

        return log_normalizer

    def calculate_Naives_Bayes_prob(self, bow, label, alpha):
        pos_label_prob = self.unnormalized_log_posterior( bow, label, alpha)
        naive_base = pos_label_prob/ self.calculate_log_normalizer(bow,alpha)
        return naive_base

    def calculate_sentiment_score(self, bow,alpha):
        prob_doc_in_pos = self.calculate_Naives_Bayes_prob( bow, POS_LABEL, alpha)
        prob_doc_in_neg = self.calculate_Naives_Bayes_prob( bow, NEG_LABEL, alpha)
        print('pos prob', prob_doc_in_pos)
        print('neg prob', prob_doc_in_neg)
        sentiment_score = prob_doc_in_pos - prob_doc_in_neg
        # if prob_doc_in_pos >= prob_doc_in_neg:
        #     sentiment_score = prob_doc_in_pos
        # else:
        #     sentiment_score = -prob_doc_in_neg
        # return sentiment_score

    def get_sentiment_score_of_data_set(self, bow,alpha, folder, _label, filename):
        pos_path = os.path.join(folder, _label)
        f_label = open(filename ,"w")
        list_score = {}
        correct=0.0
        total = 0.0

        for f in os.listdir(pos_path):
            with open(os.path.join(pos_path,f),'r') as doc:
                content = doc.read()
                bow = self.tokenize_doc(content)
                classify_result =  self.classify(bow, 1.0)
                sentiment_score = self.calculate_sentiment_score(bow, 1.0)
                if sentiment_score>0 and _label==POS_LABEL :
                    correct+=1
                    print('pos correct', correct)
                if sentiment_score<0 and _label == NEG_LABEL:
                    correct+=1
                    print('neg correct', correct)

                total +=1
                print("Sentiment score", sentiment_score)
                print("Classified as:", classify_result) #Print results of classifier
                print("Expected results:", label) #Print expected result
                list_score[f] = sentiment_score
        print('total',total)
        accuracy = 100* correct/total
        print('accuracy', 100* correct/total)
        for key, value in list_score.iteritems():
            entry = str(key) + " : " + str(value) + "\n"
            f_label.write(entry)
        f_label.close()

        return accuracy



    def classify(self, bow, alpha):
        """

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior)
        bow - a bag of words (i.e., a tokenized document)
        """

        pos_label_prob = self.unnormalized_log_posterior( bow, POS_LABEL, alpha)
        neg_label_prob = self.unnormalized_log_posterior( bow, NEG_LABEL, alpha)

        if pos_label_prob >  neg_label_prob:
            return POS_LABEL
        else:
            return NEG_LABEL


    def likelihood_ratio(self, word, alpha):
        """
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        ratio_pos = self.p_word_given_label_and_pseudocount(word, POS_LABEL, alpha)
        ratio_neg = self.p_word_given_label_and_pseudocount(word, NEG_LABEL, alpha)

        ratio =  ratio_pos/ratio_neg

        return ratio

    def evaluate_classifier_accuracy(self, alpha):
        """
        DO NOT MODIFY THIS FUNCTION

        alpha - pseudocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        correct = 0.0
        total = 0.0

        pos_path = os.path.join(self.test_dir, POS_LABEL)
        neg_path = os.path.join(self.test_dir, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            for f in os.listdir(p):
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    bow = tokenize_doc(content)
                    if self.classify(bow, alpha) == label:
                        correct += 1.0
                    total += 1.0
        return 100 * correct / total

# download the IMDB large movie review corpus from the class webpage to a file location on your computer

# set this variable to point to the location of the IMDB corpus on your computer
PATH_TO_DATA = 'C:\\Users\\buihu\\Documents\\2017_2018 Umass Amherst\\Intro to NLP\\Final project\\Data\\aclImdb_v1\\aclImdb\\'
POS_LABEL = 'pos'
NEG_LABEL = 'neg'
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")

for label in [POS_LABEL, NEG_LABEL]:
    if len(os.listdir(TRAIN_DIR + "\\" + label + "\\")) == 12500:
        print "Great! You have 12500 {} reviews in {}".format(label, TRAIN_DIR + "\\" + label)
    else:
        print "Oh no! Something is wrong. Check your code which loads the reviews"

# We have provided a tokenize_doc function in hw_1.py. Here is a short demo of how it works

d1 = "This SAMPLE doc has   words tHat  repeat repeat"
bow = tokenize_doc(d1)

assert bow['this'] == 1
assert bow['sample'] == 1
assert bow['doc'] == 1
assert bow['has'] == 1
assert bow['words'] == 1
assert bow['that'] == 1
assert bow['repeat'] == 2

bow2 = tokenize_doc("Computer science is both practical and abstract.")
for b in bow2:
    print b

nb = NaiveBayes('C:\\Users\\buihu\\Documents\\2017_2018 Umass Amherst\\Intro to NLP\\Final project\\Data\\aclImdb_v1\\aclImdb\\',
                tokenizer=tokenize_doc)
nb.train_model()

if len(nb.vocab) == 252165:
    print "Great! The vocabulary size is {}".format(252165)
else:
    print "Oh no! Something seems off. Double check your code before continuing. Maybe a mistake in update_model?"

accuracy_test_pos = nb.get_sentiment_score_of_data_set(bow, 1.0,TEST_DIR, POS_LABEL, "nb_test_pos_sentiment_score_nosub.txt")
accuracy_test_neg = nb.get_sentiment_score_of_data_set(bow, 1.0,TEST_DIR, NEG_LABEL, "nb_test_neg_sentiment_score_nosub.txt")
accuracy_train_pos = nb.get_sentiment_score_of_data_set(bow, 1.0,TRAIN_DIR, POS_LABEL, "nb_train_pos_sentiment_score_nosub.txt")
accuracy_train_neg = nb.get_sentiment_score_of_data_set(bow, 1.0,TRAIN_DIR, NEG_LABEL, "nb_train_neg_sentiment_score_nosub.txt")
print('accuracy_test_pos', accuracy_test_pos)
print('accuracy_test_neg', accuracy_test_neg)
print('accuracy_train_pos', accuracy_train_pos)
print('accuracy_train_neg', accuracy_train_neg)
print('accuracy', nb.evaluate_classifier_accuracy(1.0))
