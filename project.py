import nltk
import numpy as np
from IPython.display import display
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint
import time

# download the treebank corpus from nltk
# nltk.download('treebank')

# download the universal tagset from nltk
# nltk.download('universal_tagset')

# reading the Treebank tagged database
data = list(nltk.corpus.treebank.tagged_sents(tagset = 'universal'))

# print each word with its respective tags for first 2 sentences
print('words with its respective tags for first 2 sentences in the dataset: ')
for word in data[:2]:
    for tuple in word:
        print(tuple)

# split the data into training and test set in the ratio 80 : 20
train_set, test_set = train_test_split(data, train_size = 0.80, test_size = 0.20, random_state = 101)

# create list of train and test tagged words
train_tagged_words = [tup for word in train_set for tup in word]
test_tagged_words = [tup for word in test_set for tup in word]
print('number of training tagged words: ', len(train_tagged_words))
print('number of testing tagged words: ', len(test_tagged_words))

# check how many unique tags are present in the training dataset
tags = {tag for word,tag in train_tagged_words}
print('number of unique tags: ', len(tags))
print('Tags:', tags)

# check total words in vocabulary
vocab = {word for word,tag in train_tagged_words}

# compute Emission Probability
def emission_prob(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]

#now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)

# compute Transition Probability
def transition_prob(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)

tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)):
        tags_matrix[i, j] = transition_prob(t2, t1)[0]/transition_prob(t2, t1)[1]

# convert the matrix to a df for better readability
print('T x T transition matrix of tags:')
tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
display(tags_df)

# Veterbi Algorithm for POS tagging
def Viterbi(words, train_bag = train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = emission_prob(words[key], tag)[0]/emission_prob(words[key], tag)[1]
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))

# Let's test our Viterbi algorithm on a few sample sentences of test dataset
random.seed(1234)      #define a random seed to get same sentences when run multiple times

# choose random 10 numbers
rndom = [random.randint(1,len(test_set)) for x in range(10)]

# list of 10 sents on which we test the model
test_run = [test_set[i] for i in rndom]

# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]

# list of untagged words
test_tagged_words = [tup[0] for sent in test_run for tup in sent]

#Here We will only test 10 sentences to check the accuracy
#as testing the whole training set takes huge amount of time
print('Testing only 10 sentences: ')
start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end-start

print("Time taken in seconds: ", difference)

# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]

accuracy = len(check)/len(tagged_seq)
print('Viterbi Algorithm Accuracy: ',accuracy*100)

# Code to test all the test sentences
# This takes lot of time
# therefore I have commented this section

# print('Testing all the sentences: ')
# test_tagged_words = [tup for sent in test_set for tup in sent]
# test_untagged_words = [tup[0] for sent in test_set for tup in sent]
# test_untagged_words

# start = time.time()
# tagged_seq = Viterbi(test_untagged_words)
# end = time.time()
# difference = end-start

# print("Time taken in seconds: ", difference)

# accuracy
check = [i for i, j in zip(test_tagged_words, test_untagged_words) if i == j]

accuracy = len(check)/len(tagged_seq)
print('Viterbi Algorithm Accuracy: ',accuracy*100)

#Check how a sentence is tagged by the two POS taggers
#and compare them
test_sent="Ronaldo is the greatest player of all time"
pred_tags_rule=Viterbi_rule_based(test_sent.split())
pred_tags_withoutRules= Viterbi(test_sent.split())
print(pred_tags_rule)
print(pred_tags_withoutRules)
#Ronaldo are tagged as NUM as they are unknown words for Viterbi Algorithm
