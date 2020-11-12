import nltk
import numpy as np
from IPython.display import display
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint
import time

# compute Emission Probability
def emission_prob(word, tag, train_bag):
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]

    #now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
    return count_w_given_tag/ count_tag


# compute Transition Probability
def transition_prob(t2, t1, train_bag):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for i in range(len(tags)-1):
        if tags[i]==t1 and tags[i + 1] == t2:
            count_t2_t1 += 1
    return count_t2_t1 / count_t1

# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)

def tag_matrix(train_bag, tags):
    tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for i, t1 in enumerate(list(tags)):
        for j, t2 in enumerate(list(tags)):
            tags_matrix[i, j] = transition_prob(t2, t1, train_bag)

# convert the matrix to a df for better readability
    tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
    return tags_df

# Veterbi Algorithm for POS tagging
def Viterbi(words, train_bag, tags_df):
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
            emission_p = emission_prob(words[key], tag, train_bag)
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))
