import nltk
import numpy as np
from IPython.display import display
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint
import time
from collections import defaultdict

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
    return tags_df, tags_matrix

# Veterbi Algorithm for POS tagging
def Viterbi(words, train_bag, tags_matrix):
    state = []
    tag_list = list(set([pair[1] for pair in train_bag]))
    tags_dict = defaultdict(list)
    for i in range(len(tag_list)):
        tags_dict[tag_list[i]].append(i)

    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = []
        for tag in tag_list:
            if key == 0:
                transition_p = tags_matrix[tags_dict['.'][0]][tags_dict[tag][0]]
            else:
                transition_p = tags_matrix[tags_dict[state[-1]][0]][tags_dict[tag][0]]

            # finding emission and current probabilities
            emission_p = emission_prob(words[key], tag, train_bag)
            curr_probability = emission_p * transition_p
            p.append(curr_probability)

        viterbi_value = max(p)

        state_max = tag_list[p.index(viterbi_value)]
        state.append(state_max)
    return list(zip(words, state))
