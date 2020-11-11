from functions import *

# download the treebank corpus from nltk
# nltk.download('treebank')

# download the universal tagset from nltk
# nltk.download('universal_tagset')

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

tags_matrix = tag_matrix(train_tagged_words, tags)
display(tags_matrix)

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
tagged_seq = Viterbi(test_tagged_words, train_tagged_words, tags_matrix)
end = time.time()
difference = end-start

print("Time taken in seconds: ", difference)

# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]

accuracy = len(check)/len(tagged_seq)
print('Viterbi Algorithm Accuracy: ',accuracy*100)

# Code to test all the test sentences
# This takes lot of time
# print('Testing all the sentences: ')
# test_tagged_words = [tup for sent in test_set for tup in sent]
# test_untagged_words = [tup[0] for sent in test_set for tup in sent]
# test_untagged_words

# start = time.time()
# tagged_seq = Viterbi(test_untagged_words, train_tagged_words, tags_matrix)
# end = time.time()
# difference = end-start

# print("Time taken in seconds: ", difference)

# accuracy
# check = [i for i, j in zip(test_tagged_words, test_untagged_words) if i == j]

# accuracy = len(check)/len(tagged_seq)
# print('Viterbi Algorithm Accuracy: ',accuracy*100)

#Check how a sentence is tagged by the two POS taggers
#and compare them
print('Tagging input sentences based on the trained model:')
while(True):
    test_sent = input('Enter the sentence: ')
    pred_tags = Viterbi(test_sent.split(), train_tagged_words, tags_matrix)
    print(pred_tags)
    i = input('Do you want to continue (y/n): ')
    if i == 'n' or i == 'N':
        break
