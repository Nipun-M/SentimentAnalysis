# # word features
import string
import numpy as np

from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# define some variables

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwords = sw.words('english')


# function to clean the word

def process_word(word):
    word = lemmatizer.lemmatize(word)
    word = stemmer.stem(word)
    return word


# function to process a sentence given in string format

def clean_sentence(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = list(filter(lambda word: word not in stopwords, words))
    words = list(map(process_word, words))
    return ' '.join(words)


# function to clean the sentence

def clean(row):
    sentence = row['content']
    return clean_sentence(sentence)

# get word features for a dataframe

def get_word_features(vectorizer, dataframe):
    dataframe['clean_content'] = dataframe.apply(clean, axis=1)
    return np.array(vectorizer.fit_transform(dataframe['clean_content']).toarray())

def get_word_features_test(vectorizer, dataframe):
    dataframe['clean_content'] = dataframe.apply(clean, axis=1)
    return np.array(vectorizer.transform(dataframe['clean_content']).toarray())
