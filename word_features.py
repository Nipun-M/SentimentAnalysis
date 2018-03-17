from utility import *
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

data = load_data_set('data.csv')
stopwords = stopwords.words('english')
frequency_of_words = dict()
frequency_per_sentiment = dict()
sentiments = set([datum[0] for datum in data])
for sentiment in sentiments:
    frequency_per_sentiment[sentiment] = dict()


def get_significant_words(data):
    sentiment = data[0]
    sentence = data[1]
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = list(filter(lambda word: word not in stopwords, words))
    update_words_list(sentiment, words)
    return words


def update_words_list(sentiment, words):
    for word in words:
        if word not in frequency_of_words.keys():
            frequency_of_words[word] = 1
        else:
            frequency_of_words[word] += 1

        if word not in frequency_per_sentiment[sentiment].keys():
            frequency_per_sentiment[sentiment][word] = 1
        else:
            frequency_per_sentiment[sentiment][word] += 1


def main():
    features = list()
    for datum in data:
        print(get_significant_words(datum))
    return features


if __name__ == "__main__":
    main()
    import json
    all = open("all.json",'w')
    per_senti = open('per_sentiment.json', 'w')
    all.write(json.dumps(frequency_of_words, indent=4))
    per_senti.write(json.dumps(frequency_per_sentiment, indent=4))
    all.close()
    per_senti.close()
