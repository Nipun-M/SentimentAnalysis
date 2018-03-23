import nltk
import statistics
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
from string import punctuation


def get_preprocess_sentence(sentence):
    words_filtered = filter(lambda x: x not in punctuation, word_tokenize(sentence))
    return " ".join(map(str.lower, words_filtered))


def preprocess(dataframe):
    clean_sentences = list()
    for index, row in dataframe.iterrows():
        clean_sentence = get_preprocess_sentence(row.content)
        clean_sentences.append(clean_sentence)
    dataframe['clean_content'] = clean_sentences


def get_pos_tags_sentence(sentence):
    tokenized_words=word_tokenize(sentence)
    return nltk.pos_tag(tokenized_words)


def tag_pos(dataframe):
    pos_tags = list()
    for index, row in dataframe.iterrows():
        sentence = row['clean_content']
        tags = get_pos_tags_sentence(sentence)
        pos_tags.append(tags)
    dataframe['pos_tag'] = pos_tags


def get_parsed_tag(word, tag):
    if tag.startswith('NN'):
        newtag = 'n'
    elif tag.startswith('JJ'):
        newtag = 'a'
    elif tag.startswith('V'):
        newtag = 'v'
    elif tag.startswith('R'):
        newtag = 'r'
    else:
        newtag = ''
    return (word, newtag)


def parse_tags(dataframe):
    parsed_pos_tags = list()
    for index, row in dataframe.iterrows():
        item = row['pos_tag']
        parse_list=[]
        for word, tag in item:
            parse_list.append(get_parsed_tag(word, tag))
        parsed_pos_tags.append(parse_list)
    dataframe['parsed_pos_tags'] = parsed_pos_tags


def get_score(item):
    sent_pos_score = list()
    sent_neg_score = list()
    sent_obj_score = list()
    for (word,tag) in item:
        if tag=='':
            continue
        synsets=list(swn.senti_synsets(word,tag))
        if len(synsets) == 0:
            continue
        positiveScore=synsets[0].pos_score()
        negativeScore=synsets[0].neg_score()
        objectiveScore=synsets[0].obj_score()
        sent_pos_score.append(positiveScore)
        sent_neg_score.append(negativeScore)
        sent_obj_score.append(objectiveScore)
    return (sent_pos_score, sent_neg_score, sent_obj_score)



def raw_sentiment_score(dataframe):
    sent_positive_score = list()
    sent_negative_score = list()
    sent_objective_score = list()
    for index, row in dataframe.iterrows():
        item = row['parsed_pos_tags']
        (sent_pos_score, sent_neg_score, sent_obj_score) = get_score(item)
        sent_positive_score.append(sent_pos_score)
        sent_negative_score.append(sent_neg_score)
        sent_objective_score.append(sent_obj_score)
    dataframe['pos'] = sent_positive_score
    dataframe['neg'] = sent_negative_score
    dataframe['obj'] = sent_objective_score


def get_mean(data):
    return statistics.mean(data)


def __get_mean_score(dataframe, column):
    result = list()
    for index, row in dataframe.iterrows():
        data = row[column]
        result.append(get_mean(data))
    return np.vstack(np.array(result))


def get_rel_strength(data):
    count = 0.0
    length = len(data)
    for items in data:
        if items != 0:
            count += 1
    sentence_rel = count/length
    return sentence_rel


def __get_rel_strength(dataframe, column):
    result = list()
    for index, row in dataframe.iterrows():
        data = row[column]
        result.append(get_rel_strength(data))
    return np.vstack(np.array(result))


def get_max(data):
    return max(data)


def __get_max_score(dataframe, column):
    result = list()
    for index, row in dataframe.iterrows():
        data = row[column]
        result.append(get_max(data))
    return np.vstack(np.array(result))


def get_stddev(data):
    if len(data) < 2:
        return 0
    return statistics.stdev(data)


def __get_stddev_score(dataframe, column):
    result = list()
    for index, row in dataframe.iterrows():
        data = row[column]
        result.append(get_stddev(data))
    return np.vstack(np.array(result))


def get_median(data):
    return statistics.median(data)


def __get_median_score(dataframe, column):
    result = list()
    for index, row in dataframe.iterrows():
        data = row[column]
        result.append(get_median(data))
    return np.vstack(np.array(result))


def get_sentiment_features(dataframe):

    preprocess(dataframe)
    tag_pos(dataframe)
    parse_tags(dataframe)
    raw_sentiment_score(dataframe)

    features = __get_mean_score(dataframe, 'pos')
    features = np.hstack((features, __get_mean_score(dataframe, 'neg')))
    features = np.hstack((features, __get_mean_score(dataframe, 'obj')))

    features = np.hstack((features, __get_rel_strength(dataframe, 'pos')))
    features = np.hstack((features, __get_rel_strength(dataframe, 'neg')))
    features = np.hstack((features, __get_rel_strength(dataframe, 'obj')))

    features = np.hstack((features, __get_max_score(dataframe, 'pos')))
    features = np.hstack((features, __get_max_score(dataframe, 'neg')))
    features = np.hstack((features, __get_max_score(dataframe, 'obj')))

    features = np.hstack((features, __get_stddev_score(dataframe, 'pos')))
    features = np.hstack((features, __get_stddev_score(dataframe, 'neg')))
    features = np.hstack((features, __get_stddev_score(dataframe, 'obj')))

    features = np.hstack((features, __get_median_score(dataframe, 'pos')))
    features = np.hstack((features, __get_median_score(dataframe, 'neg')))
    features = np.hstack((features, __get_median_score(dataframe, 'obj')))

    return features