import nltk
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
data = pd.read_csv('train_data.csv')
data=data.astype(str)
data=data[['sentiment','content']]
dataTrain,dataTest=train_test_split(data,test_size=0.1)
# lines 7 to  14 are meant for reading the training dataset and testing dataset and stored in dataTain and dataTest respectively.
#dataTrain = pd.read_csv('train_data.csv')
#dataTrain=dataTrain.astype(str)
#dataTrain=dataTrain[['sentiment','content']]
#print(dataTrain)
#data.to_csv('out.csv')
dataTestEval=pd.read_csv('test_data.csv')
dataTestEval=dataTest.astype(str)
dataTestEval=dataTestEval[['content']]


stop_words=set(stopwords.words("english"))

testWords=[]
for index,row in dataTest.iterrows():
    words_filtered=[e for e in word_tokenize(row.content)]
    words_cleaned=[word for word in words_filtered if 'http' not in word and not word.startswith('@') and not word.startswith('#') and not word.startswith("&") and word !='RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stop_words]
    lemmatizedWords=[stemmer.stem(word) for word in words_without_stopwords]
    for w in lemmatizedWords:
        testWords.append(w)
    



train_empty = dataTrain[dataTrain['sentiment']=='empty']
train_empty=train_empty['content']

train_sadness = dataTrain[dataTrain['sentiment']=='sadness']
train_sadness=train_sadness['content']

train_enthusiasm = dataTrain[dataTrain['sentiment']=='enthusiasm']
train_enthusiasm=train_enthusiasm['content']

train_empty = dataTrain[dataTrain['sentiment']=='surprise']
train_empty=train_empty['content']

train_neutral = dataTrain[dataTrain['sentiment']=='neutral']
train_neutral=train_neutral['content']

train_worry = dataTrain[dataTrain['sentiment']=='worry']
train_worry=train_worry['content']

train_love = dataTrain[dataTrain['sentiment']=='love']
train_love=train_love['content']

train_fun = dataTrain[dataTrain['sentiment']=='fun']
train_fun=train_fun['content']

train_hate = dataTrain[dataTrain['sentiment']=='hate']
train_hate=train_hate['content']

train_relief = dataTrain[dataTrain['sentiment']=='relief']
train_relief=train_relief['content']

train_surprise= dataTrain[dataTrain['sentiment']=='surprise']
train_surprise=train_surprise['content']  

train_happiness= dataTrain[dataTrain['sentiment']=='happiness']
train_happiness=train_happiness['content']  

train_boredom= dataTrain[dataTrain['sentiment']=='boredom']
train_boredom=train_boredom['content']  




test_empty=dataTest[dataTest['sentiment']=='empty']
test_empty=test_empty['content']

test_sadness=dataTest[dataTest['sentiment']=='sadness']
test_sadness=test_sadness['content']

test_enthusiasm=dataTest[dataTest['sentiment']=='enthusiasm']
test_enthusiasm=test_enthusiasm['content']

test_neutral=dataTest[dataTest['sentiment']=='neutral']
test_neutral=test_neutral['content']

test_worry=dataTest[dataTest['sentiment']=='worry']
test_worry=test_worry['content']

test_love=dataTest[dataTest['sentiment']=='love']
test_love=test_love['content']

test_fun=dataTest[dataTest['sentiment']=='fun']
test_fun=test_fun['content']

test_hate=dataTest[dataTest['sentiment']=='hate']
test_hate=test_hate['content']

test_relief=dataTest[dataTest['sentiment']=='relief']
test_relief=test_relief['content']

test_surprise=dataTest[dataTest['sentiment']=='surprise']
test_surprise=test_surprise['content']

test_happiness=dataTest[dataTest['sentiment']=='happiness']
test_happiness=test_happiness['content']

test_boredom=dataTest[dataTest['sentiment']=='boredom']
test_boredom=test_boredom['content']



#print(train_empty)

wordMapList=[] # this is a list containing a list of list of words along with the corresponding sentiments . this is used for computing word features for every sentiment for data analysis purposes.
# wordMapList=[
# [['word1','word2',' '], 'sentiment1'] , [ ['word1', 'word2',' '], 'sentiment2'] ....]
# ]
def wordsExtract(data,sentiment):
    initializedList=[]
    words = ' '.join(data)
    words_filtered=[e for e in word_tokenize(words)]
    words_cleaned=[word for word in words_filtered if 'http' not in word and not word.startswith('@') and not word.startswith('#') and not word.startswith("&") and word !='RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stop_words]
    lemmatizedWords=[stemmer.stem(word) for word in words_without_stopwords]
    initializedList.append(lemmatizedWords)
    initializedList.append(sentiment)
    wordMapList.append(initializedList)


wordsExtract(train_empty,'empty')
wordsExtract(train_sadness,'sadness')
wordsExtract(train_enthusiasm,'enthusiasm')
wordsExtract(train_surprise,'surprise')
wordsExtract(train_neutral,'neutral')
wordsExtract(train_worry,'worry')
wordsExtract(train_love,'love')
wordsExtract(train_fun,'fun')
wordsExtract(train_hate,'hate')
wordsExtract(train_relief,'relief')
wordsExtract(train_surprise,'surprise')
wordsExtract(train_happiness,'happiness')
wordsExtract(train_boredom,'boredom')

#print(wordMapList)




result=[] #this list contains cleaned sentences along with their sentiments (list of tuples)
resultList=[] #this is just going to contain cleaned sentences (list of strings)
lemmas=[] #this list contains list of words that can be used to compute Frequency distribution
for index,row in dataTrain.iterrows():
    words_filtered=[e for e in word_tokenize(row.content)]
    words_cleaned=[word for word in words_filtered if 'http' not in word and not word.startswith('@') and not word.startswith('#') and not word.startswith("&") and word !='RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stop_words]
    lemmatizedWords=[stemmer.stem(word) for word in words_without_stopwords]
    cleanedSentence=" ".join(lemmatizedWords)
    resultList.append(cleanedSentence)
    result.append((cleanedSentence, row.sentiment))
    for w in lemmatizedWords:
        lemmas.append(w)

#print(resultList)
#print(result)
#print(lemmas)

all_words=nltk.FreqDist(lemmas)
#print(all_words.most_common(15))
word_features=list(all_words.keys())[:300] # this list contains top 3000 most common words
#print(word_features)
def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w] = (w in words)

    return features


#print(find_features(lemmas))

featuresets = [(find_features(map[0]), map[1]) for map in wordMapList]
#print(featuresets[:1])
#print(len(featuresets))
trainingSet=featuresets
classifier=nltk.NaiveBayesClassifier.train(trainingSet)
#classifier.show_most_informative_features(15)


empty_cnt = 0
sadness_cnt = 0
enthusiasm_cnt=0
neutral_cnt = 0
worry_cnt = 0
love_cnt = 0
fun_cnt = 0
hate_cnt = 0
relief_cnt = 0
surprise_cnt = 0
happiness_cnt = 0
boredom_cnt = 0
for obj in test_empty: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'empty'): 
        empty_cnt = empty_cnt + 1
for obj in test_sadness: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'sadness'): 
        sadness_cnt = sadness_cnt + 1

for obj in test_enthusiasm: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'enthusiasm'): 
        enthusiasm_cnt = enthusiasm_cnt + 1

for obj in test_neutral: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'neutral'): 
        neutral_cnt = neutral_cnt + 1

for obj in test_worry: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'worry'): 
        worry_cnt = worry_cnt + 1

for obj in test_love: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'love'): 
        love_cnt = love_cnt + 1

for obj in test_fun: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'fun'): 
        fun_cnt = fun_cnt + 1

for obj in test_hate: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'hate'): 
        hate_cnt = hate_cnt + 1

for obj in test_relief: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'relief'): 
        relief_cnt = relief_cnt + 1

for obj in test_surprise: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'surprise'): 
        surprise_cnt = surprise_cnt + 1

for obj in test_happiness: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'happiness'): 
        hapiness_cnt = happiness_cnt + 1

for obj in test_boredom: 
    res =  classifier.classify(find_features(word_tokenize(obj)))
    if(res == 'boredom'): 
        boredom_cnt = boredom_cnt + 1


        
print('[Empty]: %s/%s '  % (len(test_empty),empty_cnt))        
print('[Sadness]: %s/%s '  % (len(test_sadness),sadness_cnt)) 
print('[Enthusiasm]: %s/%s '  % (len(test_enthusiasm),enthusiasm_cnt)) 
print('[Neutral]: %s/%s '  % (len(test_neutral),neutral_cnt)) 
print('[Worry]: %s/%s '  % (len(test_worry),worry_cnt)) 
print('[Love]: %s/%s '  % (len(test_love),love_cnt)) 
print('[Fun]: %s/%s '  % (len(test_fun),fun_cnt)) 
print('[Hate]: %s/%s '  % (len(test_hate),hate_cnt)) 
print('[Relief]: %s/%s '  % (len(test_relief),relief_cnt)) 
print('[Happiness]: %s/%s '  % (len(test_happiness),happiness_cnt)) 
print('[Surprise]: %s/%s '  % (len(test_surprise),surprise_cnt)) 
print('[Boredom]: %s/%s '  % (len(test_boredom),boredom_cnt)) 