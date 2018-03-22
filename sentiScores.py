import nltk
import csv
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import sentiwordnet as swn
import statistics

stop_words=stopwords.words("english")
stemmer=PorterStemmer()

def get_dataframe(filename):
    raw_data = dict()
    reader = csv.reader(open('data.csv'))
    headers = list(reader.__next__())
    for header in headers:
        raw_data[header] = list()
    for row in reader:
        raw_data['sentiment'].append(row[0])
        raw_data['content'].append(','.join(row[1:]))
    df = pd.DataFrame(raw_data)
    return df



mydataFrame = get_dataframe('data.csv')

#print(mydataFrame[['content','sentiment']])


result=[] #this list contains cleaned sentences along with their sentiments (list of tuples)
resultList=[] #this is just going to contain cleaned sentences (list of strings)
lemmas=[] #this list contains list of words that can be used to compute Frequency distribution
taggedList=[]
def preprocess(data):
    for index,row in data.iterrows():
        words_filtered=[e for e in word_tokenize(row.content)]
        words_cleaned=[word for word in words_filtered if 'http' not in word and not word.startswith('@') and not word.startswith('#') and not word.startswith("&") and word !='RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stop_words]
        lemmatizedWords=[stemmer.stem(word) for word in words_without_stopwords]
        #cleanedSentence=" ".join(lemmatizedWords)
        cleanedSentence=" ".join(words_filtered)
        resultList = list()
        resultList.append(cleanedSentence)
        #resultList.append(cleanedSentence)
        result.append((cleanedSentence, row.sentiment))
        for w in lemmatizedWords:
            lemmas.append(w)
        #print(words_filtered)


preprocess(mydataFrame)
#print(resultList)
#print(result)
#print(lemmas)


sent_POS_items=[] # this stores list of items of the form (word, pos_tag)  [[(),()],[(),()],......] Each nested list represents 



def findPOS():
    for sentences in resultList:
        tokenizedWords=word_tokenize(sentences)
        mytagged=nltk.pos_tag(tokenizedWords)
        taggedList.append(mytagged)

    for item in taggedList:
        tempList=[]
        for (word,tag) in item:
            if tag.startswith('NN'):
                newtag='n'
            elif tag.startswith('JJ'):
                newtag='a'
            elif tag.startswith('V'):
                newtag='v'
            elif tag.startswith('R'):
                newtag='r'
            else:
                newtag=''
            tempList.append((word,newtag))
        sent_POS_items.append(tempList)


findPOS()
#print(taggedList[0][0])




'''
breakdown = swn.senti_synset('breakdown.n.03')
print(breakdown)
a=list(swn.senti_synsets('slow'))
print(a)

happy = swn.senti_synsets('happy', 'a')
print(happy)
print(synsets_scores['peaceful.a.01']['pos'])
'''
sent_positive_score=[] # this stores positive score of each item of the form (word, pos_tag)
sent_negative_score=[] # this stores negative score of each item of the form (word, pos_tag)
sent_objective_score=[] # this stores objective score of each item of the word (word, pos_tag)
positive_mean=[] # this stores mean of the positive score for each sentence
negative_mean=[] # this stores mean of the negative score for each sentence
objective_mean=[] # this stores mean of the objective score for each sentence
positive_relative=[] # this stores relative strength of the postive score for each sentence
negative_relative=[] # this stores relative strength of the negative score for each sentence
objective_relative=[] # this stores relative strength of the objective score for each sentence
positive_stdev=[] # this stores the standard deviation for the positive scores
negative_stdev=[] # this stores the standard deviation for the negative scores
objective_stdev=[] # this stores the standard deviation for the objective scores
positive_maxScore=[]
negative_maxScore=[]
objective_maxScore=[]
positive_minScore=[]
negative_minScore=[]
objective_minScore=[]
positive_medianScore=[]
negative_medianScore=[]
objective_medianScore=[]


def findScores():
    for item in sent_POS_items:
        sent_pos_score=[]
        sent_neg_score=[]
        sent_obj_score=[]
        for (word,tag) in item:
            if tag!='':
            #str=word+"."+tag+"."+"01"
                synsets=list(swn.senti_synsets(word,tag))
                if(len(synsets)>0):
                    positiveScore=synsets[0].pos_score()
                    negativeScore=synsets[0].neg_score()
                    objectiveScore=synsets[0].obj_score()
            
            #synsetItem=list(swn.senti_synsets(word,tag))[0]
            #positiveScore=synsetItem.pos_score()
            #positiveScore = swn.senti_synset(str).pos_score()
                sent_pos_score.append(positiveScore)
                sent_neg_score.append(negativeScore)
                sent_obj_score.append(objectiveScore)
        sent_positive_score.append(sent_pos_score)
        sent_negative_score.append(sent_neg_score)
        sent_objective_score.append(sent_obj_score)

#print(sent_positive_score)
findScores()
print(sent_negative_score)


def findTotalandAverageScores():
    for item in sent_positive_score:
        total=0
        for items in item:
            total+=items
        average=total/len(item)
        sent_positive_score.append(total)
        positive_mean.append(average)
    
    for item in sent_negative_score:
        total=0
        for items in item:
            total+=items
        average=total/len(item)
        sent_negative_score.append(total)
        negative_mean.append(average)

    for item in sent_objective_score:
        total=0
        for items in item:
            total+=items
        average=total/len(item)
        sent_objective_score.append(total)
        objective_mean.append(average)
        
    

def findRelStrength():
    for item in sent_positive_score:
        posCount=0
        length=len(item)
        for items in item:
            if items!=0:
                posCount+=1
        sentence_rel=posCount/length
        positive_relative.append(sentence_rel)
    
    for item in sent_negative_score:
        negCount=0
        length=len(item)
        for items in item:
            if items!=0:
                negCount+=1
        sentence_rel=negCount/length
        negative_relative.append(sentence_rel)


    for item in sent_objective_score:
        objCount=0
        length=len(item)
        for items in item:
            if items!=0:
                objCount+=1
        sentence_rel=objCount/length
        objective_relative.append(sentence_rel)


def findStdev():
    for item in sent_positive_score:
        positive_stdev.append(statistics.stdev(item))
    for item in sent_negative_score:
        negative_stdev.append(statistics.stdev(item))
    for item in sent_objective_score:
        objective_stdev.append(statistics.stdev(item))

def maxminScores():
    for item in sent_positive_score:
        positive_maxScore.append(max(item))
        positive_minScore.append(min(item))
    for item in sent_negative_score:
        negative_maxScore.append(max(item))
        negative_minScore.append(min(item))
    for item in sent_objective_score:
        objective_maxScore.append(max(item))
        objective_minScore.append(min(item))



def findMedian():
    for item in sent_positive_score:
        positive_medianScore.append(statistics.median(item))
    for item in sent_negative_score:
        negative_medianScore.append(statistics.median(item))
    for item in sent_objective_score:
        objective_medianScore.append(statistics.median(item))
    


    
'''
        if(newtag!=''):
            synsets=list(swn.senti_synsets(word,newtag))
            print(synsets)
            score=0.0
            if len(synsets)>0:
                for syn in synsets:
                    score+=syn.pos_score()
                sent_score.append(score/len(synsets))
'''



