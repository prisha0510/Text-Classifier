import pandas as pd
import math
import numpy as np
import os
import re

def get_reviews(path_pos,path_neg):
    dir_list_pos = os.listdir(path_pos)
    dir_list_neg = os.listdir(path_neg)
    train_data = []
    y=[] 
    num_pos=0
    num_neg=0
    for i in range(len(dir_list_pos)):
        with open(path_pos + '/' + dir_list_pos[i], encoding="utf8") as f:
            contents = f.readlines()[0]
            train_data.append(contents)
            y.append(1)
            num_pos+=1

    for i in range(len(dir_list_neg)):
        with open(path_neg + '/' + dir_list_neg[i], encoding="utf8") as f:
            contents = f.readlines()[0]
            train_data.append(contents)
            y.append(0)
            num_neg+=1
    
    y= np.array(y)
    y=y.reshape((len(train_data),1))
    return train_data,num_pos,num_neg, y

path_pos = "C:/Users/prish/Downloads/part1_data/part1_data/train/pos" #provide path of positive training data
path_neg = "C:/Users/prish/Downloads/part1_data/part1_data/train/neg" #provide path of negative training data
train_data,num_pos,num_neg, y=get_reviews(path_pos,path_neg)

vocab = set()
for i in range(len(train_data)):
    words = re.split(r'[.,;!><*-/?() ''@&""{}|\~`#$%^_+=:]', train_data[i])
    for j in range(len(words)):
        if len(words[j])>0:
            vocab.add(words[j].lower())
list_vocab=list(vocab)
vocab_dict={}
for i in range(len(list_vocab)):
    vocab_dict[list_vocab[i]]=i
x= np.zeros((num_pos+num_neg,len(list_vocab)))

for i in range(len(train_data)): # number of examples, 1....m
    words= re.split(r'[.,;!><*-/?() ''@&""{}|\~`#$%^_+=:]', train_data[i])
    for word in words:
        if(len(word)>0):
            x[i][vocab_dict[word.lower()]] = 1

phi_pos=[]
phi_neg=[] 
x_temp1 = np.logical_and(y,x)
x_temp2 = np.logical_and(np.logical_not(y),x)
num = x_temp1.sum(axis=0)
den = x_temp2.sum(axis=0)
phi_pos = (num+1) / (num_pos+2)
phi_neg = (den +1)/(num_neg+2)

def predict(review,vocab_dict,phi_pos,phi_neg,p_pos,p_neg):
    words= re.split(r'[.,;!><*-/?() ''@&""{}|\~`#$%^_+=:]', review)
    prod1= p_pos
    prod0= p_neg
    for word in words:
        if(len(word)>0):
            prod1 = prod1 * phi_pos[vocab_dict[word.lower()]]
            prod0 = prod0 * phi_neg[vocab_dict[word.lower()]]
    prob = prod1 / (prod1 + prod0)
    if(prob>=0.5):
        return 1
    else:
        return 0
y_predict= []
p_pos=num_pos/(num_pos+num_neg)
p_neg=num_neg/(num_pos+num_neg)
for i in range(len(train_data)):
    y_predict.append(predict(train_data[i],vocab_dict,phi_pos,phi_neg,p_pos,p_neg))
y_predict= np.array(y_predict)
y_predict=y_predict.reshape(len(train_data),1)
wrong = (np.logical_xor(y,y_predict)).sum()
accuracy = (len(train_data)-wrong)/len(train_data)
print ("Accuracy in percentage on training data is : " + str(100*accuracy))

def test_set(test_data,vocab_dict,phi_pos,phi_neg):
    path_1 = "C:\\Users\\prish\\Downloads\\part1_data\\part1_data\\test\\pos" # provide the path of the positive test data
    path_2 = "C:\\Users\\prish\\Downloads\\part1_data\\part1_data\\test\\neg" # provide the path of the negative test data
    test_data,num_pos,num_neg, y = get_reviews(path_1,path_2)
    p_pos=num_pos/(num_pos+num_neg)
    p_neg=num_neg/(num_pos+num_neg)
    y_predict= []
    for i in range(len(test_data)):
        y_predict.append(predict(test_data[i],vocab_dict,phi_pos,phi_neg,p_pos,p_neg))
    y_predict= np.array(y_predict)
    y_predict=y_predict.reshape(len(test_data),1)
    wrong = (np.logical_xor(y,y_predict)).sum()
    accuracy = (len(test_data)-wrong)/len(test_data)
    print ("Accuracy in percentage on test data is : " + str(100*accuracy))
