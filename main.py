# package use
# NLTK ,numpy,tflearn ( deep learning) ,tensorflow


import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
with open("intents.json") as file:
    data=json.load(file)

try: # try to open the file to fetch data
    with open('data.pickle',"rb") as f: # rb=readbyte
        words,labels,training,output=pickle.load(f)
except: # initialize label parameters if not exists
    words=[]
    labels=[]
    docs_x=[] # for saving that word
    docs_y=[] # for saving that intent[tag]  use both to figured out the word in which tag
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds=nltk.word_tokenize(pattern) # take all the different word of string convert to list
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words=[stemmer.stem(w.lower()) for w in words if w not in "?"] # stem the words and put it to lowercase,remove ?
    words=sorted(list(set(words))) # remove duplicate

    labels=sorted(labels)

    # input on neural network only understand number ( encode the string to number )

    # one hot encode if word exist add 1 ( convert into bag of words)

    training=[]
    output=[]

    out_empty=[0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag=[] # bag of words

        wrds=[stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds: # if word exists then add 1 into this bag of words
                bag.append(1)
            else:
                bag.append(0)

        output_row=out_empty[:] # list of all 0 in labes
        output_row[labels.index(docs_y[x])]=1 # set that label location =1

        training.append(bag)
        output.append(output_row)

    # put in to numpy array to use tflearn
    training=numpy.array(training)
    output=numpy.array(output)

    with open('data.pickle',"wb") as f:
        pickle.dump((words,labels,training,output),f)

# put in neural network
tensorflow.compat.v1.reset_default_graph()
net=tflearn.input_data(shape=[None,len(training[0])])  # tflearn is very similar to tensorflow
# add 3 layers
net=tflearn.fully_connected(net,8) # add hidden layers
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation='softmax') # output result
net=tflearn.regression(net)

model=tflearn.DNN(net) # apply deep neural network

try: # try to load created training model ( if exists we wont retrain it again)
    model.load("model.tflearn")
except:
    # passing data into model and training
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True) # n_epoch number of time use gonna see the same data
    model.save("model.tflearn")

# chat box

def bag_of_words(s,words):
    bag= [0 for _ in range(len(words))]

    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if se==w:
                bag[i]=1
                break
    return numpy.array(bag)

def chat(text):
    print('Hello how can i help you ( type quit to stop !!) ')
    print('Trained to answer  topics about (greet,goodbye,age,name,sell,shop,open hour)')
    results=model.predict([bag_of_words(text,words)]) # prediction probality of return value
    results_index=numpy.argmax(results) # give the index of the greatest value in results list

        # print out the response
    result_tag=labels[results_index]

    if results[0][results_index]>=0.75: # > 0.75% probability then  print result
        for tag in data['intents']:
            if tag['tag']==result_tag:
                responses=tag['responses']
                break
        return random.choice(responses) # pick random result in that list
    else:
        return 'I can not understand your question'


