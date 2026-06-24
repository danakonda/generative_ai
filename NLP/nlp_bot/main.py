import random 
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
intents=json.loads(open(r'C:\Users\raj93\OneDrive\Desktop\nlp_bot\data\intents.json').read())#read file intents.json
words=[]
classes=[]
documents=[]
ignoreLetters=["?","!",".",","]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        wordList=nltk.word_tokenize(pattern)#create tokens
        words.extend(wordList)#stores
        documents.append((wordList,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words=[lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]#
words=sorted(set(words))#remove duplicates.

classes=sorted(set(classes))

pickle.dump(words,open("words.pkl","wb"))#binary form
pickle.dump(classes,open("classes.pkl","wb"))
training=[]
outputEmpty=[0]*len(classes)#template of output data and each one 

for document in documents:
    bag=[]
    wordPatterns=document[0]
    wordPatterns=[lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    outputRow=list(outputEmpty)
    outputRow[classes.index(document[1])]=1
    training.append(bag+outputRow)

random.shuffle(training)
training=np.array(training)

trainX=training[:,:len(words)]#row
trainY=training[:,len(words):]#column
#create model
model=tf.keras.Sequential()#list layers one input layer and output layer
model.add(tf.keras.layers.Dense(128,input_shape=(len(trainX[0]),),activation="relu"))#dense layer fully connect nuerons of 128,trainX[0] give number of feature in our input data ,non linear model relu is help complex pattern
model.add(tf.keras.layers.Dropout(0.5))#dropout layers 50% of neurons each training overfitting
model.add(tf.keras.layers.Dense(64,activation='relu'))#second layers this time 64 neurons,activation relu 
model.add(tf.keras.layers.Dropout(0.5))#another dropout layers 50%
model.add(tf.keras.layers.Dense(len(trainY[0]),activation="softmax"))#output layers number of neuron == number of classes which means catagirias,activation softmax is multi class problem and its output probably distributed to class


sgd=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True)#stochastic gradient descent is sgd,update model parameters to minimize the loss this optimize,learning_rate is how much change to estimate error model weights updates ,nesterov is momentom slit modification and improve performens

model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])#loss function multi class problem,metrcis tracking accuracy during training
hist=model.fit(
    np.array(trainX),np.array(trainY),epochs=250,batch_size=5,verbose=1
)#input output data covert arrays,epochs is number cycle,sample is batch_size,verbose is progress of training  
model.save("chatbot_model.h5",hist)
print("Done")