#!/usr/bin/env python3

# Authors: Adam Ruark and Jacob Berwick

import sys
import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import LSTM, Dense
import math
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn import tree
import random
mp.use('Agg') 

count = 0

def main():
    pks_data = load_fasta('data/pks_Train.fasta')
    pkfs_data = load_fasta('data/pkfs_Train.fasta')
    random.seed()
    x_train = []
    id_train = []
    y_train = []
    x_valid = []
    id_valid = []
    y_valid = []

    leng = len(pks_data)/10

    for i,data in enumerate(pks_data):
        id, x = data
        if i > leng:
            id_train.append(id)
            x_train.append(list(x))
            y_train.append(1)
        else:
            id_valid.append(id)
            x_valid.append(list(x))
            y_valid.append(1)

    leng = len(pkfs_data)/10

    for i,data in enumerate(pkfs_data):
        id, x = data
        if i > leng:
            id_train.append(id)
            x_train.append(list(x))
            y_train.append(0)
        else:
            id_valid.append(id)
            x_valid.append(list(x))
            y_valid.append(0)

    types = {
        'A': [1,0,0,0],
        'C': [0,1,0,0],
        'G': [0,0,1,0],
        'U': [0,0,0,1],
        'N': [1,1,1,1],
        'R': [1,0,1,0],
        'M': [1,1,0,0],
        'S': [0,1,1,0],
        'K': [0,0,1,1],
        'W': [1,0,0,1],
        'Y': [0,1,0,1],
        'H': [1,1,0,1],
        'V': [1,1,1,0],
        'D': [1,0,1,1],
        'B': [0,1,1,1],
        'P': [0,0,0,0],
        'O': [0,0,0,0],
        'M': [0,0,0,0],
        'X': [1,1,1,1]
    }


    data = []
    for seq in x_train:
        seqs = []
        for x in seq:
            seqs.append(np.array(types[str(x)]))
        data.append(np.array(seqs))
    valid = []
    for seq in x_valid:
        seqs = []
        for x in seq:
            seqs.append(np.array(types[str(x)]))
        valid.append(np.array(seqs))

    model = Sequential()
    # model.add(Embedding(1000, 64))
    model.add(LSTM(100, return_sequences=True, input_shape=(None,4)))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    max_acc = 0
    curr_acc = 0
    for epoch in range(1,1001):
        print(f'Epoch: {epoch}')
        for i in range(100):            
            print('example: ' + str(i) + '\n')
            #model.fit(np.array([row]),np.array([y_train[i]]),batch_size=1,epochs=1,verbose=1)
            model.fit_generator(generator(data,y_train),steps_per_epoch=1, epochs=1, verbose=0)
            # scores = model.evaluate(x_valid, y_valid, verbose=0)
            # print("Accuracy: %.2f%%" % (scores[1]*100))
        guesses = []
        for i, row in enumerate(valid):
            guesses.append(model.predict(np.array([row]),batch_size=1)[0])
        curr_acc = acc(guesses,y_valid)
        print(f'Curr Acc: {curr_acc}')
        if curr_acc > max_acc:
            with open('max_epoch1.txt','w+') as f:
                    f.write('Epoch: ' + str(epoch))
                    f.write('Acc: ' + str(curr_acc))
            model_json = model.to_json()
            with open("model1.json", "w+") as json_file:
                json_file.write(model_json)
            model.save_weights("model1.h5")

    return

def generator(data, y):
    global count
    c = np.random.randint(len(data))
    x,y = np.array([data[c]]), np.array([y[c]])
    # print('gen: ' + str(c))
    yield x,y

# def pred_gen(valid):
#     global val
#     x = np.array([valid[val%len(valid)]])
#     val += 1
#     yield x

def acc(guess, y):
    corr = 0
    for i,g in enumerate(guess):
        temp = 1 if g[0] > 0.5 else 0
        if temp == y[i]:
            corr += 1
    return corr/len(guess)
        


def load_fasta(file_name):
    data = []
    f = open(file_name, 'r')

    while True:
        # Read in sequence data
        sequence_id = f.readline()[1:].strip('\n')
        sequence_str = f.readline().strip('\n')
        if not sequence_str: 
            break
        # Tuple of (id, sequence)
        data.append((sequence_id, sequence_str))

    f.close()  
    return data

if __name__ == "__main__":
    main()