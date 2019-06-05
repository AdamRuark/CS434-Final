#!/usr/bin/env python3

# Authors: Adam Ruark and Jacob Berwick

import sys
import numpy as np
import math
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn import tree
import random
mp.use('Agg') 

def main():
    # Load in data
    random.seed()
    # pks_data = load_fasta('data/pks_Train.fasta')
    # pkfs_data = load_fasta('data/pkfs_Train.fasta')
    major_features = load_features('data/feature103_Train.txt', 103)
    # all_features = load_features('data/featuresall_train.txt', 1053)

    decision_tree(major_features, 10)


def acc(guess, actual):
    total_acc = 0
    for i, j in zip(guess, actual):
        # print(f'{type(i)} - {type(j)}')
        if float(i) == float(j):
            total_acc += 1

    return total_acc / len(actual)


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


def decision_tree(data, k):
    best_clf = None
    best_acc = 0

    # Get random validation set
    idx = np.random.randint(data.shape[0], size=int(len(data)/10))
    validation = data[idx,:]
    data = np.delete(data, idx, axis=0)
    X_valid, Y_valid = validation[:,:102], validation[:,103]
    bag_models = []

    # Train 10 models
    for i in range(1):
        bag_data = data[np.random.choice(data.shape[0], size=len(data), replace=True)]
        X_train, Y_train = bag_data[:,:102], bag_data[:,103]

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        
        bag_models.append(clf)
        print(f'Trained model {i + 1}')

    # Validate the accuracy of each model
    predictions = np.zeros(len(X_valid), dtype=float)
    for mod in bag_models:
        predictions += np.array(mod.predict(X_valid), dtype=float)
    
    predictions = predictions / float(len(bag_models))

    for i, val in enumerate(predictions):
        predictions[i] = 0 if val < 0.5 else 1

    
    print(predictions)
    print(Y_valid)
    print(acc(predictions, Y_valid))

    return predictions

        
def load_features(file_name, data_length):
    classes, ids = [], []
    f = open(file_name, 'r')
    features = []

    # Skip header line 
    f.readline()

    while True:
        # Read in id and class
        line = f.readline()
        if not line:
            break

        line = line.strip('\n').split('\t')
        features.append([float(i) for i in line[2:]])
        ids.append(line[0])
        classes.append(int(line[1]))

    f.close()

    # Reformat data
    data = np.array(features)
    data = np.append(data, np.array([classes]).T, axis=1)
    data = np.append(data, np.array([ids]).T, axis=1)
    return data


if __name__ == "__main__":
    main()
