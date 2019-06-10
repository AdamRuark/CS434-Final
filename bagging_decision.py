#!/usr/bin/env python3

# Authors: Adam Ruark and Jacob Berwick

import sys
import numpy as np
import math
from sklearn import tree
import random


def main(out_file):
    # Load in data
    random.seed()
    # train_features_103 = load_features('data/feature103_Train.txt', 103)
    # test_features_103 = load_features('data/features103_test.txt', 103, train=False)
    # models = decision_tree(train_features_103, 10)
    # predictions = predict(models, test_features_103)

    # print(f'Writing predictions to: {out_file}')
    # write_pred(predictions, test_features_103[:,-1], out_file)

    train_features_1053 = load_features('data/featuresall_train.txt', 1053)
    test_features_1053 = load_features('data/featuresall_test.txt', 1053, train=False)
    print('Training...')
    models = decision_tree(train_features_1053, 10)
    print('Predicting...')
    predictions = predict(models, test_features_1053)
    
    print(f'Writing predictions to: {out_file}')
    write_pred(predictions, test_features_1053[:,-1], out_file)


def acc(guess, actual):
    total_acc = 0
    for i, j in zip(guess, actual):
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
    # Get random validation set
    feature_count = len(data[0]) - 1
    # idx = np.random.randint(data.shape[0], size=int(len(data)/10))
    # validation = data[idx,:]
    # data = np.delete(data, idx, axis=0)
    # X_valid, Y_valid, ids_valid = validation[:,:feature_count - 2], validation[:,feature_count - 1], validation[:,feature_count]

    # Train 20 models
    bag_models = []
    for i in range(20):
        bag_data = data[np.random.choice(data.shape[0], size=len(data), replace=True)]
        X_train, Y_train = bag_data[:,:feature_count - 2], bag_data[:,feature_count - 1]

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        
        bag_models.append(clf)
        print(f'Trained model {i + 1}')

    return bag_models

    # Validate the accuracy of each model
    # predictions = np.zeros(len(X_valid), dtype=float)
    # for mod in bag_models:
    #     predictions += np.array(mod.predict(X_valid), dtype=float)
    
    # predictions = predictions / float(len(bag_models))
    # # write_pred(predictions, ids_valid, Y_valid)

    # for i, val in enumerate(predictions):
    #     predictions[i] = 0 if val < 0.5 else 1

    # print(acc(predictions, Y_valid))


def predict(models, data):
    # Vote by averaging the prediction for each feature from all the models
    feature_count = len(data[0]) - 3
    data = data[:,:feature_count]
    predictions = np.zeros(len(data), dtype=float)
    for model in models:
        predictions += np.array(model.predict(data), dtype=float)
    return predictions / float(len(models))


def write_pred(predictions, ids, file_name):
    f = open(file_name,'w+')
    for i, pred in enumerate(predictions):
        f.write(f'{ids[i]},{pred}\n')
    f.close()


def load_features(file_name, data_length, train=True):
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

        line = line.strip('\n\t').split('\t')
        if train:
            features.append([float(i) for i in line[2:]])
            ids.append(line[0])
            classes.append(int(line[1]))
        else:
            features.append([float(i) for i in line[1:]])
            ids.append(line[0])
            # Arbitrary testing classes, this field doesn't exist and this junk data won't be considered
            classes.append(-1)
    f.close()

    # Reformat data
    data = np.array(features)
    data = np.append(data, np.array([classes]).T, axis=1)
    data = np.append(data, np.array([ids]).T, axis=1)
    return data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Specify outfile name")
        exit(1)
    else:
        main(sys.argv[1])
