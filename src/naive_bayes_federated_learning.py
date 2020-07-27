from math import sqrt
from math import exp
from math import pi
import numpy as np
# import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from configparser import ConfigParser

import phe as paillier

seed = 1234
np.random.seed(seed)

###################################################
################ HELPER FUNCTIONS #################
###################################################

def print_example_banner(title, ch='*', length=60):
    headerFooter = (ch * len(title)) + 2 * ch
    spaced_text = ' %s ' % title
    print(headerFooter.center(length, ch))
    print(spaced_text.center(length, ch))
    print(headerFooter.center(length, ch))

def get_data(n_parties, dataset='iris'):
    """
    Import the dataset via sklearn, shuffle and split train/test.
    Return training, target lists for `n_parties` and a holdout test set
    """
    print("Loading data...")
    if dataset == 'iris':
        data = load_iris()
    elif dataset == 'wine':
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # shuffle and split the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,  random_state=seed)
    # prepare for encoding
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    # Split train among multiple parties.
    # The selection is not at random. We simulate the fact that each party
    # sees a potentially very different sample of flowers.
    X, y = [], []
    step = int(X_train.shape[0] / n_parties)
    for c in range(n_parties):
        X.append(X_train[step * c: step * (c + 1), :])
        y.append(y_train[step * c: step * (c + 1)])

    return X, y, X_test, y_test

def mean_square_error(y_pred, y):
    """ 1/m * \sum_{i=1..m} (y_pred_i - y_i)^2 """
    return np.mean((y - y_pred) ** 2)

def accuracy_metric(actual, predicted):
    """Calculate accuracy percentage"""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def encrypt_vector(public_key, x, y):
    class_labels = [public_key.encrypt(i) for i in x]
    predictions = [public_key.encrypt(i) for i in y]
    return (class_labels, predictions)

def decrypt_vector(private_key, x, y):
    class_labels = np.array([private_key.decrypt(i) for i in x])
    predictions = np.array([private_key.decrypt(i) for i in y])
    return (class_labels, predictions)

def separate_by_class(dataset):
    """"Split the dataset by class values, returns a dictionary"""
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def mean(numbers):
    """"Calculate the mean of a list of numbers"""
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    """"Calculate the standard deviation of a list of numbers"""
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)

def summarize_dataset(dataset):
    """"Calculate the mean, stdev and count for each column in a dataset"""
    summaries = [(mean(column), stdev(column), len(column))
                for column in zip(*dataset)]
    del(summaries[-1]) # delete the last element
    return summaries

def summarize_by_class(dataset):
    """"Split dataset by class then calculate statistics for each row"""
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in sorted(separated.items()):
        summaries[class_value] = summarize_dataset(rows)
    return summaries

def calculate_probability(x, mean, stdev):
    """"Calculate the Gaussian probability distribution function for x"""
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    """"Calculate the probabilities of predicting each class for a given row"""
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

def calculate_marginal_probability(probabilities):
    """"Marginal probability is the sum of probabilities for all classes"""
    return sum(probabilities.values())

def calculate_posterior_probabilities(class_probabilities, marginal_probability):
    """"Calculate posterior probability for each class"""
    posterior_probabilities = dict()
    for target, class_probability in class_probabilities.items():
        posterior_probabilities[target] = class_probability / marginal_probability
    return posterior_probabilities

def predict_row(summaries, row):
    """Predict the class for a given row"""
    class_probabilities = calculate_class_probabilities(summaries, row)
    marginal_probability = calculate_marginal_probability(class_probabilities)
    posterior_probabilities = calculate_posterior_probabilities(class_probabilities, marginal_probability)
    best_label, best_prob = None, -1
    for class_value, probability in posterior_probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label, best_prob


class Master:
    """Private key holder. Decrypts the class labels and probabilities"""

    def __init__(self, key_length):
         keypair = paillier.generate_paillier_keypair(n_length=key_length)
         self.pubkey, self.privkey = keypair

    def decrypt_aggregate(self, encrypt_label, encrypt_prob):
        return decrypt_vector(self.privkey, encrypt_label, encrypt_prob)


class Party:
    """Runs naive bayes with local data.

    Using public key can encrypt locally computed summary class values
    and probabilities.
    """

    def __init__(self, name, X, y, pubkey):
        self.name = name
        self.train = np.c_[X, y]
        self.pubkey = pubkey
        self.summaries = dict()
    
    def fit(self):
        """Naive Bayes Algorithm"""
        self.summaries = summarize_by_class(self.train)

    def predict(self, X):
        """Score test data"""
        class_labels, predictions = list(), list()
        for row in X:
            output_class, output_prob = predict_row(self.summaries, row)
            class_labels.append(output_class)
            predictions.append(output_prob)
        return(class_labels, predictions)

    def encrypt_summaries(self, X):
        """Compute and encrypt class labels and probabilities"""
        self.fit()
        class_labels, predictions = self.predict(X)
        encrypted_labels, encrypted_summaries = encrypt_vector(self.pubkey, class_labels, predictions)
        
        return (encrypted_labels, encrypted_summaries)


def federated_learning(X, y, X_test, y_test, config):
    """"
        First each party learns a model on its respective dataset for comparison
        then encrypts the data and sends back to the master. After, we will
        contstruct permutation mapping table to determine class label
    """
    n_parties = config['n_parties']
    names = ['Party {}'.format(i) for i in range(1, n_parties + 1)]

    # Instantiate the master and generate private and public keys
    master = Master(key_length=config['key_length'])

    # Instantiate the parties.
    # Each party gets the public key at creation and its own local dataset
    parties = []
    for i in range(n_parties):
        parties.append(Party(names[i], X[i], y[i], master.pubkey))

    # Compute class label predictions and encrypt
    class_labels, predictions = [], []
    for p in parties:
        encrypt_label, encrypt_prob = p.encrypt_summaries(X_test)
        # Send encrypted data to master to decrypt
        _labels, _pred = master.decrypt_aggregate(encrypt_label, encrypt_prob)
        # Append to class_labels and predictions
        class_labels.append(_labels)
        predictions.append(_pred)
    
    # Build permutation mapping table and find the best prediction
    y_pred = []
    for i in range(len(y_test)):
        perm_table = np.zeros((len(predictions),(len(predictions))))
        for j in range(len(predictions)):
            for k in range(len(predictions)):
                if k == j:
                    perm_table[j][j] = 1
                else:
                    if perm_table[j][k] != 0:
                        continue
                    if predictions[j][i] - predictions[k][i] >= 0:
                        perm_table[j][k] = 1
                        perm_table[k][j] = 1 if (predictions[j][i] - predictions[k][i] == 0) else -1
                    else:
                        perm_table[j][k] = -1
                        perm_table[k][j] = 1
        res = list(map(sum, perm_table))
        y_pred.append(class_labels[res.index(max(res))][i])

    print('After running the protocol:')
    print('Accuracy Score(%): {:.2f}\tMSE: {:.2f}'.format( \
        accuracy_metric(y_pred, y_test), mean_square_error(y_pred, y_test)))


def local_learning(X, y, X_test, y_test, config):
    n_parties = config['n_parties']
    names = ['Party {}'.format(i) for i in range(1, n_parties + 1)]

    # Instantiate the parties.
    # Each party gets the public key at creation and its own local dataset
    parties = []
    for i in range(n_parties):
        parties.append(Party(names[i], X[i], y[i], None))

    # Each party trains a naive bayes on its own data
    print('Accuracy Score and error (MSE) that each party gets on test '
          'set by training on own local data:')
    for p in parties:
        p.fit()
        y_pred, _ = p.predict(X_test)
        print('{:s}:\tAccuracy Score(%): {:.2f}\tMSE: {:.2f}'.format(p.name, \
            accuracy_metric(y_pred, y_test), mean_square_error(y_pred, y_test)))


if __name__ == '__main__':
    file = "config.ini"
    configuration = ConfigParser()
    configuration.read(file)
    try:
        for n_parties in configuration['HE_paramenters']['n_parties'].split(","):
            for key_length in configuration['HE_paramenters']['key_length'].split(","):
                config = {
                    'n_parties': int(n_parties.strip()),
                    'key_length': int(key_length.strip()),
                }
                print_example_banner(f"EXAMPLE USING {n_parties.strip()} PARTIES AND {key_length.strip()} KEY LENGTH", ch='#')
                print_example_banner(f"IRIS DATASET")
                # load data, train/test split and split training data between parties
                X, y, X_test, y_test = get_data(n_parties=config['n_parties'])
                # first each party learns a model on its respective dataset for comparison.
                local_learning(X, y, X_test, y_test, config)
                # and now the full glory of federated learning
                federated_learning(X, y, X_test, y_test, config)
                print_example_banner(f"WINE DATASET")
                # load data, train/test split and split training data between parties
                X, y, X_test, y_test = get_data(n_parties=config['n_parties'], dataset='wine')
                # first each party learns a model on its respective dataset for comparison.
                local_learning(X, y, X_test, y_test, config)
                # and now the full glory of federated learning
                federated_learning(X, y, X_test, y_test, config)
                print_example_banner(f"BREAST CANCER DATASET")
                # load data, train/test split and split training data between parties
                X, y, X_test, y_test = get_data(n_parties=config['n_parties'], dataset='breast')
                # first each party learns a model on its respective dataset for comparison.
                local_learning(X, y, X_test, y_test, config)
                # and now the full glory of federated learning
                federated_learning(X, y, X_test, y_test, config)
    except ImportError:
        print ("config.ini file does not have the correct format!\nPlease use ',' for multi-value separation.")
