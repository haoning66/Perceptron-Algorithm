#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

#TODO: understand that you should not need any other imports other than those already in this file; if you import something that is not installed by default on the csug machines, your code will crash and you will lose points

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "/u/cs246/data/adult/" #TODO: if you are working somewhere other than the csug server, change this to the directory where a7a.train, a7a.dev, and a7a.test are on your machine
#DATA_PATH = "/Users/shinshukuni/Desktop/perceptron/adult"

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray(ys), np.asarray(xs) #returns a tuple, first is an array of labels, second is an array of feature vectors

def perceptron(train_ys, train_xs, dev_ys, dev_xs, args):
    weights = np.zeros(NUM_FEATURES)
    #TODO: implement perceptron algorithm here, respecting args
    iter=1
    temp1=1.0
    temp2=0.0
    acc_list=[]
    while(iter<=args.iterations):
        for n in range(0, len(train_ys)):
            res = np.dot(weights.T, train_xs[n])
            if res < 0:
                tn = -1
            elif res == 0:
                tn = 0
            elif res > 0:
                tn = 1
            if tn != train_ys[n]:
                weights = weights + args.lr*(np.dot(train_ys[n], train_xs[n]))
        if args.nodev==False:
            err=0
            for m in range(0, len(dev_ys)):
                res_dev = np.dot(weights.T, dev_xs[m])
                if res_dev < 0:
                    tn_dev = -1
                elif res_dev == 0:
                    tn_dev = 0
                elif res_dev > 0:
                    tn_dev = 1
                if tn_dev !=dev_ys[m]:
                    err+=1
            err_rate=err/len(dev_ys)
            acc_list.append(err_rate)
            if iter%10==0 and iter!=0:
                sum=0
                for i in range(iter-10,iter):
                    sum+=acc_list[i]
                temp2 = sum/10
        if temp2>temp1:
            break
        if temp2!=0.0:
            temp1 = temp2
        iter+=1
    return weights

def test_accuracy(weights, test_ys, test_xs):
    accuracy = 0.0
    #TODO: implement accuracy computation of given weight vector on the test data (i.e. how many test data points are classified correctly by the weight vector)
    err=0
    for i in range(0,len(test_ys)):
        res = np.dot(weights.T, test_xs[i])
        if res < 0:
            tn = -1
        elif res == 0:
            tn = 0
        elif res > 0:
            tn = 1
        if tn !=test_ys[i]:
            err+=1
        accuracy=1.0-(err/len(test_ys))
    return accuracy

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate to use for update in training loop.')
    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')
    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.train_file: str; file name for training data.
    args.dev_file: str; file name for development data.
    args.test_file: str; file name for test data.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    weights = perceptron(train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(weights, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))

    # x=[1,2,3,4,5]
    # a=[]
    # for item in x:
    #     accuracy=test_accuracy(perceptron(train_ys,train_xs,dev_ys,dev_xs,item),test_ys,test_xs)
    #     a.append(accuracy)
    #
    # plt.plot(x,a)
    # plt.show()


if __name__ == '__main__':
    main()
