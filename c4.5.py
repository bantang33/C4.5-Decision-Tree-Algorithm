from sklearn import datasets
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import train_test_split


#Save the decision tree
class decisionnode:
    def __init__(self, feature=None, threshold=None, results=None, Pru=None, left_node=None, right_node=None, max_label=None):
        self.feature = feature   # Represents the feature selected during feature selection
        self.threshold = threshold  # the comparison value, which divides the dataset into 2 categories
        self.results = results  # The category represented by the last leaf node
        self.Pru = Pru  # Represents the loss of each node, used for pruning
        self.left_node = left_node  # Left node in the subtree of the current node
        self.right_node = right_node  # Right node in the subtree of the current node
        self.max_label = max_label  # The current node contains the largest category of labels



def entropy(y):
    '''
    Calculate the entropy of information. Y is the label

    '''
    collection = list(set(y))
    ent = 0
    for labels in collection:
        p = len([label for label in y if label == labels]) / len(y)
        ent -= p * np.log2(p)
    return ent

def info_gain_ratio(ent_x, y_left, y_right):
    '''
    Calculate the information gain ratio.

    '''
    info_gain = ent_x - (len(y_left) / len(y)) * entropy(y_left) - (len(y_right) / len(y)) * entropy(y_right)#Information gain
    info_gain_ratio= info_gain / (ent_x + 1e-10) #Information gain ratio
    return info_gain_ratio

def info_gain_ratio_max(x, y, feature):
    '''
    Calculate the maximum information gain ratio when selecting feature, where x is dataset and y is label.

    '''
    ent_x = entropy(y)
    x_feature = x[:, feature]
    x_feature = list(set(x_feature))
    x_feature = sorted(x_feature)
    gain = 0
    threshold = 0

    for i in range(len(x_feature) - 1):
        threshold_temp = (x_feature[i] + x_feature[i + 1]) / 2
        y_left_index = [i for i in range(
            len(x[:, feature])) if x[i, feature] <= threshold_temp]
        y_right_index = [i for i in range(
            len(x[:, feature])) if x[i, feature] > threshold_temp]
        y_left = y[y_left_index]
        y_right = y[y_right_index]
        gain_ratio = info_gain_ratio(ent_x, y_left, y_right)
        if gain < gain_ratio:
            gain = gain_ratio
            threshold = threshold_temp
    return gain, threshold



def attribute_based_on_GainRatio(x, y):
    '''
    The optimal feature is selected based on the information gain ratio, where x is the dataset and y is label
    '''
    D = np.arange(len(x[0]))
    gain_max = 0
    thre = 0
    d= 0
    for i in D:
        gain, threshold = info_gain_ratio_max(x, y, i)
        if gain_max < gain:
            gain_max = gain
            thre = threshold
            d = i

    return gain_max, thre, d



def split_data(x, y, threshold, feature):
    '''
    The datasets are divided into two categories according to threshold under feature

    '''
    x_in_d = x[:, feature]
    x_left_index = [i for i in range(
        len(x[:, feature])) if x[i, feature] <= threshold]
    x_right_index = [i for i in range(
        len(x[:, feature])) if x[i, feature] > threshold]

    x_left = x[x_left_index]
    y_left = y[x_left_index]
    x_right = x[x_right_index]
    y_right = y[x_right_index]
    return x_left, y_left, x_right, y_right


def Pruning(y):
    '''
    The product of empirical entropy and sample number is calculated for pruning.y is label

    '''
    ent = entropy(y)
    Cart=ent * len(y)
    return Cart


def maxlabel(y):
    '''
    Calculates the label that appears the most times.

    '''
    label_ = Counter(y).most_common(1)
    return label_[0][0]



def buildtree(x, y):
    '''
    Construct the decision tree recursively

    '''
    if y.size > 1:
        gain_max, threshold, feature = attribute_based_on_GainRatio(x, y)
        if (gain_max > 0) :
            x_left, y_left, x_right, y_right = split_data(x, y, threshold, feature)
            left_branch = buildtree(x_left, y_left)
            right_branch = buildtree(x_right, y_right)
            pru = Pruning(y)
            max_label = maxlabel(y)
            return decisionnode(feature=feature, threshold=threshold, Pru=pru, left_node=left_branch, right_node=right_branch, max_label=max_label)
        else:
            pru = Pruning(y)
            max_label = maxlabel(y)
            return decisionnode(results=y[0], Pru=pru, max_label=max_label)
    else:
        pru = Pruning(y)
        max_label = maxlabel(y)
        return decisionnode(results=y.item(), Pru=pru, max_label=max_label)




def classify(dataset, tree):
    if tree.results != None:
        return tree.results
    else:
        v = dataset[tree.feature]
        branch = None

        if v > tree.threshold:
            branch = tree.right_node
        else:
            branch = tree.left_node

        return classify(dataset, branch)

#Extract the dataset
digits = datasets.load_digits()
x=digits.data #features
y=digits.target #labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

decision_tree = buildtree(x_train, y_train)
true_count = 0
for i in range(len(y_test)):
    predict = classify(x_test[i], decision_tree)
    if predict == y_test[i]:
        true_count += 1

print("Accuracy=",true_count/len(y_test))
