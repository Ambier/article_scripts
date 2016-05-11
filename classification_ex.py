"""
Copyright (c) 2016, Sarthak Yadav
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


This script / piece of code demonstrates various Classification Techniques as applied on
the Iris Dataset taken from here :

    https://archive.ics.uci.edu/ml/datasets/Iris

Dependencies :
    1. numpy and scipy : https://www.scipy.org/
    2. pandas : http://pandas.pydata.org/
    3. scikit - learn : http://scikit-learn.org/

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

"""
    Phase 1 : Downloading and Reading the Data

    let's read the dataset. Download the dataset from the link provided above and paste the "iris.data" file
    in the same directory
"""

iris = pd.read_csv("./datasets/iris.data")

# let's print some quick info about the dataset. Total number of datapoints = 150
print iris.info()

# let's print a lil data
print iris.head()

"""
 the target column "class" is of data type "str", and consists of 3 unique values
        1. Iris-virginica
        2. Iris-setosa
        3. Iris-versicolor

 since Classifiers don't play well directly with string data, we have to transform these string values
 into categorical integer values, representing classes
"""


def transform_class(x):
    if x == 'Iris-setosa':
        return 0
    elif x == 'Iris-versicolor':
        return 1
    else:
        return 2

# this line transforms the class column and add's the result as new column in the data set
iris.insert(iris.shape[1], "target", iris['class'].apply(transform_class))

"""
    Since the no of datapoints is extremely small (150), we will use cross validation for
    comparing various Classifiers

    for that, let us first shuffle our data, which is currently sequentially ordered in terms of classes
"""
iris = iris.iloc[np.random.permutation(len(iris))]      # to shuffle the sequence of data points

features = iris.columns[:-2]        # contains the diff features, leaving out the class and the target columns

"""
    Phase 2 : Classifier Comparison

            We Will -

            1. Study the effect of no of features used (out of a total of 4) for 4 different classifiers
                namely
                    A regression based Classifier : Logistic Regression
                    A tree based classifier : Decision Trees
                    Support Vector Machines
                    An ensemble classifier : Random Forest

            2. We'll plot a graph that shows this result

    Let's write a method for aiding in the process
"""


def model_performance(model, n_features):
    """

    This method uses Stratified Cross Validation to analyze the performance of the given Classification model
    on different number of features used.

    :param model: the Classifier we wish to analyze
    :param n_features: List of integers. Each Integer represents the corresponding num of features used
    :return: avg_accuracies - The avg accuracy scores on each corresponding number of features used
    """

    if 0 in n_features:
        raise ValueError("Number of features cannot be zero")

    skf = StratifiedKFold(iris.target.values,           # the Cross validation Technique we use for our example
                          n_folds=5,
                          shuffle=False,
                          random_state=42)
    avg_accuracies = []

    for x in n_features:
        index = 0
        avg_acc = 0.0
        for train_index, test_index in skf:
            visibletrain = iris.iloc[train_index]
            blindtrain = iris.iloc[test_index]
            x_vtrain = visibletrain[features[:x]]           # this statement selects the features to use.
            x_btrain = blindtrain[features[:x]]
            clf = model
            clf.fit(x_vtrain, visibletrain.target.values)

            preds = clf.predict(x_btrain)
            acc = accuracy_score(blindtrain.target.values, preds)

            avg_acc += acc
            index += 1

            del clf
            #print "\n"

        print "\tAverage Accuracy Score : %f | No of features : %d" % (avg_acc/index, x)
        avg_accuracies.append(avg_acc/index)
    return avg_accuracies


"""
    Phase 3 : Measure Performance

    - Here we get results of model_performance method we described above on different models.
    - Each of these models comprise of almost no parameter tuning, and are used with default parameters
    - we then visualize the results
"""
n_feats = [1, 2, 3, 4]
print "Model1 : Logistic Regression"
model1 = model_performance(LogisticRegression(max_iter=10), n_features=n_feats)
print "\nModel2 : Support Vector Machine"
model2 = model_performance(SVC(), n_feats)
print "\nModel3 : Decision Tree"
model3 = model_performance(DecisionTreeClassifier(), n_feats)
print "\nModel4 : Random Forest"
model4 = model_performance(RandomForestClassifier(), n_feats)

# lets visualize the results

fig = plt.figure(figsize=(10,8))
plt.xlabel("Features")
plt.ylabel("Accuracy Score")
plt.title('Studying Classifier Performance on Iris Data-set')
plt.plot(n_feats, model1, 'r--', label='Logistic Regression')
plt.plot(n_feats, model2, 'g--', label='Support Vector Machine')
plt.plot(n_feats, model3, 'b--', label='Decision Tree')
plt.plot(n_feats, model4, 'y--', label='Random Forest')
plt.legend(loc='upper left', shadow=True)
plt.savefig("test.png")
plt.show()
