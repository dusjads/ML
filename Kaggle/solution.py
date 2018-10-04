import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Random Forest 8", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    RandomForestClassifier(max_depth=8, n_estimators=8, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# download the file
dataset = pd.read_csv("train.csv")
# separate the data from the target attributes
X = dataset.values[:,1:28]
y = dataset.values[:,28]

def convert(v):
    if type(v) is str:
        return 17*ord(v) % 19
    return v

X = np.array(list(map(lambda ar: list(map(convert, ar)), X)))
print(y, X)
X = preprocessing.scale(X)

best_sol = {
    'name' : '',
    'features_num' : 0,
    'score' : 0
    }

def update(name_, features_num_, score_):
    best_sol['name'] = name_
    best_sol['features_num'] = features_num_
    best_sol['score'] = score_

#------------------------------ First -----------------------------------------

X_trans = StandardScaler().fit_transform(X)


# iterate over classifiers
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X_trans, y, cv=5)
    score = scores.mean()
    if best_sol['score'] < score:
        update(name, -1, score)

#------------------------------ Second -----------------------------------------

for k in range(1, X.shape[1] + 1):
    X_trans = SelectKBest(k=k).fit_transform(X, y)
    for name, clf in zip(names, classifiers):
        scores = cross_val_score(clf, X_trans, y, cv=5)
        score = scores.mean()
        if best_sol['score'] < score:
            update(name, k, score)
print(best_sol)


dataset = pd.read_csv("test.csv")
test_data = dataset.values[:,1:28]
test_data = np.array(list(map(lambda ar: list(map(convert, ar)), test_data)))
test_data = preprocessing.scale(test_data)
d = dict(zip(names, classifiers))
clf = d[best_sol['name']]

if best_sol['features_num'] == -1:
    trans = StandardScaler().fit(X, y)
    X_trans = trans.transform(X)
    clf.fit(X_trans, y)
    test_data = trans.transform(test_data)
    predicted = clf.predict(test_data)
else:
    trans = SelectKBest(k=best_sol['features_num']).fit(X, y)
    X_trans = trans.transform(X)
    clf.fit(X_trans, y)
    print(clf.score(X_trans, y))
    test_data = trans.transform(test_data)
    predicted = clf.predict(test_data)

pd.DataFrame(data = predicted, index = range(1, len(predicted)*2, 2), columns = ['class']).to_csv("predicted.csv", header = True, index_label = 'id')
