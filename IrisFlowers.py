import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
#load dataset
url ="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length','sepal-width','petal length','petalwidth','class']
dataset = pd.read_csv(url,names=names)
#number of rows and columns
print(dataset.shape)
print(dataset.head(60))
print(dataset.describe())
print(dataset.groupby('class').size())
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey
=False)
plt.show()
dataset.hist()
plt.show()
scatter_matrix(dataset)
plt.show()
#split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train,X_validation,Y_train,Y_validation = train_test_split(X,y,test_size=0.20,random_state=1)

#spot check algo
'''models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))
#evaluate each model in turn
results=[]
names=[]
for name,model in models:
    kfold = StratifiedKFold(n_splits=10,random_state = 1,shuffle=True)
    #kf = KFold(n_splits=5)
    #for train_index,test_index in kf.split(X,y):
    #print(train_index,test_index)
    scores = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    #print(type(scores))
    #print(cv_results)
    results.append(scores)
    names.append(name)
    print('%s:%f(%f)'%(name,scores.mean(),scores.std()))
#compare algorithms
plt.boxplot(results,labels=names)
plt.title('Algorithm Comparison')
plt.show()
#make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions = model.predict(X_validation)
#evaluate predictions
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))'''

