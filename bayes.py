
# coding: utf-8

# In[6]:

#!/usr/bin/env python


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time
from math import *
from matplotlib import pyplot as pl
#from matplotlib.backends.backend_pdf import PdfPages

class MyBayesClassifier():
    # For graduate and undergraduate students to implement Bernoulli Bayes
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    def train(self, X, y):
        print ("Training bernoulli NB")
# Temp storage dictionaries
        count_each_class = {}
        feature_count = {}
        alpha = self._smooth # smooth factor
        temp=[]
        temp.append(np.unique(y))
        self._Ncls.append(temp[0].size) #total number of classes
        self._Nfeat.append(X[0].size)  # total number of features
        
        for i in range(y.size):
            if y[i] in feature_count:
                continue
            else:
                feature_count[y[i]] = [0 for w in range (X[i].size)]
                
#count the features per each class across train, count occurance of each class across train
        for i in range (y.size):
            if y[i] in count_each_class:
                count_each_class[y[i]] +=1
            else:
                count_each_class[y[i]] = 1
            for j in  range(X[i].size):
                    feature_count[y[i]][j] += X[i][j]
                    
# Calculate class and feature probablities per each class       
        for cls in feature_count:
            
            num = (self._smooth+count_each_class[cls])
            din = (y.size+(self._Ncls[0]*self._smooth))
            self._class_prob.append((num/float(din)))
            ar = np.array([])
            for j in  range(X[i].size):
                
                num= (feature_count[cls][j] + self._smooth)
                din = (count_each_class[cls]+(2*self._smooth))
                ar=np.append(ar,(num/float(din)))
            self._feat_prob.append(ar)
    

    def predict(self, X):
        
        print ("Predicting Bernoulli NB")
        
        Y_predict = np.array([])

        for i in X:
            neg_log_prob = 0
            minimum_neg_log_prob=999999999999999
            category = 0  
                
            for cls in range(self._Ncls[0]):
                neg_log_prob = -log(self._class_prob[cls])
                for j in  range(self._Nfeat[0]):  
                    if (i[j])==0:
                        neg_log_prob -= log(1-self._feat_prob[cls][j])
                    else:
                        neg_log_prob -= log(self._feat_prob[cls][j])
                        
                if minimum_neg_log_prob>neg_log_prob:
                    category=cls
                    minimum_neg_log_prob=neg_log_prob
            
            Y_predict=np.append(Y_predict,category)
         
        return Y_predict

class MyMultinomialBayesClassifier():
    
    # For graduate students only
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._class_neg_prob = []
        self._Ncls = []
        self._Nfeat = []

    # Train the classifier using features in X and class labels in Y
    def train(self, X, y):
        print ("Training Multinomial NB")
        
# Temp storage dictionaries
        count_each_class = {}
        feature_count = {}
      
        for i in range(y.size):
            if y[i] in feature_count:
                continue
            else:
                feature_count[y[i]] = [0 for w in range (X[i].size)]
                
#count the features per each class across train, count occurance of each class across train
        for i in range (y.size):
            if y[i] in count_each_class:
                count_each_class[y[i]] +=1
            else:
                count_each_class[y[i]] = 1
            for j in  range(X[i].size):
                    feature_count[y[i]][j] += X[i][j]
                
        alpha = self._smooth # smooth factor
        temp=[]
        temp.append(np.unique(y))
        self._Ncls.append(temp[0].size) #total number of classes
        self._Nfeat.append(X[0].size)  # total number of features
        self._class_prob.append(count_each_class)
        self._feat_prob.append(feature_count)
        
        
    

    # should return an array of predictions, one for each row in X
    def predict(self, X):
        
        print ("Predicting Multinomial NB")
        
        Y_predict = np.array([])
        #calculate total class for train data:-----------
        total_train_count = 0
        for key in self._class_prob[0]:
            total_train_count += self._class_prob[0][key]
        #-----------------------------------------------
        
        for i in X:
            neg_log_prob = 0
            minimum_neg_log_prob=999999999999999
            category = 0
            
            for cls in self._feat_prob[0]:
                Ny = sum(self._feat_prob[0][cls])
                neg_log_prob = -log((self._class_prob[0][cls]+1)/float(total_train_count+(self._Ncls[0]*self._smooth)))
                for j in  range(self._Nfeat[0]):  
                # For multinomial we dont consider value = 0, so we continue iteration to save Computation time
                    if (i[j])==0:
                        continue    
                    for itere in range (i[j]):
                        num = (self._smooth+self._feat_prob[0][cls][j])
                        din = (Ny+(self._Nfeat[0]*self._smooth))
                        neg_log_prob -= log(num/float(din))
                        
                if minimum_neg_log_prob>neg_log_prob:
                    category=cls
                    minimum_neg_log_prob=neg_log_prob
            
            Y_predict=np.append(Y_predict,category)
         
        return Y_predict
        
""" 
Here is the calling code

"""

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a count vectorizer")
t0 = time()

# Binary = true for Bernoulli NB

vectorizer = CountVectorizer(stop_words='english', binary=True)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

# For Bernoulli NB, Binary = true, train for one default smooth value alpha = 1

print ('-------------------------------------------------------------')
print ('Expected time for execution for Bernoulli NB is 200 secs')
ta = time()
alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)
tb = time()
print ("For Bernoulli NB:  " +'alpha=%f ,accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0)))
print ("Time taken for train and predict of bernoulli:" + str(tb-ta))
print ('-------------------------------------------------------------')

"""
# For bernoulli Naive bayes Alpha Vs accuracy Code, Commented to prevent long computation. 
acc = []
alp = []

for alpha in [float(j) / 100 for j in range(1, 101, 1)]:
    print ('-----------------------------------------------------------------------------------------------------')
    ta = time()
    clf =MyBayesClassifier(alpha)
    clf.train(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc.append(np.mean((y_test-y_pred)==0))
    alp.append(alpha)
    tb = time()
    print ("training time: " + str(tb-ta) + " accuracy,alpha is: " + str(np.mean((y_test-y_pred)==0)) +","+str(alpha))

# Plotting for bernoulli Naive Bayes
# Alpha Vs Accuracy
# Alpha Vs Accuracy
with PdfPages('Bernoulli.pdf') as pdf:
    pl.plot(alp,acc,marker='.', linestyle='-', color='r')
    pl.ylabel('Accuracy',color='g')
    pl.xlabel('Alpha',color='g')
    pl.title('Alpha Vs accuracy plot for Bernoulli NB',color = 'r')
    pdf.savefig() 
    pl.close()

# Max Accuracy and Corresponding alpha.

print ("The maximum accuracy For the bernoulli NB is: " + str(max(acc)))
print ("and the corresponding value for alpha is:       " + str(alp[(acc.index(max(acc)))]))


#--------------- End of Bernoulli NB ---------------------------------#

"""
# Binary = false for Multinomial NB

print ("Extracting data with Binary =False for Multinomial NB")
vectorizer = CountVectorizer(stop_words='english', binary=False)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()



# For Multinomial NB, Binary = False, train for one default smooth value alpha = 1
print ('Expected time for execution for Multinomial NB is 130 secs')
ta = time()
alpha = 1
clf1 = MyMultinomialBayesClassifier(alpha)
clf1.train(X_train,y_train)
y_pred = clf1.predict(X_test)

print ("For Multinomial NB:  " +'alpha=%f accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0)))
tb= time()
print ("Time taken for train and predict of Multinomial:" + str(tb-ta))
print ('--------------------------------------------------------------------------------------')

""""
# For Multionomial Naive bayes Alpha Vs accuracy Code, Commented to prevent long computation. 
acc = []
alp = []

for alpha in [float(j) / 100 for j in range(1, 101, 1)]:
    print ('--------------------------------------------------------------------------------------')
    ta = time()
    clf1 =MyMultinomialBayesClassifier(alpha)
    clf1.train(X_train,y_train)
    y_pred1 = clf1.predict(X_test)
    acc.append(np.mean((y_test-y_pred1)==0))
    alp.append(alpha)
    tb = time()
    print ("training time: " + str(tb-ta) + " accuracy,alpha is: " + str(np.mean((y_test-y_pred1)==0)) +","+str(alpha))
    #print ('alpha=%f accuracy = %f' %(alpha, np.mean((y_test-y_pred1)==0)))

# Plotting for Multinomial Naive Bayes
# Alpha Vs Accuracy
with PdfPages('multinomial.pdf') as pdf:
    pl.plot(alp,acc,marker='.', linestyle='-', color='r')
    pl.ylabel('Accuracy',color='g')
    pl.xlabel('Alpha',color='g')
    pl.title('Alpha Vs accuracy plot for multinomial NB',color = 'r')
    pdf.savefig() 
    pl.close()

# Max Accuracy and Corresponding alpha.

print ("The maximum accuracy For the Multinomial NB is: " + str(max(acc)))
print ("and the corresponding value for alpha is:       " + str(alp[(acc.index(max(acc)))]))
"""
print(" ")


# In[ ]:




# In[ ]:



