#!/usr/bin/env python
# coding: utf-8

# ### Import libraries 
# ### Extra library added are contractions and Sklearn

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import contractions
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# get_ipython().system(" pip install bs4 # in case you don't have it installed")

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## Read Data

# In[3]:


data = pd.read_csv('data.tsv',sep='\t',on_bad_lines='skip', header=0, quotechar='"', dtype='unicode')


# #### The data given is read with the help of pandas read_csv using the compression gzip as data is compressed. Further I have used '\t' as seperator to format data.

# ## Keep Reviews and Ratings

# #### As mentioned below cell gets only 'review_body' and 'star_rating' columns from the whole data. Both the columns are merged and stored further

# In[4]:


getData = data[['review_body','star_rating']]
getData


# #### Next the cell check for the NaN values for both the columns
# #### Review body has 400 NaN values and Star rating has 10 NaN values

# In[5]:


getData.isnull().sum()


# #### As the NaN values are less compared to the total dataset rows, I therefore dropped the NaN values from both the columns 

# In[6]:


getData=getData.dropna()


# #### After dropping the NaN values below cell confirms that there are no further NaN values in the data 

# In[7]:


getData.isnull().sum()


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# #### label_class funtion helps to to add designated class to star_rating. In this a extra column is created with name class which stores the value of class with respect to star_rating
# #### Star_rating: 1, 2      Class: 1
# #### Star_rating: 3          Class: 2
# #### Star_rating: 4, 5      Class: 3

# In[8]:


pd.options.mode.chained_assignment = None 
def labelClass(rating):
    if rating == "1" :
          return 1
    if rating == "2" :
          return 1
    if rating == "3" :
          return 2
    if rating == "4":
          return 3
    if rating  == "5":
          return 3
getData['class'] = getData['star_rating'].map(labelClass)


# #### Ramdom 20000 data rows from all three columns(review_body, star_rating, class) gets selected 

# In[9]:


classOne = getData.loc[getData['class'] == 1].sample(n=20000)
classTwo = getData.loc[getData['class'] == 2].sample(n=20000)
classThree = getData.loc[getData['class'] == 3].sample(n=20000)


# #### The 20000 randomly slected data from all the three classes are further merged into one data providing the total rows of 60000

# In[10]:


getRandomData = pd.concat([classOne, classTwo, classThree])
getRandomData


# In[11]:


getRandomData['review_body'] = getRandomData['review_body'].astype(str)


# # Data Cleaning
# 
# 

# #### average length of review's are recorded before data cleaning

# In[12]:


avgLenBeforeClean = getRandomData['review_body'].apply(len).mean()


# #### various data cleaning strategies are are applied to clean and process the review data
# #### 1) all the reviews are converted to lower case
# #### 2) HTML and URL's are removed from the reviews
# #### 3) Last contractions are performed on the review's this will change won’t → will not, I'm → i am and so on with the use of contraction library
# #### 4) Extra spaces are removed
# #### 5) Non alphabetical characters are also removed

# In[13]:


getRandomData['review_body'] = getRandomData['review_body'].apply(str.lower)
getRandomData['review_body'] = getRandomData['review_body'].apply(lambda x: re.sub(re.compile('<.*?>'), " ", x))
getRandomData['review_body'] = getRandomData['review_body'].apply(lambda x: re.sub(re.compile('http\S+|https\S+')," ", x))
getRandomData['review_body'] = getRandomData['review_body'].apply(lambda x: re.sub(' +',' ', x))


# In[14]:


def contra(text):
    words = []   
    for word in text.split():
        words.append(contractions.fix(word))  
    word_text = ' '.join(words)
    return word_text
getRandomData['review_body'] = getRandomData['review_body'].apply(contra)


# In[15]:


getRandomData['review_body'] = getRandomData['review_body'].apply(lambda x: re.sub(re.compile("[^A-Za-z]")," ",x))


# In[16]:


avgLenAfterClean = getRandomData['review_body'].apply(len).mean()


# In[17]:


print('Average length of review_body before and after data cleaning is '+ str(avgLenBeforeClean) + ', '+ str(avgLenAfterClean))


# # Pre-processing

# ## remove the stop words 

# #### average length of review's are recorded before data pre-processing

# In[18]:


avgLenBeforePre = getRandomData['review_body'].apply(len).mean()


# #### Stop words are the commonly used words which does not create much impact in a sentance and ignoring such words will ease in creating the model for prediction
# #### below cell removes the stop words from the review_body column by using the nltk.corpus library

# In[19]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
getRandomData['review_body'] = getRandomData['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# ## perform lemmatization  

# #### lemmatization can be said as the step done to group toegther same type of words, to remove inflectional words and to return the base word
# #### below cell does the lemmatization by using the nltk.stem library and importing WordNetLemmatizer

# In[20]:


from nltk.stem import WordNetLemmatizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

getRandomData['review_body'] = getRandomData['review_body'].apply(lemmatize_text)


# In[21]:


avgLenAfterPre = getRandomData['review_body'].apply(len).mean()


# In[22]:


print('Average length of review_body before and after pre-processing is '+ str(avgLenBeforePre) + ', ' + str(avgLenAfterPre))


# # TF-IDF Feature Extraction

# #### Term Frequency – Inverse Document Frequency, this shows that how much important a word or phrase to a given document. We can use this technique when we have a Bag of words and we need to extract some information.

# In[23]:


X = getRandomData['review_body']
y = getRandomData['class']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


# #### After TF-IDF Feature Extraction final data is splitted with 20% test size and 80% train size with the help of train_test_split using random_state = 10 

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)


# # Perceptron

# #### Grid search cross validation is used to further train the perceptron model on various parameter of max_iter, alpha, penalty and eta0
# #### This will help to get the exact value of parameters which can be used with the final perceptron model. This will help to achieve higher accuracy. 

# In[26]:


params = {'max_iter': [50, 100, 500, 1000], 'penalty': ['l2', 'l1', 'elasticnet'], 'eta0': [0.0001, 0.001, 0.01, 0.1,1.0,1.5]}
gridPermodel = GridSearchCV(Perceptron(), params, cv=5).fit(X_train, y_train)
perceptronModel = Perceptron(max_iter = gridPermodel.best_params_['max_iter'], eta0 = gridPermodel.best_params_['eta0'], penalty = gridPermodel.best_params_['penalty'], random_state = 42).fit(X_train, y_train)
y_predict = perceptronModel.predict(X_test)


# #### Generating classification report for perceptron model and printing precision, recall and f1-score for all the three classes.

# In[27]:


perceptronCR = classification_report(y_predict,y_test, output_dict=True)
# print(classification_report(y_predict,y_test))


# In[27]:

print('classification report of Perceptron')
print('Class        Precision            Recall             F1-score')
print('1'+"      "+ str(perceptronCR['1']['precision'])+",   "+str(perceptronCR['1']['recall'])+",   "+ str(perceptronCR['1']['f1-score']))
print('2'+"      "+ str(perceptronCR['2']['precision'])+",   "+str(perceptronCR['2']['recall'])+",   "+ str(perceptronCR['2']['f1-score']))
print('3'+"      "+ str(perceptronCR['3']['precision'])+",   "+str(perceptronCR['3']['recall'])+",   "+ str(perceptronCR['3']['f1-score']))
print('average' +" "+str((perceptronCR['1']['precision']+perceptronCR['2']['precision']+perceptronCR['3']['precision'])/3)+", "+str((perceptronCR['1']['recall']+perceptronCR['2']['recall']+perceptronCR['3']['recall'])/3)+", "+str((perceptronCR['1']['f1-score']+perceptronCR['2']['f1-score']+perceptronCR['3']['f1-score'])/3))


# # SVM

# #### Grid search cross validation is used to further train the SVM model on parameter of C
# #### This will help to get the exact value of C which can be used with the final SVM model. This will help to achieve higher accuracy. 

# In[28]:


params = {"C": [0.01,0.1,1,10,100,1000,10000]}
gridSVMmodel = GridSearchCV(LinearSVC(), params, cv=5).fit(X_train, y_train)
svmModel = LinearSVC(C = gridSVMmodel.best_params_['C']).fit(X_train, y_train)
svmModel = LinearSVC().fit(X_train, y_train)
y_predict = svmModel.predict(X_test)


# #### Generating classification report for SVM model and printing precision, recall and f1-score for all the three classes.

# In[29]:


svmCR = classification_report(y_predict,y_test, output_dict=True)
# print(classification_report(y_predict,y_test))


# In[30]:

print('classification report of SVM')
print('Class        Precision            Recall             F1-score')
print('1'+"      "+ str(svmCR['1']['precision'])+",   "+str(svmCR['1']['recall'])+",   "+ str(svmCR['1']['f1-score']))
print('2'+"      "+ str(svmCR['2']['precision'])+",   "+str(svmCR['2']['recall'])+",   "+ str(svmCR['2']['f1-score']))
print('3'+"      "+ str(svmCR['3']['precision'])+",   "+str(svmCR['3']['recall'])+",   "+ str(svmCR['3']['f1-score']))
print('average' +" "+str((svmCR['1']['precision']+svmCR['2']['precision']+svmCR['3']['precision'])/3)+", "+str((svmCR['1']['recall']+svmCR['2']['recall']+svmCR['3']['recall'])/3)+", "+str((svmCR['1']['f1-score']+svmCR['2']['f1-score']+svmCR['3']['f1-score'])/3))


# # Logistic Regression

# #### Grid search cross validation is used to further train the logistic regression model on parameter of penalty
# #### This will help to get the exact value of parameter penalty which can be used with the final logistic regression model. This will help to achieve higher accuracy. 

# In[31]:


params = {"penalty": ['l1','l2','elasticnet',None]}
gridLRmodel = GridSearchCV(LogisticRegression(), params, cv=5).fit(X_train, y_train)
logRegModel = LogisticRegression(penalty = gridLRmodel.best_params_['penalty'], solver='lbfgs', max_iter=3000).fit(X_train, y_train)
y_predict = logRegModel.predict(X_test)


# #### Generating classification report for logistic regression model and printing precision, recall and f1-score for all the three classes.

# In[33]:


logRegCR = classification_report(y_predict,y_test, output_dict=True)
# print(classification_report(y_predict,y_test))


# In[34]:

print('classification report of Logistic Regression')
print('Class        Precision            Recall             F1-score')
print('1'+"      "+ str(logRegCR['1']['precision'])+",   "+str(logRegCR['1']['recall'])+",   "+ str(logRegCR['1']['f1-score']))
print('2'+"      "+ str(logRegCR['2']['precision'])+",   "+str(logRegCR['2']['recall'])+",   "+ str(logRegCR['2']['f1-score']))
print('3'+"      "+ str(logRegCR['3']['precision'])+",   "+str(logRegCR['3']['recall'])+",   "+ str(logRegCR['3']['f1-score']))
print('average' +" "+str((logRegCR['1']['precision']+logRegCR['2']['precision']+logRegCR['3']['precision'])/3)+", "+str((logRegCR['1']['recall']+logRegCR['2']['recall']+logRegCR['3']['recall'])/3)+", "+str((logRegCR['1']['f1-score']+logRegCR['2']['f1-score']+logRegCR['3']['f1-score'])/3))


# # Naive Bayes

# #### Grid search cross validation is used to further train the naive bayes model on various parameter of alpha
# #### This will help to get the exact value of parameter alpha which can be used with the final naive bayes model. This will help to achieve higher accuracy. 

# In[35]:


params = {'alpha': [0.01,0.1,0,1.0,10.0]}
gridNBmodel = GridSearchCV(MultinomialNB(), params, cv=5).fit(X_train, y_train)
naiBayModel = MultinomialNB(alpha = gridNBmodel.best_params_['alpha']).fit(X_train, y_train)
y_predict = naiBayModel.predict(X_test)


# #### Generating classification report for naive bayes model and printing precision, recall and f1-score for all the three classes.

# In[36]:


naiBayCR = classification_report(y_predict,y_test, output_dict=True)
# print(classification_report(y_predict,y_test))


# In[37]:

print('classification report of Naive Bayes')
print('Class        Precision            Recall             F1-score')
print('1'+"      "+ str(naiBayCR['1']['precision'])+",   "+str(naiBayCR['1']['recall'])+",   "+ str(naiBayCR['1']['f1-score']))
print('2'+"      "+ str(naiBayCR['2']['precision'])+",   "+str(naiBayCR['2']['recall'])+",   "+ str(naiBayCR['2']['f1-score']))
print('3'+"      "+ str(naiBayCR['3']['precision'])+",   "+str(naiBayCR['3']['recall'])+",   "+ str(naiBayCR['3']['f1-score']))
print('average' +" "+str((naiBayCR['1']['precision']+naiBayCR['2']['precision']+naiBayCR['3']['precision'])/3)+", "+str((naiBayCR['1']['recall']+naiBayCR['2']['recall']+naiBayCR['3']['recall'])/3)+", "+str((naiBayCR['1']['f1-score']+naiBayCR['2']['f1-score']+naiBayCR['3']['f1-score'])/3))


# In[ ]:




