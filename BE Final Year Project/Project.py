# -*- coding: utf-8 -*-

from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer
import csv
import nltk
#nltk.download('all')
import pandas as pd
import numpy 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import csv

# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tweepy

# Define Flask app
app = Flask(__name__)

#Data cleaning Functions:
def isEnglish(s):
    try:
        s.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True

    #The following function removes the part of the string that contains the substring eg. if
    #substring = 'http' , then http://www.google.com is removed, that means, remove until a space is found
def rem_substring(tweets,substring):
    m=0;
    for i in tweets:
        if (substring in i):
        #while i.find(substring)!=-1:
            k=i.find(substring)
            d=i.find(' ',k,len(i))
            if d!=-1:               #substring is present somwhere in the middle(not the end of the string)
                i=i[:k]+i[d:]
            else:                   #special case when the substring is present at the end, we needn't append the
                i=i[:k]             #substring after the junk string to our result
        tweets[m]=i #store the result in tweets "list"
        m+= 1
    return tweets
def removeNonEnglish(tweets):
    result=[]
    for i in tweets:
        if isEnglish(i):
            result.append(i)
    return result

#the following function converts all the text to the lower case
def lower_case(tweets):
    result=[]
    for i in tweets:
        result.append(i.lower())
    return result

def rem_punctuation(tweets):
    #print(len(tweets))
    m=0
    validLetters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    for i in tweets:
        x = ""
        for j in i:
            if (j in validLetters)==True:
                x += j
        tweets[m]=x
        m=m+1
    return tweets

def stop_words(tweets):
    #Removal of Stop words like is, am , be, are, was etc.
    stop_words1 = set(stopwords.words('english')) 
    indi=0
    for tweet in tweets:
        new_s=[]
        Br_tweet = word_tokenize(tweet)
        for word in Br_tweet:
            if (word not in stop_words1):
                new_s.append(word)
        et=" ".join(new_s)
        tweets[indi]=et
        indi+=1
    return tweets
        
                

def score(college_name):
    filename = 'data_emotions_words_list.csv'
    pos_file_name= "Pos_tagged_" + college_name + ".csv"
    POS=pd.read_csv(pos_file_name)
    POS_tweets=POS['POS_Tweet'].values
    adverb1=pd.read_csv("adverb.csv")
    verb1=pd.read_csv("verb.csv")
    
    ''' Verb and adverb are dictionaries having values for verbs and adverbs'''
    verb={};adverb={}
    l=adverb1['value'].values
    j=0
    for i in adverb1['adverb'].values:
        adverb[i]=l[j]
        j+=1
    l=verb1['Value'].values
    j=0
    for i in verb1['Verb'].values:
        verb[i]=l[j]
        j+=1
    
    ''' Add the adjectives in the dictionary'''
    
    Adjectives={}
    df=pd.read_csv("data_emotions_words_list.csv")
    for i in range(len(df)) : 
        Adjectives[df.loc[i, "Words"]]= [df.loc[i, "Happiness"],df.loc[i, "Anger"],df.loc[i, "Sadness"],df.loc[i, "Fear"],df.loc[i, "Disgust"]] 

    ''' Assign Scores to each tweet'''
    FINAL={};FINAL1={'Tweets':[],'Happiness':[],'Sadness':[],'Fear':[],'Disgust':[],'Anger':[],'Sentiment':[]}
    for tweet in POS_tweets:
        sum_adverb=0;sum_verb=0
        score_list=[]
        words=word_tokenize(tweet)
        stem=stemming(words)
        f_stem=0
        for i in words :
            if (i in adverb):
                sum_adverb+=adverb[i]
    
                
            elif (stem[f_stem] in adverb):
                sum_adverb+=adverb[stem[f_stem]]
               
                
            elif (i in verb):
                sum_verb+=verb[i]
                
                
            elif (stem[f_stem] in verb):
                sum_verb+=verb[stem[f_stem]]
            else:
                if (i in Adjectives ) or (stem[f_stem] in Adjectives):
                    if i in Adjectives:
                        # ADJ=[Happiness,Anger,Sadness,Fear,disgust]
                        ADJ=Adjectives[i]
                    elif (stem[f_stem] in Adjectives):
                        ADJ=Adjectives[stem[f_stem]]
                    else:
                        pass
                    
                    # Calculate Score
                    c=sum_adverb + sum_verb
                    #The formula is derived from the research paper
                    if (c) <0 :
                        for j in range(len(ADJ)):
                            ADJ[j]=5.0-ADJ[j]
                    elif (c>=0.5):
                        for j in range(len(ADJ)):
                            ADJ[j]=c*ADJ[j]
                    else:
                        for j in range(len(ADJ)):
                            ADJ[j]=0.5*ADJ[j]
                    score_list.append(ADJ)
                    sum_adverb=0;sum_verb=0
            f_stem+=1
        total_adj=len(score_list)
        s=[0.0 for i in range(5)]
        emo=''
        if (total_adj != 0):
            for i in score_list:
                s[0]+=i[0] #Happiness
                s[1]+=i[1]#Anger
                s[2]+=i[2] #Sadness
                s[3]+=i[3] #Fear
                s[4]+=i[4] #Disgust
            for i in range(len(s)):
                s[i]= "{0:.6f}".format(s[i]/total_adj)
            emotion=0.0
            for i in range(len(s)):
                if (float(s[i])> emotion):
                    emotion=max(emotion,float(s[i]))
                    if i==0 :
                        emo='Happiness'
                    elif i==1:
                        emo='Anger'
                    elif i==2:
                        emo='Sadness'
                    elif i==3:
                        emo='Fear'
                    elif i==4:
                        emo='Disgust'
                
        else:
            # if adj is not in vocabulary assign 
            s=[0.2000 for i in range(5)]
            emo='Neutral'
            
        #find the Max emotion value for the tweet
        s.append(emo)
            
        
        #Add the final tweet and score
        FINAL[tweet]=s
        FINAL1['Tweets'].append(tweet)
        FINAL1['Happiness'].append(s[0])
        FINAL1['Anger'].append(s[1])
        FINAL1['Fear'].append(s[3])
        FINAL1['Sadness'].append(s[2])
        FINAL1['Disgust'].append(s[4])
        FINAL1['Sentiment'].append(s[5])
    DB=pd.DataFrame(FINAL1,columns=['Tweets','Happiness','Anger','Fear','Sadness','Disgust','Sentiment'])
    file_name = "FINAL_" + college_name + "_SENTIMENTS.csv"
    DB.to_csv(file_name)
        

 #POS Tagger Function used to identify the adjectives, verbs, adverbs.

def POS_tagger(tweets, username):
    final = []
        # for each line in tweets list
    for line in tweets:
        t = []
            # for each sentence in the line
            # tokenize this sentence
        text= word_tokenize(line)
        k = nltk.pos_tag(text)
        for i in k:
                # Only Verbs, Adverbs & Adjectives are Considered
                if ((i[1][:2] == "VB") or (i[1][:2] == "JJ") or (i[1][:2] == "RB")):
                    t.append(i[0])
        one_tweet=" ".join(t)
        if (len(one_tweet)>0):
            final.append(one_tweet)
    
    dict1={'POS_Tweet':final}
    db1=pd.DataFrame(dict1)
    filename = "Pos_tagged_" + username + ".csv"
    
    db1.to_csv(filename)

def stemming(tweets):
    # Find the root word
    # stemming of words
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tweets]
    return stemmed

def print_score(clf,X_train,Y_train,X_test, Y_test,train=True):
    if train:
        print("\nTraining Result:")
        print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_train,clf.predict(X_train))))
        print("\nClassification_report: \n{}\n".format(classification_report(Y_train,clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(Y_train,clf.predict(X_train))))
        
        res=cross_val_score(clf,X_train,Y_train,cv=10,scoring="accuracy")
        print("Average Accuracy:\t {0:.4f}".format(np.mean(res)))
        print ("Accuracy SD:\t\t {0:.4f}".format(np.std(res)))
        return "{0:.4f}".format(accuracy_score(Y_train,clf.predict(X_train)))
    
    elif train==False:
        print("\nTest Results:")
        print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_test,clf.predict(X_test))))
        print("\nClassification_report: \n{}\n".format(classification_report(Y_test,clf.predict(X_test))))
        print("\nConfusion Matrix:\n{}\n".format(confusion_matrix(Y_test,clf.predict(X_test))))
        return "{0:.4f}".format(accuracy_score(Y_test,clf.predict(X_test)))

def main():
    c= input("Enter the name of the raw_tweets college:")
    c_f=c+'_tweets.csv'
    db=pd.read_csv(c_f)
    tweets=list(db['text'])
    tweets=rem_substring(tweets,'#')
    tweets=rem_substring(tweets,'http')
    tweets=rem_substring(tweets,'https')
    tweets=rem_substring(tweets,'www')
    tweets=rem_substring(tweets,'@')
    tweets=rem_substring(tweets,'RT')
    
    tweets=rem_punctuation(tweets)
    tweets=stop_words(tweets)
    tweets= removeNonEnglish(tweets)
   
    tweets=lower_case(tweets)
    
    #tweets = stemming(tweets)
   
    
    #tweets.replace("."," ")
    for tweet in tweets:
        tweet=tweet.replace("."," ")

    
    dict1={'Tweet':tweets}
    db1=pd.DataFrame(dict1)
    r_f='cleaned_'+ c + '.csv'
    db1.to_csv(r_f)
    
     
    POS_tagger(tweets,c)
    print("Tweets have now been cleaned !!")
    score(c)
    
    college_name= input("Enter the name of the college:")
    k = "FINAL_" + college_name + "_SENTIMENTS.csv"
    df=pd.read_csv(k)
    df.dropna(inplace=True)
    tweet=[ i for i in df['Tweets']]
    d1={'Happiness':1,'Neutral':0,'Anger':-2,'Fear':-3,'Disgust':-4,'Sadness':-1}
    df['Sentiment']=df['Sentiment'].map(d1)

    ''' Used for forming feature vectors through bag of words technique'''
    '''
    #y=[]
    for i in range(len(df)):
        l.append([df.loc[i,'Happiness'],df.loc[i,'Anger'],df.loc[i,'Fear'],df.loc[i,'Sadness'],df.loc[i,'Disgust']]) 
        #y.append(df.loc[i,'Sentiment'])
    '''
    cv = CountVectorizer(token_pattern='(?u)\\b\\w+\\b',max_features = 2000)
    X = cv.fit_transform(tweet).toarray()
    y=df.loc[:,'Sentiment'].values
    X_train,X_test,Y_train,Y_test= train_test_split(X,y,train_size=0.7,random_state=42)

    scoring_r=[]
    scoring_t=[]
    '''
    with open("SCORES_Test" +".csv", 'w+') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(("College","KNN", "SVM", "Naive_Bayes"))

    with open("SCORES_Training"+".csv", 'w+') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(("College","KNN", "SVM", "Naive_Bayes"))
    '''
        


    """KNN"""
    knn= KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
    knn.fit(X_train,Y_train)
    print ("\n \t\t------KNN Classifier----\n")       
    #Scores for training data
    v=print_score(knn,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    t=print_score(knn,X_train,Y_train,X_test, Y_test,train=False)
    scoring_r.append(v)
    scoring_t.append(t)

    '''SVM'''
    clf=svm.SVC(kernel='rbf', degree=3,  gamma=0.7)
    clf.fit(X_train,Y_train)

    print("\n\n\t\t----- SVM Details-------\n\n")
    #Scores for training data
    v=print_score(clf,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    t=print_score(clf,X_train,Y_train,X_test, Y_test,train=False)
    scoring_r.append(v)
    scoring_t.append(t)

    """ Naive Bayes"""
    model = GaussianNB()
    model.fit(X_train,Y_train)
    print("\n\n Naive Bayesian\n\n")
    #Scores for training data
    v=print_score(model,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    t=print_score(model,X_train,Y_train,X_test, Y_test,train=False)
    scoring_r.append(v)
    scoring_t.append(t)

    ''' Append the accuracy to the scores file for further analysis''' 
    with open("SCORES_Training" +".csv", 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow((college_name,scoring_r[0], scoring_r[1],scoring_r[2]))

    with open("SCORES_Test" +".csv", 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow((college_name,scoring_t[0], scoring_t[1],scoring_t[2]))
        
    # Read the datafile and plot 
    """ For different colleges different ML techniques perform differently
    dg=pd.read_csv("SCORES_Training" +".csv")
    dg.plot(x="College", y=["KNN","SVM","Naive_Bayes"], kind="bar",figsize=(8,5))

    dg=pd.read_csv("SCORES_Test" +".csv")
    dg.plot(x="College", y=["KNN","SVM","Naive_Bayes"], kind="bar",figsize=(8,5))


    """

"""main()"""

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = sentiment_analysis(text)
    return render_template('index.html', prediction=prediction)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)


