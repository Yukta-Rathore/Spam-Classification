import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tkinter import *
import string

df=pd.read_csv('spam.csv',encoding='latin-1')
df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.rename(columns={'v1':'labels','v2':'message'},inplace=True)
df.drop_duplicates(inplace=True)
df['label']=df['labels'].map({'ham':0,'spam':1})
df.drop(['labels'],axis=1,inplace=True)

custom_stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
])

def preprocess_text(message):
    without_punc=[char for char in message if char not in string.punctuation]
    without_punc=''.join(without_punc)
    return [word for word in without_punc.split() if word.lower() not in custom_stopwords]


from sklearn.feature_extraction.text import CountVectorizer
x=df['message']
y=df['label']
cv=CountVectorizer()
x=cv.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB().fit(x_train,y_train)


def sms():
    if hasattr(sms, 'classification') and sms.classification.winfo_exists():
        sms.classification.destroy()
    classes=['not spam','spam']
    x=cv.transform([e.get()]).toarray()
    p=classifier.predict(x)
    s=[str(i) for i in p]
    a=int("".join(s))
    res=str("This is "+classes[a])
    print(res) 
    if classes[a]=='spam':
        sms.classification=Label(root,text=res,font=('helvetica',15,'bold'),fg='red')
        sms.classification.pack()
    else:
        sms.classification=Label(root,text=res,font=('helvetica',15,'bold'),fg='green')
        sms.classification.pack()


root=Tk()
root.title('Spam Checker')
root.geometry('400x400')

head=Label(root,text='SPAM Checker',font=('helvetica',24,'bold'))
head.pack()
e=Entry(root,width=400,borderwidth=5)
e.pack()
b=Button(root,text='Check',font=('helvetica',20,'bold'),fg='white',bg='green',command=sms)
b.pack()
root.mainloop()
        