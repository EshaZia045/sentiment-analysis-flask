import pandas as pd

from google.colab import drive
drive.mount("/content/drive", force_remount=True)
x_train = pd.read_csv("/content/drive/MyDrive/Train.csv")

import re
def supprimer_crochets(text):
  return re.sub('\[[^]]*\]', '', text)
x_train['text']=x_train['text'].apply(supprimer_crochets)
def supprimer_special(text, remove_digits=True):
 pattern=r'[^a-zA-z0-9\s]'
 text=re.sub(pattern,'',text)
 return text

import nltk
nltk.download(["names","stopwords","punkt"])

#Stemming the text
def simple_stemmer(text):
 ps=nltk.porter.PorterStemmer()
 text= ' '.join([ps.stem(word) for word in text.split()])
 return text

#Apply function on review column
x_train['text']=x_train['text'].apply(simple_stemmer)

from nltk.tokenize.toktok import ToktokTokenizer
tokenizer=ToktokTokenizer()
stopword_list=nltk.corpus.stopwords.words('english')

from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
print(stop)

def remove_stopwords(text):
 tokens = tokenizer.tokenize(text)
 tokens = [token.strip() for token in tokens]
 filtered_tokens = [token for token in tokens if token not in stopword_list]
 filtered_text = ' '.join(filtered_tokens)
 return filtered_text
x_train['text']=x_train['text'].apply(remove_stopwords)

#30000 commentaires pour l'entrainement
norm_train_text=x_train.text[:30000]
norm_train_text[0]

#10000 commentaires pour le test
norm_test_text=x_train.text[30000:]
norm_test_text[35005]

from sklearn.feature_extraction.text import TfidfVectorizer
#Tfidf vectorizer
tv=TfidfVectorizer()
#transformed train reviews
tv_train_text=tv.fit_transform(norm_train_text)
#transformed test reviews
tv_test_text=tv.transform(norm_test_text)
print('Tfidf_train:',tv_train_text.shape)
print('Tfidf_test:',tv_test_text.shape)

sentiment_data = x_train['label']

#Spliting the sentiment data (labels)
train_sentiments=sentiment_data[:30000]
test_sentiments=sentiment_data[30000:]
print(train_sentiments)
print(test_sentiments)

from sklearn.linear_model import LogisticRegression
#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_text,train_sentiments)
print(lr_tfidf)

#Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(tv_test_text)
print(lr_tfidf_predict)

from sklearn.metrics import confusion_matrix,accuracy_score
lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)

confusion_matrix(test_sentiments,lr_tfidf_predict)

#one comment prediction
liste = []
commentaire = "hello it is a bad movie"
commentaire = supprimer_crochets(commentaire)
commentaire = supprimer_special(commentaire)
commentaire = simple_stemmer(commentaire)
tokens = remove_stopwords(commentaire)
liste.append(tokens)
x = tv.transform(liste)
res = lr.predict(x)
print(res)