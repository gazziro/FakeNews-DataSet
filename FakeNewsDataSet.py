#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


# In[91]:


df = pd.read_csv(R"C:\Users\Gabriel\Documents\news.csv")

###########################################
# 0 - Fake
# 1 - Real


# In[92]:


df


# In[5]:


le = LabelEncoder()


# In[9]:


df['label'] = le.fit_transform(df.label)


# In[10]:


df


# In[93]:


df['label'].value_counts().plot.bar(figsize=(10,5), grid=True, rot=0)


# In[94]:


df.shape


# In[95]:


labels=df.label


# In[96]:


labels.head()


# In[97]:


x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.3)


# In[99]:


vetorizador = TfidfVectorizer(stop_words = 'english', max_df=0.8)


# In[100]:


vetor_train = vetorizador.fit_transform(x_train)


# In[101]:


vetor_test = vetorizador.transform(x_test)


# In[103]:


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(vetor_train, y_train)


# In[105]:


y_pred = pac.predict(vetor_test)
pontuacao = accuracy_score(y_test, y_pred)
print("Pontuação: ",pontuacao )


# In[106]:


confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])


# In[108]:


print(classification_report(y_test, y_pred))


# In[ ]:




