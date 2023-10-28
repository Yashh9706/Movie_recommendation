#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movie=pd.read_csv('tmdb_5000_movies.csv')
credit=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movie.head()


# In[4]:


credit.head()


# In[5]:


movie=movie.merge(credit,on='title')


# In[6]:


movie.info()


# In[7]:


movie=movie[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


movie.isnull().sum()


# In[9]:


movie.dropna(inplace=True)


# In[10]:


movie.duplicated().sum()


# In[11]:


import yaml


# In[12]:


def convert(obj):
    l=[]
    for i in yaml.safe_load(obj):
        l.append(i['name'])
    return l


# In[13]:


movie['genres']=movie['genres'].apply(convert)


# In[14]:


movie


# In[15]:


movie['keywords']=movie['keywords'].apply(convert)


# In[16]:


def top_3_cast(obj):
    l=[]
    counter=0
    for i in yaml.safe_load(obj):
        if counter!=3:
            l.append(i['name'])
            counter+=1
        else:
            break
    return l


# In[17]:


movie['cast']=movie['cast'].apply(top_3_cast)


# In[18]:


movie['cast'][0]


# In[19]:


movie


# In[20]:


def fetch_director(obj):
    l=[]
    for i in yaml.safe_load(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l


# In[21]:


movie['crew']=movie['crew'].apply(fetch_director)


# In[22]:


movie


# In[23]:


movie['crew'][0]


# In[24]:


movie['overview'][0]


# In[25]:


movie['overview'] = movie['overview'].apply(str.split)


# In[26]:


movie


# In[27]:


movie['genres']=movie['genres'].apply(lambda x:[i.replace(' ','')for i in x])
movie['keywords']=movie['keywords'].apply(lambda x:[i.replace(' ','')for i in x])
movie['cast']=movie['cast'].apply(lambda x:[i.replace(' ','')for i in x])
movie['crew']=movie['crew'].apply(lambda x:[i.replace(' ','')for i in x])


# In[28]:


movie


# In[29]:


movie['tags']=movie['overview']+movie['genres']+movie['keywords']+movie['cast']+movie['crew']


# In[30]:


movie['tags']


# In[31]:


movies=movie[['movie_id','title','tags']]


# In[32]:


movies


# In[33]:


movies['tags']=movies['tags'].apply(lambda x:' '.join(x))


# In[34]:


movies.head()


# In[35]:


movies['tags'][0]


# In[36]:


movies['tags'] = movies['tags'].str.lower()


# In[37]:


movies['tags']


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer


# In[39]:


cv=CountVectorizer(max_features=5000,stop_words='english')


# In[40]:


vectors=cv.fit_transform(movies['tags'])


# In[41]:


cv.get_feature_names_out()


# In[42]:


pip install nltk


# In[43]:


import nltk


# In[44]:


from nltk.stem.porter import PorterStemmer


# In[45]:


ps=PorterStemmer()


# In[46]:


def stem(text):
    return [ps.stem(word) for word in text.split()]


# In[47]:


movies['tags']=movies['tags'].apply(stem)


# In[48]:


movies.head()


# In[49]:


from sklearn.metrics.pairwise import cosine_similarity


# In[50]:


similarity=cosine_similarity(vectors)


# In[51]:


similarity[1]


# In[52]:


def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    distaces=similarity[movie_index]
    recommended_movies=sorted(enumerate(distaces),key=lambda x: x[1],reverse=True)[:5]

    
    for i in recommended_movies:
        print(movies['title'].values[i[0]])

    return


# In[53]:


recommend("The Dark Knight Rises")


# In[54]:


recommend("Pirates of the Caribbean: At World's End")


# In[ ]:




