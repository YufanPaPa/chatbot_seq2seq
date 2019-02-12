
# coding: utf-8

# In[1]:


import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
if '../../personalized-rinna/' not in sys.path:
    sys.path.insert(0, '../../personalized-rinna/')

import logging
import json
import re

import MeCab
import pandas as pd
from gensim import corpora
from tqdm import tqdm_notebook as tqdm

from persona.data.preprocessing.normalizing import normalize_neologd


# In[2]:


get_ipython().run_cell_magic('time', '', "with open('../jiren2_2018-09-14.txt') as f:\n    data = [line.split('\\t') for line in f]")


# In[3]:


mt = MeCab.Tagger(r'-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
# Prevent utf-8 codec error
mt.parse('')

def tokenize(text):
    # remove white spaces
    regex = re.compile(r"\s+")
    text = regex.sub("", text)
    text = normalize_neologd(text)
    # remove trailing new line
    text = mt.parse(text).strip()
    return text


# In[4]:


user_desc = pd.read_csv('../../twitter-user-selection/selected_user_id_desc_noun_score.csv', sep='\t')


# In[5]:


user_desc = user_desc.set_index('id')
id_to_desc = user_desc['description']


# In[6]:


def data_transform(line):
    instance = {}
    user_id, context, query, response = line
    # '<context_end>' indicate the boundary between context and query
    # remove possible leading white space
    instance['title'] = ' '.join([tokenize(context), '<context_end>', tokenize(query)]).strip()
    desc = id_to_desc[int(line[0])]
    instance['content'] = tokenize(desc)
    # add dummy scores/votes to be compatible with existing code
    instance['comment'] = [[tokenize(response), 0]]
    # use utf8
    return json.dumps(instance, ensure_ascii=False)


# In[7]:


with open('../data/twitter_train.data', 'w') as f:
    for line in tqdm(data):
        f.write(data_transform(line) + '\n')


# In[8]:


def sentence_iterator(path):
    with open(path) as f:
        for line in f:
            instance = json.loads(line)
            yield instance['title'].split(' ')
            yield instance['comment'][0][0].split(' ')


# In[9]:


logging.basicConfig(format='(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)


# In[10]:


dictionary = corpora.Dictionary(sentence_iterator('../data/twitter_train.data'),
                                prune_at=1e7)


# In[14]:


sorted_id_freq = sorted(dictionary.dfs.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


#dictionary.filter_extremes(no_below=0, no_above=1, keep_n=30_000)


# In[16]:


with open('../data/twitter_vocab.txt', 'w') as f:
    for id, freq in sorted_id_freq:
        word = dictionary[id]
        f.write(f'{word}\t{freq}\n')


# In[15]:


sorted_id_freq

