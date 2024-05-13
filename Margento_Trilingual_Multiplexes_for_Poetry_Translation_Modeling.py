#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from os import listdir
from os.path import isfile, join
import nltk
import numpy as np
import re
import codecs
import string
from string import punctuation


# In[3]:


def read_and_sort_text_files(folder_path):
    files = []  # List to store the content of each text file
    file_names = []  # List to store the names of the text files
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                files.append(content)
                file_names.append(filename)
    
    # Sort file names based on the first four segments separated by underscore
    sorted_file_names = sorted(file_names, key=lambda x: x.split('_')[:4])

    # Sort files based on the sorted file names
    sorted_files = [files[file_names.index(name)] for name in sorted_file_names]

    return sorted_files, sorted_file_names


# In[4]:


pwd


# In[5]:


folder_path_0 = '/Users/Margento/fr_orig'


# In[6]:


documents_0, file_names_0 = read_and_sort_text_files(folder_path_0)


# In[7]:


file_names_0


# In[8]:


folder_path_1 = '/Users/Margento/en_yale_&_gpt'


# In[9]:


documents_1, file_names_1 = read_and_sort_text_files(folder_path_1)


# In[10]:


file_names_1


# In[165]:


file_names_1.index('2_Aventin_Parricide_I_trans_en_gpt.txt')


# In[11]:


folder_path_2 = '/Users/Margento/ro_dl_&_gpt'


# In[12]:


documents_2, file_names_2 = read_and_sort_text_files(folder_path_2)


# In[13]:


file_names_2


# In[34]:


documents_1[0]


# In[14]:


def preprocess_text(text):
    text = text.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    return text


# In[15]:


def read_and_preprocess_text_files(documents):
    preprocessed_documents = [] 
    
    for file in documents:
                preprocessed_content = preprocess_text(file)
                preprocessed_documents.append(preprocessed_content)
    
    return preprocessed_documents


# In[16]:


preprocessed_documents_0 = read_and_preprocess_text_files(documents_0)


# In[37]:


preprocessed_documents_0[0]


# In[17]:


preprocessed_documents_1 = read_and_preprocess_text_files(documents_1)


# In[18]:


preprocessed_documents_2 = read_and_preprocess_text_files(documents_2)


# In[19]:


stopwords_fr = nltk.corpus.stopwords.words('stopwords_French.txt')


# In[20]:


stopwords_fr.extend(['ainf', 'ains', 'ainsi', 'append', 'auec', 'autr', 'autre', 'ce', 'ces', 'cest', 'cet', 'cette', 'comme', 'cum2', 'doit', 'doivent', 'dont', 'dun', 'et', 'faire', 'fait', 'faut', 'fois', 'laquel', 'lautr', 'lon', 'multus', 'nest', 'peut', 'plus', 'plusieur', 'pourquoi', 'pourquoy', 'punc', 'quel', 'quelqu', 'quil', 'quon', 'results', 'sans', 'sont', 'stopword', 'tout', 'vers', 'étre', 'ſans', 'ſont'])


# In[21]:


stopwords_ro = nltk.corpus.stopwords.words('stop_words_ro.txt')


# In[22]:


stopwords_ro.extend(['pe'])


# In[23]:


stopwords_en = nltk.corpus.stopwords.words('stop_words_poetry.txt')


# In[24]:


stopwords_en.extend(['a', 'like', 'you', 'they', 'he', 'be', 'it', 'your', 'her', 'of', 'more', 'there', 'no', 'not', '’', 'what', 'my', 'his', 'she', 'to', 'our', 'me', 'we', 'in', 'can', 'us', 'an', 'if', 'do', 'this', '”', 'because', 'who', 'and', 'but', 'him'])


# In[25]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

tf_idf_vect0 = TfidfVectorizer(stop_words = stopwords_fr)

tfidf0 = tf_idf_vect0.fit_transform(preprocessed_documents_0)

#print(type(tfidf))
#W = tfidf.toarray()
#print(type(W))

dt = [('correlation', float)]

similarity_tfidf0 = np.matrix((tfidf0 * tfidf0.T).A, dtype=dt)


# In[26]:


# Set diagonal elements to 0
np.fill_diagonal(similarity_tfidf0, 0)


# In[27]:


# SAME AS ABOVE FOR ENGLISH
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

tf_idf_vect1 = TfidfVectorizer(stop_words = stopwords_en)

tfidf1 = tf_idf_vect1.fit_transform(preprocessed_documents_1)

dt = [('correlation', float)]

similarity_tfidf1 = np.matrix((tfidf1 * tfidf1.T).A, dtype=dt)

np.fill_diagonal(similarity_tfidf1, 0)


# In[28]:


# SAME AS ABOVE FOR ROMANIAN
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

tf_idf_vect2 = TfidfVectorizer(stop_words = stopwords_ro)

tfidf2 = tf_idf_vect2.fit_transform(preprocessed_documents_2)

dt = [('correlation', float)]

similarity_tfidf2 = np.matrix((tfidf2 * tfidf2.T).A, dtype=dt)

np.fill_diagonal(similarity_tfidf2, 0)


# In[29]:


import spacy


# In[30]:


nlp_fr = spacy.load('fr_core_news_lg')


# In[31]:


nlp_en = spacy.load('en_core_web_lg')


# In[33]:


scores_en, counts_en, enjambs_en = enjamb_en(documents_1)


# In[34]:


scores_en


# In[35]:


# WE MOVE ON TO OBTAINING THE (EN) ENJAMBMENT-SCORE SIMILARITY MATRIX 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 


# In[36]:


scores_array_en = np.array(scores_en)


# In[37]:


# Reshape scores into a column vector
scores_vec_en = scores_array_en.reshape(-1, 1)


# In[334]:


scores_vec_en


# In[39]:


scores_array_en = np.array(scores_en)

# Reshape scores into a column vector
scores_vec_en = scores_array_en.reshape(-1, 1)

# Normalize scores
norms = np.linalg.norm(scores_vec_en, axis=0)
normalized_scores = scores_vec_en / norms

# Calculate cosine similarity manually
dot_product = np.dot(normalized_scores, normalized_scores.T)
cosine_similarity_manual = dot_product

# Fill diagonal with zeros
np.fill_diagonal(cosine_similarity_manual, 0)

# Ensure values are within range [-1, 1]
cosine_similarity_manual = np.clip(cosine_similarity_manual, -1, 1)

# Print or use the resulting similarity matrix (cosine_similarity_manual)
print(cosine_similarity_manual)


# In[40]:


similarity_enjamb1 = cosine_similarity_manual


# In[41]:


n = len(scores_en)


# In[42]:


# Create an empty multiplex similarity matrix
SAM1 = np.zeros((2 * n, 2 * n))


# In[43]:


# Fill the first layer of the multiplex matrix with similarity_tfidf1
SAM1[:n, :n] = similarity_tfidf1

# Fill the second layer of the multiplex matrix with similarity_enjamb1
SAM1[n:, n:] = similarity_enjamb1


# # Add coupling strength between layers
# for node in range(n):
#     for layer1 in range(2):  # There are two layers
#         for layer2 in range(2):
#             if layer1 != layer2:
#                 SAM1[node + n, node + n] = coupling_strength

# In[44]:


# Rewrite the SAM1[:n, :n] assignment to extract float values from tuples
for i in range(n):
    for j in range(n):
        SAM1[i, j] = similarity_tfidf1[i, j][0]


# In[45]:


coupling_strength = 0.5


# In[46]:


for node in range(n):
    for layer1 in range(2):  # There are two layers
        for layer2 in range(2):
            if layer1 != layer2:
                SAM1[node + n, node + n] = coupling_strength


# In[47]:


S1 = np.sum(SAM1, axis = 1)
d1 = np.zeros(2*n)

for i in range(2*n):
    d1[i] = 1/S1[i]

D1 = np.diag([d1[i] for i in range(2*n)])

T1 = np.dot(SAM1, D1) 
# T is the transition matrix; it is a left stochastic matrix (i.e., all entries are non-negative and each column sums to 1)

RWBCsum1 = np.zeros(n)

from numpy import linalg as la


# In[48]:


number_of_layers = 2


# In[49]:


for i in range (n):  # absorbing the node
    U = np.copy(T1)
    for j in range(number_of_layers):
        U[: , i + j*n] = 0       # absorbing the node in all [2] layers
    V = la.inv((np.identity(2*n) - U))  # V = la.inv((np.identity(2*len(filelabels)) - U))
    Vsumcol = np.zeros((2*n, n))
    for m in range(n):
        for p in range(n):    
            Vsumcol[: , m] = Vsumcol[: , m] + np.squeeze(np.asarray(V[: , m + p*n]))
    Vsumrow = np.zeros((n, n))
    for m in range(n):
        for p in range(n):
            Vsumrow[m, :] = Vsumrow[m, :] + Vsumcol[m + p*n, :]
    Vsum = Vsumrow/number_of_layers
    RWBCsum1 = RWBCsum1 + np.sum(Vsum, axis = 1)


# In[ ]:


# WE NEED TO REGULARIZE THE FAULTY MATRIX


# In[50]:


epsilon = 1e-10


# In[51]:


for i in range (n):  # absorbing the node
    U = np.copy(T1)
    for j in range(number_of_layers):
        U[: , i + j*n] = 0       # absorbing the node in all [2] layers
    V = la.inv((np.identity(2*n) - U + epsilon * np.identity(2*n)))  # V = la.inv((np.identity(2*len(filelabels)) - U))
    Vsumcol = np.zeros((2*n, n))
    for m in range(n):
        for p in range(number_of_layers):    
            Vsumcol[: , m] = Vsumcol[: , m] + np.squeeze(np.asarray(V[: , m + p*n]))
    Vsumrow = np.zeros((n, n))
    for m in range(n):
        for p in range(number_of_layers):
            Vsumrow[m, :] = Vsumrow[m, :] + Vsumcol[m + p*n, :]
    Vsum = Vsumrow/number_of_layers
    RWBCsum1 = RWBCsum1 + np.sum(Vsum, axis = 1)


# In[52]:


RWBC1 = RWBCsum1/(n*(n - 1))

rankRWBC1 = RWBC1/sum(RWBC1)     # Normalizing the rank sizes so that they sum up to 1


# In[53]:


len(rankRWBC1)


# In[54]:


node_w_bc = []

for i, val in enumerate(rankRWBC1):
    node_w_bc.append((i, val))


# In[55]:


sorted1 = sorted(node_w_bc, key=lambda x: x[1], reverse = True)


# In[362]:


sorted1


# In[363]:


file_names_1[53]


# In[163]:


file_names_1[63]


# In[164]:


file_names_1[80]


# In[364]:


SAM1[98, 99]


# In[ ]:


# RO


# In[56]:


nlp_ro = spacy.load('ro_core_news_lg')


# In[59]:


scores_ro, counts_ro, enjambs_ro = enjamb_ro(documents_2)


# In[367]:


scores_ro


# In[60]:


scores_array_ro = np.array(scores_ro)

# Reshape scores into a column vector
scores_vec_ro = scores_array_ro.reshape(-1, 1)

# Normalize scores
norms_ro = np.linalg.norm(scores_vec_ro, axis=0)
normalized_scores_ro = scores_vec_ro / norms_ro

# Calculate cosine similarity manually
dot_product_ro = np.dot(normalized_scores_ro, normalized_scores_ro.T)
cosine_similarity_manual_ro = dot_product_ro

# Fill diagonal with zeros
np.fill_diagonal(cosine_similarity_manual_ro, 0)

# Ensure values are within range [-1, 1]
cosine_similarity_manual_ro = np.clip(cosine_similarity_manual_ro, -1, 1)

# Print or use the resulting similarity matrix (cosine_similarity_manual)
print(cosine_similarity_manual_ro)


# In[210]:


n


# In[211]:


len(scores_ro)


# In[61]:


similarity_enjamb2 = cosine_similarity_manual_ro


# In[62]:


# Create an empty multiplex similarity matrix
SAM2 = np.zeros((2 * n, 2 * n))

# Fill the second layer of the multiplex matrix with similarity_enjamb2
SAM2[n:, n:] = similarity_enjamb2

# First layer: tfidf2
for i in range(n):
    for j in range(n):
        SAM2[i, j] = similarity_tfidf2[i, j][0]


# In[63]:


for node in range(n):
    for layer1 in range(2):  # There are two layers
        for layer2 in range(2):
            if layer1 != layer2:
                SAM2[node + n, node + n] = coupling_strength


# In[388]:


SAM2[98, 99]


# In[389]:


SAM2[99, 105]


# In[64]:


S2 = np.sum(SAM2, axis = 1)
d2 = np.zeros(2*n)

for i in range(2*n):
    d2[i] = 1/S2[i]

D2 = np.diag([d2[i] for i in range(2*n)])

T2 = np.dot(SAM2, D2) 
# T is the transition matrix; it is a left stochastic matrix (i.e., all entries are non-negative and each column sums to 1)

RWBCsum2 = np.zeros(n)

from numpy import linalg as la


# In[65]:


for i in range (n):  # absorbing the node
    U = np.copy(T2)
    for j in range(number_of_layers):
        U[: , i + j*n] = 0       # absorbing the node in all [2] layers
    V = la.inv((np.identity(2*n) - U))
    Vsumcol = np.zeros((2*n, n))
    for m in range(n):
        for p in range(number_of_layers):    
            Vsumcol[: , m] = Vsumcol[: , m] + np.squeeze(np.asarray(V[: , m + p*n]))
    Vsumrow = np.zeros((n, n))
    for m in range(n):
        for p in range(number_of_layers):
            Vsumrow[m, :] = Vsumrow[m, :] + Vsumcol[m + p*n, :]
    Vsum = Vsumrow/number_of_layers
    RWBCsum2 = RWBCsum2 + np.sum(Vsum, axis = 1)


# In[216]:


# ONCE AGAIN, WE NEED TO REGULARIZE THE FAULTY MATRIX


# In[66]:


for i in range (n):  # absorbing the node
    U = np.copy(T2)
    for j in range(number_of_layers):
        U[: , i + j*n] = 0       # absorbing the node in all [2] layers
    V = la.inv((np.identity(2*n) - U + epsilon * np.identity(2*n))) # regularizing the matrix so that it does not end up singular again
    Vsumcol = np.zeros((2*n, n))
    for m in range(n):
        for p in range(number_of_layers):    
            Vsumcol[: , m] = Vsumcol[: , m] + np.squeeze(np.asarray(V[: , m + p*n]))
    Vsumrow = np.zeros((n, n))
    for m in range(n):
        for p in range(number_of_layers):
            Vsumrow[m, :] = Vsumrow[m, :] + Vsumcol[m + p*n, :]
    Vsum = Vsumrow/number_of_layers
    RWBCsum2 = RWBCsum2 + np.sum(Vsum, axis = 1)


# In[67]:


RWBC2 = RWBCsum2/(n*(n - 1))

rankRWBC2 = RWBC2/sum(RWBC2)     # Normalizing the rank sizes so that they sum up to 1


# In[68]:


node_w_bc2 = []

for i, val in enumerate(rankRWBC2):
    node_w_bc2.append((i, val))

sorted2 = sorted(node_w_bc2, key=lambda x: x[1], reverse = True)

sorted2


# In[223]:


file_names_2[68]


# In[222]:


file_names_2[69]


# In[69]:


making_cut_1 = []

for i, el in enumerate(sorted1):
    if i < 50 and el[0] > 68:
        print(i, el[0])
        making_cut_1.append((i, el[0]))


# In[70]:


making_cut_2 = []

for i, el in enumerate(sorted2):
    if i < 50 and el[0] > 68:
        print(i, el[0])
        making_cut_2.append((i, el[0]))


# In[71]:


print(len(making_cut_1), len(making_cut_2))


# In[229]:


set([el[1] for el in making_cut_1]).difference(set([ele[1] for ele in making_cut_2]))


# In[72]:


set([el[1] for el in making_cut_1]).difference(set([ele[1] for ele in making_cut_2]))


# In[230]:


set([el[1] for el in making_cut_2]).difference(set([ele[1] for ele in making_cut_1]))


# In[73]:


set([el[1] for el in making_cut_2]).difference(set([ele[1] for ele in making_cut_1]))


# In[231]:


# FR


# In[75]:


scores_fr, counts_fr, enjambs_fr = enjamb_fr(documents_0)


# In[235]:


scores_fr


# In[76]:


scores_array_fr = np.array(scores_fr)

# Reshape scores into a column vector
scores_vec_fr = scores_array_fr.reshape(-1, 1)

# Normalize scores
norms_fr = np.linalg.norm(scores_vec_fr, axis=0)
normalized_scores_fr = scores_vec_fr / norms_fr

# Calculate cosine similarity manually
dot_product_fr = np.dot(normalized_scores_fr, normalized_scores_fr.T)
cosine_similarity_manual_fr = dot_product_fr

# Fill diagonal with zeros
np.fill_diagonal(cosine_similarity_manual_fr, 0)

# Ensure values are within range [-1, 1]
cosine_similarity_manual_fr = np.clip(cosine_similarity_manual_fr, -1, 1)

# Print or use the resulting similarity matrix (cosine_similarity_manual)
print(cosine_similarity_manual_fr)


# In[77]:


similarity_enjamb0 = cosine_similarity_manual_fr


# In[78]:


# Create an empty multiplex similarity matrix
SAM0 = np.zeros((2 * n, 2 * n))

# Fill the second layer of the multiplex matrix with similarity_enjamb2
SAM0[n:, n:] = similarity_enjamb0

# First layer: tfidf2
for i in range(n):
    for j in range(n):
        SAM0[i, j] = similarity_tfidf0[i, j][0]


# In[79]:


for node in range(n):
    for layer1 in range(2):  # There are two layers
        for layer2 in range(2):
            if layer1 != layer2:
                SAM0[node + n, node + n] = coupling_strength


# In[80]:


S0 = np.sum(SAM0, axis = 1)
d0 = np.zeros(2*n)

for i in range(2*n):
    d0[i] = 1/S0[i]

D0 = np.diag([d0[i] for i in range(2*n)])

T0 = np.dot(SAM0, D0) 
# T is the transition matrix; it is a left stochastic matrix (i.e., all entries are non-negative and each column sums to 1)

RWBCsum0 = np.zeros(n)

from numpy import linalg as la


# In[81]:


for i in range (n):  # absorbing the node
    U = np.copy(T0)
    for j in range(number_of_layers):
        U[: , i + j*n] = 0       # absorbing the node in all [2] layers
    V = la.inv((np.identity(2*n) - U))
    Vsumcol = np.zeros((2*n, n))
    for m in range(n):
        for p in range(number_of_layers):    
            Vsumcol[: , m] = Vsumcol[: , m] + np.squeeze(np.asarray(V[: , m + p*n]))
    Vsumrow = np.zeros((n, n))
    for m in range(n):
        for p in range(number_of_layers):
            Vsumrow[m, :] = Vsumrow[m, :] + Vsumcol[m + p*n, :]
    Vsum = Vsumrow/number_of_layers
    RWBCsum0 = RWBCsum0 + np.sum(Vsum, axis = 1)


# In[241]:


# ONCE AGAIN, WE NEED TO REGULARIZE THE FAULTY MATRIX


# In[82]:


for i in range (n):  # absorbing the node
    U = np.copy(T0)
    for j in range(number_of_layers):
        U[: , i + j*n] = 0       # absorbing the node in all [2] layers
    V = la.inv((np.identity(2*n) - U + epsilon * np.identity(2*n)))
    Vsumcol = np.zeros((2*n, n))
    for m in range(n):
        for p in range(number_of_layers):    
            Vsumcol[: , m] = Vsumcol[: , m] + np.squeeze(np.asarray(V[: , m + p*n]))
    Vsumrow = np.zeros((n, n))
    for m in range(n):
        for p in range(number_of_layers):
            Vsumrow[m, :] = Vsumrow[m, :] + Vsumcol[m + p*n, :]
    Vsum = Vsumrow/number_of_layers
    RWBCsum0 = RWBCsum0 + np.sum(Vsum, axis = 1)


# In[83]:


RWBC0 = RWBCsum0/(n*(n - 1))

rankRWBC0 = RWBC0/sum(RWBC0)     # Normalizing the rank sizes so that they sum up to 1


# In[84]:


node_w_bc0 = []

for i, val in enumerate(rankRWBC0):
    node_w_bc0.append((i, val))

sorted0 = sorted(node_w_bc0, key=lambda x: x[1], reverse = True)

sorted0


# In[85]:


making_cut_0 = []

for i, el in enumerate(sorted0):
    if i < 50 and el[0] > 68:
        print(i, el[0])
        making_cut_0.append((i, el[0]))


# In[248]:


set([el[1] for el in making_cut_1]).intersection(set([ele[1] for ele in making_cut_2]))


# In[86]:


set([el[1] for el in making_cut_1]).intersection(set([ele[1] for ele in making_cut_2]))


# In[87]:


m_c_1_2 = set([el[1] for el in making_cut_1]).intersection(set([ele[1] for ele in making_cut_2]))


# In[252]:


m_c_0_1_2 = set([el[1] for el in making_cut_0]).intersection(m_c_1_2)

m_c_0_1_2


# In[88]:


m_c_0_1_2 = set([el[1] for el in making_cut_0]).intersection(m_c_1_2)

m_c_0_1_2


# In[253]:


m_c_0_1 = set([el[1] for el in making_cut_0]).intersection(set([el[1] for el in making_cut_1]))


# In[255]:


m_c_0_1


# In[89]:


m_c_0_1 = set([el[1] for el in making_cut_0]).intersection(set([el[1] for el in making_cut_1]))


# In[90]:


m_c_0_1


# In[257]:


m_c_0_2 = set([el[1] for el in making_cut_0]).intersection(set([el[1] for el in making_cut_2]))
m_c_0_2


# In[91]:


m_c_0_2 = set([el[1] for el in making_cut_0]).intersection(set([el[1] for el in making_cut_2]))
m_c_0_2


# In[254]:


m_c_0_1 == m_c_1_2


# In[256]:


m_c_0_1_2 == m_c_1_2


# In[ ]:





# In[258]:


diff_0_1 = set([el[1] for el in making_cut_0]).difference(set([el[1] for el in making_cut_1]))

diff_0_1


# In[92]:


diff_0_1 = set([el[1] for el in making_cut_0]).difference(set([el[1] for el in making_cut_1]))

diff_0_1


# In[260]:


diff_1_0 = set([el[1] for el in making_cut_1]).difference(set([el[1] for el in making_cut_0]))

diff_1_0


# In[93]:


diff_1_0 = set([el[1] for el in making_cut_1]).difference(set([el[1] for el in making_cut_0]))

diff_1_0


# In[261]:


diff_0_2 = set([el[1] for el in making_cut_0]).difference(set([el[1] for el in making_cut_2]))

diff_0_2


# In[94]:


diff_0_2 = set([el[1] for el in making_cut_0]).difference(set([el[1] for el in making_cut_2]))

diff_0_2


# In[262]:


diff_2_0 = set([el[1] for el in making_cut_2]).difference(set([el[1] for el in making_cut_0]))

diff_2_0


# In[95]:


diff_2_0 = set([el[1] for el in making_cut_2]).difference(set([el[1] for el in making_cut_0]))

diff_2_0


# In[96]:


cut_69_0 = sorted0[:69]


# In[97]:


cut_69_1 = sorted1[:69]


# In[98]:


cut_69_2 = sorted2[:69]


# In[275]:


# DO NOT RUN; THIS WAS IT BEFORE FIXING THE ENJAMB SIMILARITY MATRIX
mk_cut_0 = []

for i, ele in enumerate(cut_69_0):
    if ele[0] > 68:
        print(i, ele[0])
        mk_cut_0.append((i, ele[0]))


# In[99]:


mk_cut_0 = []

for i, ele in enumerate(cut_69_0):
    if ele[0] > 68:
        print(i, ele[0])
        mk_cut_0.append((i, ele[0]))


# In[429]:


len(mk_cut_0)


# In[100]:


len(mk_cut_0)


# In[277]:


# DO NOT RUN; THIS WAS IT BEFORE FIXING THE ENJAMB SIMILARITY MATRIX
mk_cut_1 = []

for i, ele in enumerate(cut_69_1):
    if ele[0] > 68:
        print(i, ele[0])
        mk_cut_1.append((i, ele[0]))


# In[101]:


mk_cut_1 = []

for i, ele in enumerate(cut_69_1):
    if ele[0] > 68:
        print(i, ele[0])
        mk_cut_1.append((i, ele[0]))


# In[102]:


len(mk_cut_1)


# In[279]:


# DO NOT RUN. OLD OUTPUT
mk_cut_2 = []

for i, ele in enumerate(cut_69_2):
    if ele[0] > 68:
        print(i, ele[0])
        mk_cut_2.append((i, ele[0]))


# In[103]:


mk_cut_2 = []

for i, ele in enumerate(cut_69_2):
    if ele[0] > 68:
        print(i, ele[0])
        mk_cut_2.append((i, ele[0]))


# In[104]:


len(mk_cut_2)


# In[423]:


inters_69 = set([el[1] for el in mk_cut_0]).intersection(set([el[1] for el in mk_cut_1]), set([el[1] for el in mk_cut_2]))

inters_69


# In[105]:


inters_69 = set([el[1] for el in mk_cut_0]).intersection(set([el[1] for el in mk_cut_1]), set([el[1] for el in mk_cut_2]))

inters_69


# In[412]:


inters_69 = set([el[1] for el in mk_cut_0]).intersection(set([el[1] for el in mk_cut_1]), set([el[1] for el in mk_cut_2]))

inters_69


# In[285]:


# DO NOT RUN, OLD OUTPUT
len(inters_69)


# In[106]:


len(inters_69)


# In[107]:


inters_names = []

for i in inters_69:
    inters_names.append(file_names_0[i])
    
inters_names


# In[414]:


inters_names = []

for i in inters_69:
    inters_names.append(file_names_0[i])
    
inters_names


# In[444]:


len(inters_names)


# In[432]:


diff_0_1 = set([el[1] for el in mk_cut_0]).difference(set([el[1] for el in mk_cut_1]))

diff_0_1


# In[108]:


diff_0_1 = set([el[1] for el in mk_cut_0]).difference(set([el[1] for el in mk_cut_1]))

diff_0_1


# In[109]:


for i in diff_0_1:
    print(file_names_0[i])


# In[110]:


diff_1_0 = set([el[1] for el in mk_cut_1]).difference(set([el[1] for el in mk_cut_0]))

diff_1_0


# In[435]:


for i in diff_1_0:
    print(file_names_0[i])


# In[ ]:





# In[436]:


diff_0_2 = set([el[1] for el in mk_cut_0]).difference(set([el[1] for el in mk_cut_2]))

diff_0_2


# In[111]:


diff_0_2 = set([el[1] for el in mk_cut_0]).difference(set([el[1] for el in mk_cut_2]))

diff_0_2


# In[112]:


for i in diff_0_2:
    print(file_names_0[i])


# In[113]:


diff_2_0 = set([el[1] for el in mk_cut_2]).difference(set([el[1] for el in mk_cut_0]))

diff_2_0


# In[114]:


file_names_0[77]


# In[115]:


diff_1_2 = set([el[1] for el in mk_cut_1]).difference(set([el[1] for el in mk_cut_2]))

diff_1_2


# In[116]:


for i in diff_1_2:
    print(file_names_0[i])


# In[117]:


diff_2_1 = set([el[1] for el in mk_cut_2]).difference(set([el[1] for el in mk_cut_1]))

diff_2_1


# In[118]:


file_names_0[77]


# In[119]:


union_0_1_2 = set([el[1] for el in mk_cut_0]).union(set([el[1] for el in mk_cut_1]), set([el[1] for el in mk_cut_2]))


# In[120]:


len(union_0_1_2)


# In[473]:


union_0_1_2


# In[121]:


for i in union_0_1_2:
    print(file_names_0[i])


# In[ ]:





# In[ ]:


# NOW LET'S BUILD THE TRILINGUAL MULTIPLEX


# In[ ]:





# In[305]:


similarity_enjamb0[0, 0]


# In[308]:


similarity_enjamb0[1, 3]


# In[311]:


similarity_tfidf0[44]


# In[122]:


similarity_tfidf = [similarity_tfidf0, similarity_tfidf1, similarity_tfidf2]
similarity_enjamb = [similarity_enjamb0, similarity_enjamb1, similarity_enjamb2]


# In[123]:


coupling_strength = 0.5


# In[124]:


def construct_multiplex_similarity_matrix(similarity_tfidf, similarity_enjamb, coupling_strength):
    n = similarity_tfidf[0].shape[0]  # Number of nodes
    num_layers = len(similarity_tfidf) + len(similarity_enjamb)
    
    # Initialize multiplex similarity matrix
    SAM = np.zeros((n*num_layers, n*num_layers))
    
    # Fill in similarity matrices
    for layer_idx, similarity_matrices in enumerate([similarity_tfidf, similarity_enjamb]):
        for sub_idx, similarity_matrix in enumerate(similarity_matrices):
            for i in range(n):
                for j in range(n):
                    SAM[i + (layer_idx*len(similarity_tfidf) + sub_idx)*n, 
                        j + (layer_idx*len(similarity_tfidf) + sub_idx)*n] = similarity_matrix[i, j][0] if layer_idx == 0 else similarity_matrix[i, j]
    
    # Add coupling strength between layers
    for node in range(n):
        for layer1 in range(num_layers):
            for layer2 in range(num_layers):
                if layer1 != layer2:
                    SAM[node + n*layer1, node + n*layer2] = coupling_strength
    
    return SAM


# In[125]:


SAM = construct_multiplex_similarity_matrix(similarity_tfidf, similarity_enjamb, coupling_strength)


# In[126]:


SAM.shape


# In[127]:


SAM[0,1]


# In[316]:


SAM[1,0]


# In[450]:


SAM[100,0]


# In[451]:


SAM[100,3]


# In[452]:


SAM[98, 0]


# In[453]:


SAM[98, 99]


# In[128]:


S = np.sum(SAM, axis = 1)
d = np.zeros(6*n)

for i in range(6*n):
    d[i] = 1/S[i]

D = np.diag([d[i] for i in range(6*n)])

T = np.dot(SAM, D) 
# T is the transition matrix; it is a left stochastic matrix (i.e., all entries are non-negative and each column sums to 1)

RWBCsum = np.zeros(n)

from numpy import linalg as la


# In[129]:


num_layers = len(similarity_tfidf) + len(similarity_enjamb)


# In[130]:


for i in range (n):  # absorbing the node
    U = np.copy(T)
    for j in range(num_layers):
        U[: , i + j*n] = 0       # absorbing the node in all layers
    V = la.inv((np.identity(6*n) - U))
    Vsumcol = np.zeros((6*n, n))
    for m in range(n):
        for p in range(num_layers):    
            Vsumcol[: , m] = Vsumcol[: , m] + np.squeeze(np.asarray(V[: , m + p*n]))
    Vsumrow = np.zeros((n, n))
    for m in range(n):
        for p in range(num_layers):
            Vsumrow[m, :] = Vsumrow[m, :] + Vsumcol[m + p*n, :]
    Vsum = Vsumrow/num_layers
    RWBCsum = RWBCsum + np.sum(Vsum, axis = 1)


# In[ ]:


# NO NEED TO REGULARIZE THE U MATRIX THIS TIME ROUND, WOW...


# In[131]:


RWBC = RWBCsum/(n*(n - 1))

rankRWBC = RWBC/sum(RWBC)     # Normalizing the rank sizes so that they sum up to 1


# In[132]:


node_w_bc = []

for i, val in enumerate(rankRWBC):
    node_w_bc.append((i, val))

sorted_all = sorted(node_w_bc, key=lambda x: x[1], reverse = True)

sorted_all


# In[133]:


cut_all = sorted_all[:69] # TOP 69 IN BETWEENNESS [RELEVANT IF WE WERE TO SELECT 
# THE SAME NUMBER OF POEMS AS IN THE YALE ABTHOLOGY]


# In[135]:


mk_cut_all = [] # THOSE THAT MAKE THE CUT

for i, ele in enumerate(cut_all):
    if ele[0] > 68:  # SELECTING POEMS THAT ARE NOT IN THE YALE CORPUS
        print(i, ele[0])
        mk_cut_all.append((i, ele[0]))


# In[136]:


len(mk_cut_all)


# In[137]:


names_all = []

for i in [el[1] for el in mk_cut_all]:
    names_all.append(file_names_0[i])
    
names_all


# In[138]:


names_0_1_2 = []

for i in union_0_1_2:     # THE UNION OF POEMS THAT MADE THE CUT IN EACH LANGUAGE SEPARATELY
    names_0_1_2.append(file_names_0[i])

names_0_1_2


# In[ ]:





# In[478]:


set(names_0_1_2) == set(names_all) 
# UNION OF TITLES THAT MADE CUT IN EACH LANGUAGE IS 
# NOT THE SAME AS TITLES THAT MADE CUT IN TRILINGUAL MULTIPLEX


# In[144]:


len(set(names_0_1_2)) == len(set(names_all))


# In[145]:


len(set(names_all))


# In[139]:


set(names_0_1_2).difference(set(names_all))


# In[140]:


set(names_all).difference(set(names_0_1_2))


# In[141]:


set(names_0_1_2).intersection(set(names_all))


# In[142]:


s5 = set(names_0_1_2).intersection(set(names_all))


# In[143]:


len(s5)


# In[ ]:





# In[264]:


print(file_names_0[69], file_names_0[76], file_names_0[78])


# In[266]:


print(file_names_0[77], file_names_0[83], file_names_0[85], file_names_0[87], file_names_0[91])


# In[268]:


file_names_0[69]


# In[ ]:




