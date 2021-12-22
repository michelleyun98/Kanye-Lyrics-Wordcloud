# Kanye-Lyrics-Wordcloud

# Install packages and import libraries
!pip install wordcloud
!pip install pillow

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Read and tokenize lyrics

file = open("lyrics.txt") # Open and read file
txt = file.read()

nltk.download('averaged_perceptron_tagger')
tokenized_txt = nltk.word_tokenize(txt) 

# Remove stopwords

nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))


# Function to delete stopwords

def delete_stops(t):
    txt_wo_stopwords = [] # Empty set of non-stopwords
    for w in t:
        # if word w is not in the set of stopwords, append w to txt_wo_stopwords 
        if w not in stops:
            txt_wo_stopwords.append(w)
    return txt_wo_stopwords


# List of tokenized words without stopwords
stopless_txt = delete_stops(tokenized_txt)

# Remove punctuation

punctuation = ['1', '2', '!', ',', "''", '[',']', '...', ':', '--', '``', '?', '.', '(', ')', "''"]


def delete_punct(st):
    txt_wo_punct = []
    for w in st:
        if w not in punctuation:
            txt_wo_punct.append(w)
    return txt_wo_punct

punctless_txt = delete_punct(stopless_txt)

# Remove contractions and other meaningless morphemes

meaningless = ['cause','wan','yeah', 'oh','yeah','la','said','see','yeah','let','say', 'back','It', 'ai', "n't", 'I', "'m'", "'s", "'re'", "'m", "'",'And', 'You', 'get', 'got', 'Oh', "'re",'Do','This', '\ufeff', 'Chorus', 'Verse', 'Refrain', 'Intro', 'Outro', 'Bridge', 'West', 'go', 'ca', 'gon', 'na', "'ll", 'If', 'My', "'ve", 'know', 'Kanye', 'like', 'That', 'The', 'They', 'But', 'How', 'So', 'She', 'Get', 'He', 'put']

def delete_ml(pt):
    txt_wo_ml = []
    for w in pt:
        if w not in meaningless:
            txt_wo_ml.append(w)
    return txt_wo_ml

# This is the final list of strings

final_txt = delete_ml(punctless_txt)

# Write final_txt to a text file 

file = open('a_file.txt', 'w')
for s in final_txt:
    file.writelines(s + ' ')
file.close()

# Read new text file
rfile = open('a_file.txt', 'r')
text = rfile.read()

# Test if the contents in final_txt has been written to a_file.txt

f = open('a_file.txt', 'r', encoding = 'utf-8')

# Read mask image
#ye_mask = np.array(Image.open(path.join(d, "ye.png")))
# Generate WordCloud

wordcloud = WordCloud(max_words=200, background_color='white').generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
