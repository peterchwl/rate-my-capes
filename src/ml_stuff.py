import pandas as pd
import random

import nltk
nltk.download('wordnet')

import nltk
nltk.download('omw-1.4')

pip install -U nltk

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import regularizers
from keras.utils.vis_utils import plot_model
import torch

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

from numpy import array
from numpy import asarray
from numpy import zeros

!git clone https://github.com/jasonwei20/eda_nlp

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)



df = pd.read_csv('labelling_spreadsheet - labelling_spreadsheet.csv')
df.head()

df2 = df.iloc[:550, :]
df2

df3 = df2.dropna()
df3.reset_index(inplace=True)
df3.head()


update_df = df3.drop([df3.index[126], df3.index[127], df3.index[228], df3.index[257]])

update_df.drop('index', axis=1, inplace=True)
update_df

update_df.drop(update_df[update_df['Review']=='No Comments'].index, inplace=True)

count=0
with open("eda_nlp/data/file.txt", "w") as output:
    for i in update_df['Review']:
      output.write(str(random.randint(0,1))+"\t"+str(i)+"\n")
      count+=1
print(count)

f = open("eda_nlp/data/file.txt", "r")
print(f.read())

!python eda_nlp/code/augment.py --input='eda_nlp/data/file.txt' --num_aug=12

f = open("eda_nlp/data/eda_file.txt", "r")
print(f.read())

with open("eda_nlp/data/eda_file.txt", "r") as output:
  lines=output.readlines()
  print(lines)
  
new_lines=[]
for i in lines:
  temp = i.split(' ')
  temp[0] = temp[0].replace("\t", "")
  temp[0] = temp[0].replace("1", "")
  temp[0] = temp[0].replace("0", "")
  temp[-1] = temp[-1].replace("\n", "")
  new_lines.append(' '.join(temp))
  
print(new_lines)

new_df = pd.DataFrame()
new_df['Review'] = pd.Series(new_lines)
print(new_df)

labels = ['Easygoing',
'strict',
'accessible',
'inaccessible',
'(review)_content_with_testing',
'(review)_content_with_prof',
'(review)_content_with_coursework',
'(review)_content_with_grading', 
'(review)_discontent_with_testing',
'(review)_discontent_with_prof',
'(review)_discontent_with_coursework',
'(review)_discontent_with_grading']

for i in labels:
  new_df[i] = pd.Series()
print(new_df)

count=0
for i in range(0,6942,13):
  new_df.iloc[i:i+13,1:] = update_df.iloc[count,1:]
  count+=1
print(new_df)

print(new_df.sample(10))

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

print(new_df.head())
  
new_df.drop(['Easygoing', 'strict', 'accessible', 'inaccessible'], axis=1, inplace=True)
print(new_df.head())

X = []
sentences = list(new_df["Review"])
for sen in sentences:
    X.append(preprocess_text(sen))

y = new_df.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('/content/drive/My Drive/ratemycapes2/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
from tensorflow.keras import regularizers

deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
#LSTM_Layer_2 = LSTM(128)(LSTM_Layer_1)

dense_layer_1 = Dense(8, activation='sigmoid')(LSTM_Layer_1)
#dense_layer_1 = Dense(8, activation='sigmoid', kernel_regularizer='l2')(LSTM_Layer_2)
#dropout_layer_1 = tf.keras.layers.Dropout(0.2)(dense_layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['acc'])

print(model.summary())

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

print(y_train.shape)

import torch

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_split=0.1, steps_per_epoch=170)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

df_test = pd.DataFrame()
df_test['Review']=pd.Series('''lectures are useless, he goes way to fast and does explain, jumps from question to answer, test were really hard, class average extremely low and he is threatening to CURVE TO A C-!!!!!!!''')
print(df_test)

X = []
sentences = list(df_test["Review"])
for sen in sentences:
    X.append(preprocess_text(sen))
    
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)


vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X = pad_sequences(X, padding='post', maxlen=maxlen)

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('/content/drive/My Drive/ratemycapes2/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
pred = model.predict(X)
labels = [
'(review)_content_with_testing',
'(review)_content_with_prof',
'(review)_content_with_coursework',
'(review)_content_with_grading', 
'(review)_discontent_with_testing',
'(review)_discontent_with_prof',
'(review)_discontent_with_coursework',
'(review)_discontent_with_grading']
print(pred, labels[np.argmax(pred)])
