from data import faqs
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])
tokenizer.word_index

input_sequence = []

for sentence in faqs.split('\n'):
  tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

  for i in range(1,len(tokenized_sentence)):
    input_sequence.append(tokenized_sentence[:i+1])
    
input_sequence

max_len = max([len(x) for x in input_sequence])
padded_input_sequences = pad_sequences(input_sequence, maxlen = max_len, padding='pre')

X = padded_input_sequences[:,:-1]
y = padded_input_sequences[:,-1]
y = to_categorical(y,num_classes=283)

model = Sequential()
model.add(Embedding(283,100,input_length=56))
model.add(LSTM(150))
model.add(Dense(283,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

print(model.summary())

model.fit(X,y,epochs=100)
text = "i want to learn C++"

for i in range(10):
  # tokenize
  token_text = tokenizer.texts_to_sequences([text])[0]
  # padding
  padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
  # predict
  pos = np.argmax(model.predict(padded_token_text))

  for word,index in tokenizer.word_index.items():
    if index == pos:
      text = text + " " + word
      print(text)
      time.sleep(2)