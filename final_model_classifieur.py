import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import ast
import os
from nn_utils import TrainingHistory
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Dropout, MaxPooling1D, Conv1D, Flatten, Reshape
from keras.models import Model
import itertools
from keras.utils import np_utils
from sklearn.metrics import (classification_report, 
                             precision_recall_fscore_support, 
                             accuracy_score)

from keras.preprocessing import text, sequence

path = "Tobacco3482-OCR/"
classes = os.listdir(path)

nb = []
x = []
y = []
for cls in classes:
    files = os.listdir(path + cls)
    for file in files:
        with open(path + cls + "/" + file, 'r') as f:
            txt = f.read()
        x.append(txt)
        y.append(cls)
    nb.append(str(len(files)))
print(str(nb))
x = np.array(x)
y = np.array(y)
#print(np.unique(y))

# To replace the \n with space
for i in range(x.shape[0]):
    x[i] = x[i].replace("\n", " ")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Create document vectors
vectorizer = CountVectorizer(max_features=2000)
vectorizer.fit(x_train)
x_train_counts = vectorizer.transform(x_train)
x_test_counts = vectorizer.transform(x_test)

# With TF-IDF representation
tf_transformer = TfidfTransformer()
tfidf = tf_transformer.fit(x_train_counts)
x_train_tf = tfidf.transform(x_train_counts)
x_test_tf = tfidf.transform(x_test_counts)

# Model parameters
MAX_FEATURES = 2000
MAX_TEXT_LENGTH = 2000
BATCH_SIZE = 16
EPOCHS = 20
VALIDATION_SPLIT = 0.1
NB_CLASS = len(np.unique(y_train))

x_train_tf = x_train_tf.toarray()
x_train_tf = np.reshape(x_train_tf, (x_train_tf.shape[0], x_train_tf.shape[1], 1))
x_test_tf = x_test_tf.toarray()
x_test_tf = np.reshape(x_test_tf, (x_test_tf.shape[0], x_test_tf.shape[1], 1))

def get_model():

    inp = Input(shape=(MAX_TEXT_LENGTH,1))
    #model = Embedding(MAX_FEATURES, EMBED_SIZE)(inp)
    model = Dropout(0.5)(inp)
    model = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(model)
    model = MaxPooling1D(pool_size=2)(model)
    model = Flatten()(model)
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(NB_CLASS, activation="softmax")(model)
    model = Model(inputs=inp, outputs=model)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train_fit_predict(model, x_train, x_test, y, history):
    
    model.fit(x_train, y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS, verbose=1,
              validation_split=VALIDATION_SPLIT)

    return model.predict(x_test)


# Get the list of different classes
CLASSES_LIST = np.unique(y_train)
n_out = len(CLASSES_LIST)
print(CLASSES_LIST)

# Convert clas string to index
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(CLASSES_LIST)
y_train = le.transform(y_train) 
y_test = le.transform(y_test) 
train_y_cat = np_utils.to_categorical(y_train, n_out)

# define the NN topology
model = get_model()

# Define training procedure
history = TrainingHistory(x_test_tf, y_test, CLASSES_LIST)

# Train and predict
y_predicted = train_fit_predict(model, x_train_tf, x_test_tf, train_y_cat, history).argmax(1)


print("Test Accuracy:", accuracy_score(y_test, y_predicted))

p, r, f1, s = precision_recall_fscore_support(y_test, y_predicted, 
                                              average='micro',
                                              labels=[x for x in 
                                                      np.unique(y_train) 
                                                      if x not in ['CSDECMOTV']])

print('p r f1 %.1f %.2f %.3f' % (np.average(p, weights=s)*100.0, 
                                 np.average(r, weights=s)*100.0, 
                                 np.average(f1, weights=s)*100.0))


print(classification_report(y_test, y_predicted, labels=[x for x in 
                                                       np.unique(y_train) 
                                                       if x not in ['CSDECMOTV']]))