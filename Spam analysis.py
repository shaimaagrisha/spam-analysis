import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("spam.csv")
df = df.iloc[:,0:2]

df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)

df.describe()

df.groupby('target').describe()


import seaborn as sns

sns.distplot(df.text.str.len())
sns.countplot(df.target)
plt.xlabel("Class")
plt.title('Number of ham and spam messages')

# convert label to a numerical variable
df['label_num'] = df.target.map({'ham':0, 'spam':1})
df.head()

df["target"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()

'''
As we continue our analysis we want to start thinking about the features we are going to be using.
 This goes along with the general idea of feature engineering. 
 The better your domain knowledge on the data, the better your ability to engineer more features from it.
'''


df['text_len'] = df.text.apply(len)
df.head()
df[['target','text_len']].groupby('target').mean()


import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['patch.force_edgecolor'] = True
plt.style.use('seaborn-bright')
df.hist(column='text_len', by='target', bins=60,figsize=(15,7))
plt.legend()
plt.xlabel("Text Length frequency")

import nltk
# nltk.download()

from nltk.corpus import stopwords 
import string    

def text_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    
    return " ".join(text)

df["text"] = df["text"].apply(text_process)



from collections import Counter

def feq_words(data , text_column , class_column , classes_list):
    classes_dict = dict()
    for c in classes_list:
        z = data[text_column].where(data[class_column] == c)
        z = np.array(z.dropna())
        z = pd.DataFrame(data = np.array(z),columns=["text"])
        z = Counter(" ".join(z["text"]).split()).most_common(100)
        classes_dict[c] = z
    return classes_dict

classes_dict = feq_words(df,"text","target",["ham","spam"])


''''
from wordcloud import WordCloud ,STOPWORDS

def feq_words_cloud (data , text_column , class_column , classes_list):
    for c in classes_list:
        z = data[text_column].where(data[class_column] == c)
        wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(z) 
# plot the WordCloud image                        
        plt.figure(figsize = (8, 8), facecolor='k') 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
        plt.show() 


feq_words_cloud(df,"text","target",["ham","spam"])
'''

df_ham = df[df["target"]=="ham"]
df_spam = df[df["target"]=="spam"]
df_ham = df_ham.iloc[0:800]

balanced_df = pd.concat([df_ham, df_spam])

#balanced_df["text"] = balanced_df["text"].apply(split(" "))

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
stemmed = []
for t in balanced_df["text"]:
    text = t[0]
    stemmed_text = []
    for word in text:
        stemmed_word = stemmer.stem(word.lower())
        stemmed_text.append(stemmed_word)
    stemmed.append((stemmed_text, t[1]))

stemmed[51]

X = balanced_df.text
y = balanced_df.target
print(X.shape)
print(y.shape)


from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
Y = le.fit_transform(y)
Y = Y.reshape(-1,1)

from keras.utils import to_categorical
Y = to_categorical(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)



from keras.optimizers import adam
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping



def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=adam(),metrics=['f1-score','accuracy'])

model.fit(sequences_matrix,y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

acc = model.evaluate(test_sequences_matrix,y_test)

