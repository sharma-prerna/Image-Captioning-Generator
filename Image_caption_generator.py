
import numpy as np
from numpy import array
import matplotlib.pyplot as plt


import string
import os
import glob
from PIL import Image
from time import time

from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from tensorflow.keras.layers.wrappers import Bidirectional
from tensorflow.keras.layers.merge import add
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

'''from google.colab import drive
drive.mount('/content/drive')'''

token_path = "/content/drive/My Drive/Flickr8k/Flickr8k.token.txt"
train_images_path = '/content/drive/My Drive/Flickr8k/Flickr8k.trainImages.txt'
test_images_path = '/content/drive/My Drive/Flickr8k/Flickr8k.testImages.txt'
images_path = '/content/drive/My Drive/Flickr8k/Images/'
glove_path = '/content/drive/My Drive/Flickr8k/glove.6B.200d.txt'
with open(token_path,'r') as f:
  desc = f.read()
print(desc[:410])

#it's a dictionry having image id's as keys and list of all captions of that key as values
captions_dict = dict()
for l in desc.split('\n'):
        #this splits the line into image id and captions
        tokens = l.split()
        if len(l) > 2:
          #split the image id into name and .jpg extension and keep only image name without extension
          image_name = tokens[0].split('.')[0]
          #image caption
          image_cap = ' '.join(tokens[1:])
          if image_name not in captions_dict:
              captions_dict[image_name] = list()
          captions_dict[image_name].append(image_cap)
          
print(captions_dict["1000268201_693b08cb0e"])

#it creates the mapping of to be replaced characters with respective desired characters
#here fisrt parameter contains the string of characters that has to be replaced
#second parameters contains the string of respective chars that are the replacements of first parameter's chars
#third parameter is to map characters to None (completely remove that from the string)
map_table = str.maketrans('', '', string.punctuation)


for image_name, cap_list in captions_dict.items():
    for i in range(len(cap_list)):
        cap = cap_list[i]
        cap = cap.split()
        cap = [word.lower() for word in cap]
        #translate does rest of the work of map table by finish mapping
        cap = [w.translate(map_table) for w in cap]
        cap_list[i] =  ' '.join(cap)
    captions_dict[image_name] = cap_list

print(captions_dict["1000268201_693b08cb0e"])

pic = '1000268201_693b08cb0e.jpg'
#imread reads and converts the image into numpy array
x=plt.imread(images_path+pic)
plt.imshow(x)
plt.show()
captions_dict['1000268201_693b08cb0e']

vocabulary = set()
for image_name in captions_dict.keys():
        [vocabulary.update(c.split()) for c in captions_dict[image_name]]
print('Original Vocabulary Size: %d' % len(vocabulary))

lines = list()
for image_name, cap_list in captions_dict.items():
    for cap in cap_list:
        lines.append(image_name + ' ' + cap)

#without .jpg extension. A multiline string
new_captions = '\n'.join(lines)
print(new_captions.split('\n')[0])

with open(train_images_path,'r') as f:
  train_images = f.read().strip().split("\n")

with open(test_images_path,'r') as f:
  test_images = f.read().strip().split("\n")
  
dataset = list()
for img in train_images:
    if len(img) > 1:
      image_name = img.split('.')[0]
      dataset.append(image_name)

#set of all distinct training images names
train_data = set(dataset)
#print(train_data)
print(test_images[:5])

#glob gets complete paths or let's say global of the files of particular directory
img = glob.glob(images_path + '*.jpg')
print(len(img))
#list of all training images that contains complete path of the images unlike train_images
train_img = list()
test_img = list()
for i in img: 
    if i[len(images_path):] in train_images:
        train_img.append(i)
    if i[len(images_path):] in test_images:
        test_img.append(i) 
      
print(test_img[0])
print(train_img[0])

train_captions = dict()
for line in new_captions.split('\n'):
    tokens = line.split()
    image_name, image_cap = tokens[0], tokens[1:]
    if image_name in train_data:
        if image_name not in train_captions:
            train_captions[image_name] = list()
        cap = 'START ' + ' '.join(image_cap) + ' END'
        train_captions[image_name].append(cap)
print(train_captions["1000268201_693b08cb0e"])

all_train_captions = []
count = 0
for name,caps in train_captions.items():
    for cap in caps:
      if cap not in all_train_captions:
        all_train_captions.append(cap)

print(all_train_captions[:5])
print(len(all_train_captions))

threshold = 10
word_counts = dict()
for cap in all_train_captions:
    for word in cap.split(' '):
        word_counts[word] = word_counts.get(word, 0) + 1

#it creates the dictionary of word in which each word occurs atleast 10 times
vocab = [word for word in word_counts if word_counts[word] >= threshold]
print('Vocabulary = %d' % (len(vocab)))

ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1

max_length = max([len(cap.split()) for cap in all_train_captions])
print('capription Length: %d' % max_length)

embeddings_index = {} 
with open(glove_path,"r",encoding="utf-8") as f:
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs

embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)

#use transfer learning
model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    #flattens the 3 dimensional array of RGB image
    x = image.img_to_array(img)
    #expands the dimension along given axis
    x = np.expand_dims(x, axis=0)
    #preprocess input performs 3 operations : noramlization of vector(/255) , change the range of data from (0 to 1) to (-2 to 0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image)
    #changes the dimesnion (n,1) or (1,n) to (n,) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

encoding_train = {}
for img in train_img:
    encoding_train[img[len(images_path):]] = encode(img)
train_features = encoding_train

encoding_test = {}
for img in test_img:
    encoding_test[img[len(images_path):]] = encode(img)
test_features = encoding_test
print(len(train_features))

train_features["1000268201_693b08cb0e"+".jpg"].shape

#creates input layer with no. of neurons = 2048 in it, for the cnn model
inp_1 = Input(shape=(2048,))
#regularize the neurons activation by reducing it to 50%
reg_1 = Dropout(0.5)(inp_1)
#creates dense layer with no. of neurons = 256 in it, this layer is connected to inp layer defined above
out_1 = Dense(256, activation='relu')(reg_1)

#creates input layer with size max_length of caption, for lstm
inp_2 = Input(shape=(max_length,))
#200 dimensional vector to each input word (vocab). This statement only creates a mapping table that maps word to 
#a vector in n dimension, in our case 200, and then whenever a input word comes, it replaces with the corresponding vector
#note here,embedding layer is always connected to the first layer or the input layer of the network
#mask_zero identifies the special zero padding and ignores it to continue variable size computing 
hidden_1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inp_2)
reg_2 = Dropout(0.5)(hidden_1)
#create lstm unit having output size 256 or let's say neuorns. It is connceted to the input layer
out_2 = LSTM(256)(reg_2)

#merging by adding the outpts of lstm and cnn having same dimesion 256 and return single layer with same dimension
decoder1 = add([out_1, out_2])
#dense layer (or let's say fully connected layer to above layer)
decoder2 = Dense(256, activation='relu')(decoder1)
#final probabilistic output 
output = Dense(vocab_size, activation='softmax')(decoder2)

#creating a model finaly
model = Model(inputs=[inp_1, inp_2], outputs=output)
model.summary()

#set weights the first hidden layer and set trainable = False as the weights are already trained
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
#adam optimizer is the combination of two most useful algorithms : AdaGrad, RMSprop
#categorial_crossentropy: refined version of binary cross entrop(only for 2 classes)
model.compile(loss='categorical_crossentropy', optimizer='adam')

def data_generator(captions, photos, wordtoix, max_length, batch_size):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images until yeild doesn't return data
    while 1:
        for key, cap_list in captions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for cap in cap_list:
                # encode the sequence
                seq = [wordtoix[word] for word in cap.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequences with leading zeros
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence to 
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n==batch_size:
                yield ([array(X1), array(X2)], array(y))
                X1, X2, y = list(), list(), list()
                n=0

epochs = 30
batch_size = 3
steps = len(train_captions)//batch_size
#data_generator returns the list (yielded)
generator = data_generator(train_captions, train_features, wordtoix, max_length, batch_size)

#verbose = 0: show progress baar steady along with loss
#verbose = 1: show progress baar dynamic along with loss
#verbose = 2: no progress bar, only loss
print("\n================== MODEL TRAINING BEGAN ==================\n")
model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)

def greedySearch(photo):
    caption = 'START'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in caption.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y = model.predict([photo,sequence], verbose=0)
        #select the one with the geatest probability
        y = np.argmax(y)
        word = ixtoword[y]
        caption += ' ' + word
        if word == 'END':
            break

    #remove star and end tags
    caption.replace("START","")
    caption.replace("END","")
    return caption

pic = list(encoding_test.keys())[10]
image = encoding_test[pic].reshape((1,2048))
x=plt.imread(images_path+pic)
plt.imshow(x)
plt.show()

print("Greedy Search:",greedySearch(image))

pic = list(encoding_test.keys())[1]
image = encoding_test[pic].reshape((1,2048))
x=plt.imread(images_path+pic)
plt.imshow(x)
plt.show()

print("Greedy:",greedySearch(image))

"""# New Section"""