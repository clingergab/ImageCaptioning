"""Copy of Gabriel_Clinger_gc2821_homework5 from google colab

# COMS W4705 - Homework 5 
## Image Captioning with Conditioned LSTM Generators
Daniel Bauer <bauer@cs.columbia.edu>


"""

# Commented out IPython magic to ensure Python compatibility.
import os
from collections import defaultdict
import numpy as np
import PIL
from matplotlib import pyplot as plt
# %matplotlib inline

from keras import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate, Activation
from keras.activations import softmax
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from keras.applications.inception_v3 import InceptionV3

from keras.optimizers import Adam

from google.colab import drive



#this is where you put the name of your data folder.
#Please make sure it's correct because it'll be used in many places later.
MY_DATA_DIR="hw5data"

"""### Mounting your GDrive so you can access the files from Colab"""

#running this command will generate a message that will ask you to click on a link where you'll obtain your GDrive auth code.
#copy paste that code in the text box that will appear below
drive.mount('/content/drive')

"""Please look at the 'Files' tab on the left side and make sure you can see the 'hw5_data' folder that you have in your GDrive."""

FLICKR_PATH = os.path.join("/content/drive/My Drive", MY_DATA_DIR)
print(FLICKR_PATH)

"""## Part I: Image Encodings """

def load_image_list(filename):
    with open(filename,'r') as image_list_f: 
        return [line.strip() for line in image_list_f]

train_list = load_image_list(os.path.join(FLICKR_PATH, 'Flickr_8k.trainImages.txt'))
dev_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.devImages.txt'))
test_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.testImages.txt'))

"""Let's see how many images there are"""

len(train_list), len(dev_list), len(test_list)

"""Each entry is an image filename."""

dev_list[20]

"""The images are located in a subdirectory.  """

IMG_PATH = os.path.join(FLICKR_PATH, "Flickr8k_Dataset")

"""We can use PIL to open the image and matplotlib to display it. """

image = PIL.Image.open(os.path.join(IMG_PATH, dev_list[20]))
image



plt.imshow(image)


np.asarray(image).shape

"""The values range from 0 to 255. """

np.asarray(image)

"""We can use PIL to resize the image and then divide every value by 255. """

new_image = np.asarray(image.resize((299,299))) / 255.0
plt.imshow(new_image)

new_image.shape

"""Let's put this all in a function for convenience. """

def get_image(image_name):
    image = PIL.Image.open(os.path.join(IMG_PATH, image_name))
    return np.asarray(image.resize((299,299))) / 255.0

plt.imshow(get_image(dev_list[25]))

"""Next, we load the pre-trained Inception model. """

img_model = InceptionV3(weights='imagenet') # This will download the weight files for you and might take a while.

img_model.summary() # this is quite a complex model.


new_input = img_model.input
new_output = img_model.layers[-2].output
img_encoder = Model(new_input, new_output) # This is the final Keras image encoder model we will use.



encoded_image = img_encoder.predict(np.array([new_image]))

encoded_image


def img_generator(img_list):
    for img in img_list:
      yield np.array([get_image(img)])

"""Now we can encode all images (this takes a few minutes)."""

enc_train = img_encoder.predict_generator(img_generator(train_list), steps=len(train_list), verbose=1)

enc_train[11]

enc_dev = img_encoder.predict_generator(img_generator(dev_list), steps=len(dev_list), verbose=1)

enc_test = img_encoder.predict_generator(img_generator(test_list), steps=len(test_list), verbose=1)

OUTPUT_PATH = "/content/drive/My Drive/4705_hw5_output" 
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

np.save(os.path.join(OUTPUT_PATH,"encoded_images_train.npy"), enc_train)
np.save(os.path.join(OUTPUT_PATH,"encoded_images_dev.npy"), enc_dev)
np.save(os.path.join(OUTPUT_PATH,"encoded_images_test.npy"), enc_test)

"""## Part II Text (Caption) Data Preparation """


def read_image_descriptions(filename):  
  image_descriptions = defaultdict(list)    
  with open(filename) as f:
    for line in f:
      lineParts = line.lower().split()
      name = lineParts[0][:-2]
      caption = ['<START>'] + lineParts[1:] + ['<END>']
      image_descriptions[name].append(caption)
  return image_descriptions

descriptions = read_image_descriptions(os.path.join(FLICKR_PATH, "Flickr8k.token.txt"))
#print(train_list.index("1000268201_693b08cb0e.jpg"))
#print(descriptions[train_list[1077]])

print(descriptions[dev_list[0]])


tokens = set()
for val in descriptions.values():
  for v in val:
    for token in v:
      tokens.add(token)
tokens = sorted(list(tokens))
#print(tokens)

id_to_word = {}
word_to_id = {}

for i in range(len(tokens)):
  id_to_word[i] = tokens[i]
  word_to_id[tokens[i]] = i

word_to_id["wharfs"] # should print an integer
# print(len(word_to_id))
# id_to_word[8919]

id_to_word[8654] # should print a token

## Part III Basic Decoder Model


max(len(description) for image_id in train_list for description in descriptions[image_id])


MAX_LEN = 40
EMBEDDING_DIM=300
vocab_size = len(word_to_id)
# Text input
text_input = Input(shape=(MAX_LEN,))
embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(text_input)
x = Bidirectional(LSTM(512, return_sequences=False))(embedding)
pred = Dense(vocab_size, activation='softmax')(x)
model1 = Model(inputs=[text_input],outputs=pred)
model1.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model1.summary()


def text_training_generator(batch_size=128):
  i = 0
  input = np.zeros((batch_size, MAX_LEN))
  output = np.zeros((batch_size, vocab_size))

  while True:
    for image_id in train_list:  
      for description in descriptions[image_id]:
        for j in range(len(description) - 1):
          temp = description[:j + 1]

          for k in range(len(temp)):
            input[i][k] = word_to_id[temp[k]]

          output[i][word_to_id[description[j + 1]]] = 1 
          i += 1

          if i >= batch_size:
            i = 0
            yield input, output
            input = np.zeros((batch_size, MAX_LEN))
            output = np.zeros((batch_size, vocab_size))
    yield input, output



batch_size = 128
generator = text_training_generator(batch_size)

steps = sum([len(descr) - 1 for img in train_list for descr in descriptions[img]]) // batch_size

model1.fit_generator(generator, steps_per_epoch=steps, verbose=True, epochs=20)

model1.save_weights("drive/My Drive/4705_hw5_output/model.h5a")

model1.load_weights("drive/My Drive/4705_hw5_output/model.h5a")



def decoder():
  sentence = np.zeros(MAX_LEN)
  sentence[0] = word_to_id["<START>"]
  output = ["<START>"]
  for i in range(1, MAX_LEN):
    predictions = model1.predict(np.array([sentence]))
    pred = np.argmax(predictions)
    sentence[i] = pred
    output.append(id_to_word[pred])
    if pred == word_to_id["<END>"]:
      break
  return output
decoder()

print(decoder())


def sample_decoder():
  sentence = np.zeros(MAX_LEN)
  sentence[0] = word_to_id["<START>"]
  output = ["<START>"]

  for i in range(1, MAX_LEN):
    predictions = model1.predict(np.array([sentence]))[0].astype('float64')

    if np.sum(predictions) > 1.0:
      predictions = predictions / np.sum(predictions)

    predictions = np.random.multinomial(1, predictions)
    pred = np.argmax(predictions)
    sentence[i] = pred
    output.append(id_to_word[pred])

    if pred == word_to_id["<END>"]:
      break

  return output

for i in range(10): 
    print(sample_decoder())

"""## Part III - Conditioning on the Image """


MAX_LEN = 40
EMBEDDING_DIM=300
IMAGE_ENC_DIM=300

# Image input
img_input = Input(shape=(2048,))
img_enc = Dense(300, activation="relu") (img_input)
images = RepeatVector(MAX_LEN)(img_enc)

# Text input
text_input = Input(shape=(MAX_LEN,))
embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(text_input)
x = Concatenate()([images,embedding])
y = Bidirectional(LSTM(256, return_sequences=False))(x) 
pred = Dense(vocab_size, activation='softmax')(y)
model = Model(inputs=[img_input,text_input],outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer="RMSProp", metrics=['accuracy'])

model.summary()


my_data_dir = "4705_hw5_output"
enc_train = np.load("drive/My Drive/"+my_data_dir+"/encoded_images_train.npy")
enc_dev = np.load("drive/My Drive/"+my_data_dir+"/encoded_images_dev.npy")

def training_generator(batch_size=128):
  i = 0
  image_inputs = np.zeros((batch_size, 2048))
  input = np.zeros((batch_size, MAX_LEN))
  output = np.zeros((batch_size, vocab_size))
  
  while True:
    for image_id in train_list + dev_list:  
      for description in descriptions[image_id]:
        for j in range(len(description) - 1):
          temp = description[:j + 1]
          if image_id in train_list:
            image_inputs[i] = enc_train[train_list.index(image_id)]
          else:
            image_inputs[i] = enc_dev[dev_list.index(image_id)]

          for k in range(len(temp)):
            input[i][k] = word_to_id[temp[k]]

          output[i][word_to_id[description[j + 1]]] = 1 
          i += 1

          if i >= batch_size:
            i = 0
            yield [image_inputs, input], output
            image_inputs = np.zeros((batch_size, 2048))
            input = np.zeros((batch_size, MAX_LEN))
            output = np.zeros((batch_size, vocab_size))

    yield [image_inputs, input], output


batch_size = 128
generator = training_generator(batch_size)
steps = sum([len(descr) - 1 for img in (train_list+dev_list) for descr in descriptions[img]]) // batch_size

model.fit_generator(generator, steps_per_epoch=steps, verbose=True, epochs=20)


model.save_weights("drive/My Drive/"+my_data_dir+"/model.h5")

"""to load the model: """
model.load_weights("drive/My Drive/"+my_data_dir+"/model.h5")


def image_decoder(enc_image): 
  sentence = np.zeros(MAX_LEN)
  sentence[0] = word_to_id["<START>"]
  output = ["<START>"]

  for i in range(1, MAX_LEN):
    predictions = model.predict([np.array([enc_image]), np.array([sentence])])[0]
    pred = np.argmax(predictions)
    sentence[i] = pred
    output.append(id_to_word[pred])

    if pred == word_to_id["<END>"]:
      break
      
  return np.array([output])[0]


plt.imshow(get_image(train_list[0]))
image_decoder(enc_train[0])


plt.imshow(get_image(dev_list[1]))
image_decoder(enc_dev[0])


## Part IV - Beam Search Decoder (24 pts)


def img_beam_decoder(n, image_enc):
  sequences = [(list(), 0.0)]

beam_decoder(3, dev_list[1])
