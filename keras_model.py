import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf 


data = keras.datasets.fashion_mnist

#get test and training data

(train_images, train_labels) , (test_images, test_labels) = data.load_data()

class_names = ["t-shirt", "trouser", "pullover", "dress", "coat"
               ,"sandal", "shirt", "sneaker", "bag", 'ankleboot']


# convert the data from 0 and 1

train_images= train_images/255.0
train_labels= train_labels/255.0

# build the model with keras input,hidden and output layers
model = keras.Sequential([
   keras.layers.Flatten(input_shape=(28,28)), #input layer with flattened data ie no line arrays
   keras.layers.Dense(128, activation="relu"), #hidden layer with 128 nodes 
   keras.layers.Dense(10, activation="softmax"), #output layer with 10 nodes/neurons
])

#compile the model

model.compile( optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# epoch means shuffling the same images from the dataset and giving it again to the 
#  for more accuracy.
  
model.fit(train_images, train_labels, epochs=4)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)

prediction = model.predict(test_images)

for i in range(1055, 1060):
   plt.grid(False)
   plt.imshow(test_images[i])
   plt.xlabel("Actual : " + class_names[test_labels[i]])
   plt.title("Predictions : " + class_names[np.argmax(prediction[i])])
   plt.show()



