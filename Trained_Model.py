from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

#load the dataset into the variables 
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# reshape y_train and y_test to one dimension array
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# data normalization
X_train = X_train/255
X_test = X_test/255

labels=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

def print_image(x, y, index):
    plt.imshow(x[index])
    plt.xlabel(labels[y[index]])
    
#Artificial_Neural_Network
def ANN(X,y,epochs):
    # create normal neural network.
    model = models.Sequential([
            layers.Flatten(input_shape=(32,32,3)),#first layer contain 3072 input.
            layers.Dense(3000, activation = 'relu'),
            layers.Dense(1000, activation='relu'),
            layers.Dense(10, activation='sigmoid')
        ])
    
    #compile the model.
    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #fit the model with training parameters.
    model.fit(X, y , epochs= epochs)
    
    #evaluate the model with test/evaluation parameters.
    model.evaluate(X_test, y_test)
    
    #predict the model with test data.
    y_pred = model.predict(X_test)
    y_label = [np.argmax(e) for e in y_pred]
    
    print(y_test[:10])
    print(y_label[:10])
    
    print("Classification Report: \n", classification_report(y_test, y_label))
 

# convolutional Neural Network
# apply the same data set with convolutional neural network to compare the result with the Artificial_Neural_Network
def CNN(X,y,epochs):
    model = models.Sequential([
            # creae the first convolutional layer with filter=32, kernal size = 3x3
            layers.Conv2D(32, (3,3), activation='relu', input_shape = (32, 32 ,3)),
            # the second pooling layer with pooling size = 2x2
            layers.MaxPool2D((2,2)),
            # repeat the above two layers also extract more features
            layers.Conv2D(64, (3,3), activation='relu', input_shape = (32, 32 ,3)),
            layers.MaxPool2D((2,2)),
            # then send the pooling output to latten layer and then to normal neural network.
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y , epochs= epochs)
    model.evaluate(X_test, y_test)
    
    y_pred = model.predict(X_test)
    y_label = [np.argmax(e) for e in y_pred]
    
    print(y_test[:10])
    print(y_label[:10])
    
    print("Classification Report: \n", classification_report(y_test, y_label))
   

ANN(X_train,y_train,10)
CNN(X_train,y_train,10)

