# Image Classification using CIFAR10 dataset.

Compare Image Classification model accuracy between Artificial Neural Network(ANN) And Convolutional Neural Network(CNN) using CIFAR-10 dataset

## Artificial Neural Network 
* By training the model with an Artificial Neural Network(ANN), we got the following results.

```
Epoch 1/10
1563/1563 [==============================] - 65s 41ms/step - loss: 1.8129 - accuracy: 0.3541
Epoch 2/10
1563/1563 [==============================] - 49s 32ms/step - loss: 1.6249 - accuracy: 0.4263
Epoch 3/10
1563/1563 [==============================] - 50s 32ms/step - loss: 1.5440 - accuracy: 0.4560
Epoch 4/10
1563/1563 [==============================] - 50s 32ms/step - loss: 1.4839 - accuracy: 0.4795
Epoch 5/10
1563/1563 [==============================] - 49s 32ms/step - loss: 1.4349 - accuracy: 0.4942
Epoch 6/10
1563/1563 [==============================] - 50s 32ms/step - loss: 1.3900 - accuracy: 0.5114
Epoch 7/10
1563/1563 [==============================] - 50s 32ms/step - loss: 1.3502 - accuracy: 0.5262
Epoch 8/10
1563/1563 [==============================] - 50s 32ms/step - loss: 1.3173 - accuracy: 0.5384
Epoch 9/10
1563/1563 [==============================] - 49s 31ms/step - loss: 1.2840 - accuracy: 0.5492
Epoch 10/10
1563/1563 [==============================] - 50s 32ms/step - loss: 1.2550 - accuracy: 0.5606
313/313 [==============================] - 3s 8ms/step - loss: 1.3359 - accuracy: 0.5272

```
You can check the Classification Report below.
```
Classification Report: 
               precision    recall  f1-score   support

           0       0.63      0.53      0.58      1000
           1       0.63      0.66      0.64      1000
           2       0.41      0.40      0.40      1000
           3       0.38      0.37      0.38      1000
           4       0.48      0.43      0.45      1000
           5       0.55      0.30      0.38      1000
           6       0.51      0.65      0.57      1000
           7       0.55      0.62      0.58      1000
           8       0.61      0.68      0.64      1000
           9       0.52      0.65      0.58      1000

    accuracy                           0.53     10000
   macro avg       0.53      0.53      0.52     10000
weighted avg       0.53      0.53      0.52     10000

```
let's predict the X_test  and then comparer it with y_test.

```python
    #predict the model with X_test variable.
    y_pred = model.predict(X_test)
    # select the maximum number from the prediction set.
    y_label = [np.argmax(e) for e in y_pred]
    

    print(y_test[:10])
    print(y_label[:10])
    #you can see below the result of print statements and compare it together
[3, 8, 8, 0, 6, 6, 1, 6, 3, 1]
[3, 9, 8, 8, 4, 6, 1, 6, 4, 1]
```
## Convolutional Neural Network
* By training the model with an Convolutional Neural Network(CNN), we got the following results.

```
Epoch 1/10
1563/1563 [==============================] - 24s 15ms/step - loss: 1.4628 - accuracy: 0.4716
Epoch 2/10
1563/1563 [==============================] - 23s 15ms/step - loss: 1.1190 - accuracy: 0.6050
Epoch 3/10
1563/1563 [==============================] - 23s 15ms/step - loss: 0.9799 - accuracy: 0.6586
Epoch 4/10
1563/1563 [==============================] - 23s 15ms/step - loss: 0.8976 - accuracy: 0.6876
Epoch 5/10
1563/1563 [==============================] - 23s 15ms/step - loss: 0.8350 - accuracy: 0.7096
Epoch 6/10
1563/1563 [==============================] - 23s 15ms/step - loss: 0.7741 - accuracy: 0.7313
Epoch 7/10
1563/1563 [==============================] - 23s 15ms/step - loss: 0.7273 - accuracy: 0.7479
Epoch 8/10
1563/1563 [==============================] - 23s 15ms/step - loss: 0.6814 - accuracy: 0.7628
Epoch 9/10
1563/1563 [==============================] - 23s 15ms/step - loss: 0.6412 - accuracy: 0.7772
Epoch 10/10
1563/1563 [==============================] - 23s 15ms/step - loss: 0.6035 - accuracy: 0.7895
313/313 [==============================] - 2s 5ms/step - loss: 0.9193 - accuracy: 0.7018

```
You can check the Classification Report below.
```
Classification Report: 
               precision    recall  f1-score   support

           0       0.75      0.72      0.73      1000
           1       0.83      0.80      0.81      1000
           2       0.56      0.61      0.59      1000
           3       0.57      0.46      0.51      1000
           4       0.62      0.67      0.65      1000
           5       0.57      0.66      0.61      1000
           6       0.80      0.76      0.78      1000
           7       0.78      0.72      0.75      1000
           8       0.81      0.81      0.81      1000
           9       0.77      0.80      0.78      1000

    accuracy                           0.70     10000
   macro avg       0.70      0.70      0.70     10000
weighted avg       0.70      0.70      0.70     10000
```
let's predict the X_test  and then comparer it with y_test.

```python
    #predict the model with X_test variable.
    y_pred = model.predict(X_test)
    # select the maximum number from the prediction set.
    y_label = [np.argmax(e) for e in y_pred]
    
    print(y_test[:10])
    print(y_label[:10])
    
    """
    You can see the result below and compare it with the result of ANN 
    to show how effective it is to use CNN in image classification.
    """
[3, 8, 8, 0, 6, 6, 1, 6, 3, 1]
[3, 8, 8, 0, 6, 6, 1, 6, 3, 1]
```
