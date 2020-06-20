Credit Card Fraud Detection Model

Language: Python (v3.7)
Editor: PyCharm Community Edition (v2019.1.3)
Descirption: AI model which inputs a list of transactions some fraudulent, some legitimate 
and calculates the percentage of fraudulent v/s legitimate cases, the accuracy and loss.

Note:
Basic packages applied in this project are pandas, numpy, tensorflow. The Neural Netwrok algorithm is used to train
the model on the test dataset.

OVERVIEW:
-Fetch dataset from kaggle repository, learn about dimensions of data through kaggle description and EDA
-link: https://www.kaggle.com/dalpozz/creditcardfraud
-Manipulate dataset into a format that we want to work with
-Build a model as a tensorflow computational graph
-Train and test the model
-Output accuracy when comparing predicted and actual values

MEANS TO TWEAK THE MODEL FOR ACCURACY:
-Change the number of cells in each layer of neural network (Set: i/p ~ 30 - 100 - 150 - 2 ~ o/p)
-Increase logit weighting for fraudulent transactions (Set: 0.172% which is the percentage of fraudulent cases)
-Add a layer of neural netwrok (Set: 3 Layers)
-Change functions used in neural network (Used: Sigmoid, dropout, softmax) according to nature of dataset and layer
 in which it is used
-Change optimizer (Used: AdamOptimizer)
-Decrease learning rate (Set: 0.005)
-Increase number of epochs (Set: 100)
