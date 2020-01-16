# UT-Final-Project
Training a machine learning model to identify whether particular transactions are fraud.
## Attention!
Make sure XGBClassifier is installed: conda install -c conda-forge xgboost

Also, you may have to revert matplotlib library back to 1.5:

pip uninstall matplotlib

pip install matplotlib==1.3.1



## Introduction
The library that i have chosen to use is the XGBoost library, which i think is somehow connected to sklearn.
More specifically, the type of machine learning that is functioning is a combonation between logistical regression and gradiant boosting trees.
Logistical regression is optimal for binary factors such as in this case of whether a transaction can be classified as fraud or not.






## Explanation of the code
The jupyter file starts out by running "prep". This is a file that takes in the "transactions.csv" and cleans it so it is ready for processing.

Next "training_testing" is ran which calls upon the "get_data" function in "prep" to collect the x and y needed

Spliting the data into four different groups called trainX, testX, trainY, and testY is then fitted into the xgbclassifier.

The probability predictions and regular predictions are generated and used to be the basis of three different metrics which I got from https://scikit-learn.org/stable/modules/classes.html

Then the "pickle" library is used to save the model into "models" which can be called upon later for prediction.

A really cool visulaization is creted with the next section of code, which I found on https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

There is some code in "training_testing.py" which I was experimeting with, but could not figure out how to function. Its in there so I can go back later and tweak.
