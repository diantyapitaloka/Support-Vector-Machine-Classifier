## Support Vector Machine Classifier
Developed a model that was able to differentiate between chickens and pythons so that they could place the two animals in different cages. We certainly don't want to put chickens and snakes together in the same cage.
![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/1dcc7abc-a230-485b-b047-d61dc7382c5b)

We can create a classification model that separates the two classes using a Support Vector Machine. According to Aurelien Geron in the book Hands on Machine Learning, SVM works by creating a decision boundary or a field that is able to separate two classes. In this problem the decision boundary that is able to separate the chicken class and the snake class is a straight line which can be seen in the picture.
![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/ff1d60e5-fa59-4106-b2b6-1072ad8eda46)

Returning to the case of chicken and snake classification, the chicken and snake samples in the red circle are support vectors. Then we look for the widest path of the 2 support vectors. After finding the widest road, a decision boundary is then drawn based on that road.
![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/d33da4e3-531e-44b4-a8ff-7f34ac65bdd2)

## Support Vector Machine Multi-class Classification
SVM is actually a binary classifier or model for 2 class classification. However, SVM can also be used for multi-class classification using a technique namely "one-vs-rest".

![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/2d88757b-fe83-4b89-8afc-a4ddada9c7e7)

In multi-class classification problems, SVM performs binary classification for each class. The model then separates that class from all other classes, producing as many binary models as there are classes. To make predictions, all binary classification processes are run in the test phase.

For example, if there are 3 classes: donuts, chicken, and burgers, SVM will perform 3 classifications. First, build a separation between the donut class and the non-donut class.

![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/1fd81dec-9102-45f0-9db7-7e8420cd133e)

Then build a separator between the chicken class and the non-chicken class, then a separator between the burger class and the non-burger class. This technique is called "One-vs-Rest".

## Decision Boundary
Decision boundary is a line that divides a road or margin into 2 equal parts. The hyperplane is the plane that separates the two classes, while the margin is the width of the 'road' that divides the two classes. This dataset was collected by the National Institute of Diabetes and Digestive and Kidney Diseases. The dataset contains 8 attribute columns and 1 label column which contains 2 classes, namely 1 and 0. The number 1 indicates that the person is positive for diabetes and 0 indicates otherwise. There were 768 samples consisting of 768 female patients of Pima Indian descent.

## Purpose Machine Learning Model
The machine learning model that we will create aims to classify whether a patient is positive for diabetes or not. In the next stage we will import the pandas library and convert the dataset into a dataframe. Then, import the pandas library and convert the CSV file into a dataframe with the following code.
- import pandas as pd
- df = pd.read_csv('diabetes.csv')

## Display Row
Then we display the top 5 rows of the dataframe to see the contents of the dataset. To do this we can run the df.head() code as below.
- df.head()

## Check Missing Value
The next most important thing is that we need to check whether there are missing values in the dataset and whether there are attributes that do not contain numeric numbers. We can do this by calling the .info() function on the dataframe.
- df.info()

## Separating Attributes
The output from the info() function shows that all attribute values are complete, and also the values of each column have numeric data types, namely int64 and float64. At this stage the data can be used for model training.

Separating attributes in a dataset and storing them in a variable
- X = df[df.columns[:8]]
 
Separating the labels in the dataset and storing them in a variable
- Y = df['Outcome']

## Standardize Values from Dataset
If we look, the values in the dataset have different scales. For example, in the Glucose column and the Diabetes Pedigree Function column. We need to change the values of each attribute to be on the same scale. We can try to use standardization with the StandardScaler() function from SKLearn.
- from sklearn.preprocessing import StandardScaler
- scaler = StandardScaler()
- scaler.fit(X)
- X = scaler.transform(X)

## Testing Attributes
After the attributes and labels are separated, we can separate the data for training and testing using the .train_test_split() function.
- from sklearn.model_selection import train_test_split
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## Support Vector Classifier object
We then create a Support Vector Classifier object and store it in the clf variable. Finally we arrive at the stage we have been waiting for, we call the fit function to train the model.
- from sklearn.svm import SVC
 
## Create an SVC object
Call the fit function to train the model
- clf = SVC()
- clf.fit(X_train, y_train)

## Prediction Accuracy
Finally, we can see how accurate the predictions of the model we trained are on the testing data.

## Displays Prediction Accuracy Score
Successfully developed a Support Vector Classifier model to detect diabetes.
- clf.score(X_test, y_test)
![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/a230278b-fcbc-431f-b98b-1d0bc2758e20)



