## 🌷🌼🌻 Support Vector Machine Classifier 🌻🌼🌷
Developed a model that was able to differentiate between chickens and pythons so that they could place the two animals in different cages. We certainly don't want to put chickens and snakes together in the same cage.
![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/efc67d17-ee6f-4a60-add8-b80e5b6566fe)

We can create a classification model that separates the two classes using a Support Vector Machine. According to Aurelien Geron in the book Hands on Machine Learning, SVM works by creating a decision boundary or a field that is able to separate two classes. In this problem the decision boundary that is able to separate the chicken class and the snake class is a straight line which can be seen in the picture.
![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/209c764a-2384-43d5-b079-1037baf2d19b)

Returning to the case of chicken and snake classification, the chicken and snake samples in the red circle are support vectors. Then we look for the widest path of the 2 support vectors. After finding the widest road, a decision boundary is then drawn based on that road.
![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/6ce86813-dc12-48bc-9527-fc557edbf5e6)

## 🌷🌼🌻 Support Vector Machine Non-Linear Classification 🌻🌼🌷
Previously we learned about the support vector classifier for the linear case. Support vector classifier works by looking for the largest margin, or the widest path that is able to separate 2 classes. The problem is, the data in the field is much more complex than the data on ornamental chickens and snakes as above.

![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/5745dee1-39fd-4695-aee7-4a75121540c9)

The data above is data that cannot be separated linearly, so we call it non-linear data. In non-linear data, the decision boundary calculated by the SVM algorithm is not a straight line. Even though it is quite complicated to determine the decision boundary in this case, we also get the advantage, namely, we can capture more complex relationships from each scattered data point.

For data like above, the Support Vector Classifier uses a method, namely the "kernel trick" so that the data can be separated linearly. What are kernel tricks? It is a method for converting data in certain dimensions (eg 2D) into higher dimensions (3D) so that it can produce an optimal hyperplane. Look at the following image.

![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/f109e43b-6a06-4c6c-b0d7-7d088799b2eb)

First, we need to calculate the distance score of two data points, for example x_i and x_j. Scores will be higher for closer data points, and vice versa. Then we use this score to map the data in higher dimensions (3D). This technique is useful for reducing computing time and resources, especially for large amounts of data. This also prevents the need for a more complex transformation process. That's why this technique is often referred to as a kernel trick.

As the image above shows, mapping data points from 2D to 3D space uses a kernel function. The red dots that were previously in the center are now in the vertical plane at a lower position after being converted to 3D space. Data points that were previously difficult to separate can now be easily separated using kernel techniques.


## 🌷🌼🌻 Support Vector Machine Multi-class Classification 🌻🌼🌷
SVM is actually a binary classifier or model for 2 class classification. However, SVM can also be used for multi-class classification using a technique namely "one-vs-rest".

![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/2d88757b-fe83-4b89-8afc-a4ddada9c7e7)

In multi-class classification problems, SVM performs binary classification for each class. The model then separates that class from all other classes, producing as many binary models as there are classes. To make predictions, all binary classification processes are run in the test phase.

For example, if there are 3 classes: donuts, chicken, and burgers, SVM will perform 3 classifications. First, build a separation between the donut class and the non-donut class.

![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/1fd81dec-9102-45f0-9db7-7e8420cd133e)

Then build a separator between the chicken class and the non-chicken class, then a separator between the burger class and the non-burger class. This technique is called "One-vs-Rest".

## 🌷🌼🌻 Decision Boundary 🌻🌼🌷
Decision boundary is a line that divides a road or margin into 2 equal parts. The hyperplane is the plane that separates the two classes, while the margin is the width of the 'road' that divides the two classes. This dataset was collected by the National Institute of Diabetes and Digestive and Kidney Diseases. The dataset contains 8 attribute columns and 1 label column which contains 2 classes, namely 1 and 0. The number 1 indicates that the person is positive for diabetes and 0 indicates otherwise. There were 768 samples consisting of 768 female patients of Pima Indian descent.

## 🌷🌼🌻 Purpose Machine Learning Model 🌻🌼🌷
The machine learning model that we will create aims to classify whether a patient is positive for diabetes or not. In the next stage we will import the pandas library and convert the dataset into a dataframe. Then, import the pandas library and convert the CSV file into a dataframe with the following code.
- import pandas as pd
- df = pd.read_csv('diabetes.csv')

## 🌷🌼🌻 Display Row 🌻🌼🌷
Then we display the top 5 rows of the dataframe to see the contents of the dataset. To do this we can run the df.head() code as below.
- df.head()

## 🌷🌼🌻 Check Missing Value 🌻🌼🌷
The next most important thing is that we need to check whether there are missing values in the dataset and whether there are attributes that do not contain numeric numbers. We can do this by calling the .info() function on the dataframe.
- df.info()

## 🌷🌼🌻 Separating Attributes 🌻🌼🌷
The output from the info() function shows that all attribute values are complete, and also the values of each column have numeric data types, namely int64 and float64. At this stage the data can be used for model training.

Separating attributes in a dataset and storing them in a variable
- X = df[df.columns[:8]]
 
Separating the labels in the dataset and storing them in a variable
- Y = df['Outcome']

## 🌷🌼🌻 Standardize Values from Dataset 🌻🌼🌷
If we look, the values in the dataset have different scales. For example, in the Glucose column and the Diabetes Pedigree Function column. We need to change the values of each attribute to be on the same scale. We can try to use standardization with the StandardScaler() function from SKLearn.
- from sklearn.preprocessing import StandardScaler
- scaler = StandardScaler()
- scaler.fit(X)
- X = scaler.transform(X)

## 🌷🌼🌻 Testing Attributes 🌻🌼🌷
After the attributes and labels are separated, we can separate the data for training and testing using the .train_test_split() function.
- from sklearn.model_selection import train_test_split
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## 🌷🌼🌻 Support Vector Classifier object 🌻🌼🌷
We then create a Support Vector Classifier object and store it in the clf variable. Finally we arrive at the stage we have been waiting for, we call the fit function to train the model.
- from sklearn.svm import SVC
 
## 🌷🌼🌻 Create an SVC object 🌻🌼🌷
Call the fit function to train the model
- clf = SVC()
- clf.fit(X_train, y_train)

## 🌷🌼🌻 Prediction Accuracy 🌻🌼🌷
Finally, we can see how accurate the predictions of the model we trained are on the testing data.
![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/faa34831-a093-4e7a-a512-6ca6525ff903)

## 🌷🌼🌻 Displays Prediction Accuracy Score 🌻🌼🌷
Successfully developed a Support Vector Classifier model to detect diabetes.
- clf.score(X_test, y_test)
![image](https://github.com/diantyapitaloka/Support-Vector-Machine-Classifier/assets/147487436/a230278b-fcbc-431f-b98b-1d0bc2758e20)



