import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Select features and target variable
X = dataset.iloc[:, 3:-1].values  # Features (input variables)
y = dataset.iloc[:, -1].values  # Target variable (output)

# Encoding categorical data
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # Encoding gender column

# Applying OneHotEncoder to the 'Geography' column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # One-hot encoding for country

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Scaling training data
X_test = sc.transform(X_test)  # Scaling test data

# Initializing the ANN
model = tf.keras.models.Sequential()

# Adding the input layer and first hidden layer
model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Making predictions and evaluating the model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # Converting probabilities to binary output

# Comparing predicted results with actual results
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculating the accuracy
print(accuracy_score(y_test, y_pred))
