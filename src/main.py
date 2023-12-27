import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import tensorflow_probability as tfp

# Load the training dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')

# Visualize the correlation matrix
plt.figure(figsize=(12, 6))
corr_matrix = train_data.drop(columns=['id']).corr()
sns.heatmap(corr_matrix, annot=True)

# Visualize some scatter plots
plt.figure(figsize=(15, 7))
plt.subplot(221)
sns.scatterplot(data=train_data, x='allelectrons_Average', y='atomicweight_Average', hue='Hardness')
plt.subplot(222)
sns.scatterplot(data=train_data, x='allelectrons_Average', y='R_vdw_element_Average', hue='Hardness')
plt.subplot(223)
sns.scatterplot(data=train_data, x='allelectrons_Average', y='R_cov_element_Average', hue='Hardness')
plt.subplot(224)
sns.scatterplot(data=train_data, x='allelectrons_Average', y='density_Average', hue='Hardness')

# Define features and target variable
features = ['allelectrons_Total', 'density_Total', 'allelectrons_Average',
            'val_e_Average', 'atomicweight_Average', 'ionenergy_Average',
            'el_neg_chi_Average', 'R_vdw_element_Average', 'R_cov_element_Average',
            'zaratio_Average', 'density_Average', 'Hardness']

X = train_data[features].drop(columns='Hardness')
y = train_data.Hardness

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

# Train a preliminary LightGBM model
model_preliminary = LGBMRegressor()
model_preliminary.fit(X, y)

# Create a new DataFrame for predictions
X_new = X.copy()
X_new['Hardness_pred'] = model_preliminary.predict(X)

# Define loss and metric functions for TensorFlow model
def loss_function(y_true, y_pred):
    return tfp.stats.percentile(tf.abs(y_true - y_pred), q=50)

def metric_function(y_true, y_pred):
    return tfp.stats.percentile(tf.abs(y_true - y_pred), q=100) - tfp.stats.percentile(tf.abs(y_true - y_pred), q=0)

# Define callbacks for TensorFlow model
callbacks_list_tf = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=2, mode='min', restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.00001),
    tf.keras.callbacks.TerminateOnNaN()
]

# Create the TensorFlow model
def create_tf_model():
    input_layer = tf.keras.Input(shape=(len(features),))
    x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(input_layer)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.013, beta_1=0.5),
                  loss=loss_function,
                  metrics=metric_function)
    return model

# Train the TensorFlow model
model_tf = create_tf_model()
history_tf = model_tf.fit(X_new.astype('float32'), y.astype('float32'),
                          epochs=100,
                          class_weight=model_preliminary.class_weight,
                          callbacks=callbacks_list_tf,
                          validation_split=0.1)

# Load the sample submission and test datasets
sample_submission = pd.read_csv('/kaggle/input/playground-series-s3e25/sample_submission.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')

# Make predictions using the preliminary model for the test dataset
test_data['Hardness_pred'] = model_preliminary.predict(test_data.astype('float32').drop(columns='id'))

# Make predictions using the TensorFlow model for the test dataset
test_data["Hardness"] = model_tf.predict(test_data.astype('float32').drop(columns='id'))

# Prepare the submission file
submission = test_data[['id', "Hardness"]]
submission.to_csv("submission.csv", index=False)