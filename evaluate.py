import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf

def Evaluate_performance(model_type, model, history, test_data, test_label, display=True):
    
    # Prediction
    if model_type == 'nn':
        scaler = StandardScaler()
        test_data_scl = scaler.fit_transform(test_data)
        test_data_ts = tf.constant(test_data_scl, dtype=tf.float16)
        test_pred = np.argmax(model.predict(test_data_ts), axis=1)
    if model_type == 'knn':
        test_pred = model.predict(test_data)
    
    # Performance
    print('Performance in tset dataset:')
    print(classification_report(test_label, test_pred))
    print('Accuracy: ', accuracy_score(test_label, test_pred))
    print('Confusion matrix:')
    print(confusion_matrix(test_label, test_pred))
    
    if display == True:
        # Display learning curve
        key_names = list(history.keys())
        colors = ['-r','--b']

        plt.figure(figsize=(16, 5))
        for i in range(len(key_names)):
            plt.subplot(1,2,i+1)
            plt.plot(history[key_names[i]], colors[i], label=key_names[i])
            plt.legend(fontsize=15,ncol=2)
            plt.title('Learning Curves', size=15)