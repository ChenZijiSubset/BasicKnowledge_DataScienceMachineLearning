from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def polynomial_regression(data, degrees=2, frac_samples=.7):
    fig, ax = plt.subplots(1,1,figsize=(15,9))
    
    X = data.drop(columns=['Species', 'Weight'])
    y = data[['Weight']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=frac_samples, random_state=3)
    
    poly_tr = PolynomialFeatures(degree=degrees).fit(X_train)
    poly = poly_tr.transform(X_train)
    
    lin_model = LinearRegression().fit(poly, y_train)
    y_pred = lin_model.predict(poly)
    
    model_results = pd.DataFrame({'X':X_train.Length1.values,
                                  'y_pred':y_pred.flatten()}).sort_values(by='X')
    
    ax.scatter(X_train.Length1, y_train, label='Training data')
    ax.scatter(X_test.Length1, y_test, c='purple', alpha=0.2, label = 'Testing data')
    ax.plot(model_results.X, model_results.y_pred, c='r', label = f'Polynomial Regression (deg={degrees})')
    ax.set_xlabel('Fish Length1',size=14)
    ax.set_ylabel('Fish Weight',size=14)
    ax.legend()
    plt.show()
    print(f'RMSE Train set: {np.sqrt(mean_squared_error(y_train, y_pred))}')
    print(f'RMSE Test set: {np.sqrt(mean_squared_error(y_test, lin_model.predict(poly_tr.transform(X_test))))}')



def polynomial_regression_with_PCA(data, degrees=2, frac_samples=.7, nb_pc=5):
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    
    X = data.drop(columns=['Species', 'Weight'])
    y = data[['Weight']]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=frac_samples, random_state=3)
    
    #⚠️ Polynomial Transformation first ⚠️
    poly_tr = PolynomialFeatures(degree=degrees).fit(X_train_raw)
    X_train = poly_tr.transform(X_train_raw)
    X_test = poly_tr.transform(X_test_raw)
    
    
    #⚠️ Data must be centered around its mean before applying PCA ⚠️
    scaler = StandardScaler().fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #⚠️ Applying PCA ⚠️
    pca = PCA().fit(X_train)
    
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(X_train_pca.shape)
    lin_model = LinearRegression().fit(X_train_pca[:,0:nb_pc], y_train)
    y_pred = lin_model.predict(X_train_pca[:,0:nb_pc])
    y_test_pred = lin_model.predict(X_test_pca[:,0:nb_pc])
    
    model_results = pd.DataFrame({'X':X_train_raw.Length1,
                                  'y_pred':y_pred.flatten()}).sort_values(by='X')
    
    ax.scatter(X_train_raw.Length1, y_train, label='Training data')
    ax.scatter(X_test_raw.Length1, y_test, c='purple', alpha=0.2, label = 'Testing data')
    ax.plot(model_results.X, model_results.y_pred, c='r', label = f'Polynomial Regression (deg={degrees})')
    ax.set_xlabel('Fish Length1',size=14)
    ax.set_ylabel('Fish Weight',size=14)
    ax.legend()
    plt.show()
    print(f'RMSE Train set: {np.sqrt(mean_squared_error(y_train, y_pred))}')
    print(f'RMSE Test set: {np.sqrt(mean_squared_error(y_test, y_test_pred))}')