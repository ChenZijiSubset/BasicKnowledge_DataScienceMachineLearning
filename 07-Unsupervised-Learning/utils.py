import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def reconcile_labels(data, index, encoder):
    df = pd.DataFrame(data=data,index=index).T
    df = df.astype(dict.fromkeys(index , 'str'))
    num_cat = df['Actual Labels'].value_counts().sort_values(ascending=False).index
    actual_cat_names = list(encoder.inverse_transform(num_cat.astype(int)))
    actual_cat_names.append('unlabelled/noise')
    
    for cat in num_cat:
        actual_name = encoder.inverse_transform([int(cat)])[0]
        
        for col in df.columns:
            df.loc[df[col]=='-1',col]='unlabelled/noise'
            sorted_labels = df.loc[((df['Actual Labels']==cat) & (df[col]!='-1') & (~df[col].isin(actual_cat_names))),col].value_counts().sort_values()
            
            if sorted_labels.shape[0]>0:
                corresponding_label = sorted_labels.index[-1]
                df.loc[(df[col]==corresponding_label),col] = actual_name
            else:
                idx = df.loc[df['Actual Labels']==cat].sample(1).index[0]
                df.loc[idx][col]=actual_name
            
    for col in df.columns:
        mask = ~df[col].isin(actual_cat_names)
        df.loc[mask, col]=df.loc[mask, col].apply(lambda x: f'x_{x}')
            
    return df

def multi_plot(params, X, encoder, marker_size=5,X_y_label=('PC 1','PC 2'), fig_title='Projected space',loc=(-.12,0)):
    X = X.copy()
    n_axes = len(params.keys())
    fig, axes = plt.subplots(1,n_axes,figsize=(30,8))
    labels_data = reconcile_labels(params.values(),list(params.keys()),encoder)
    for ax, title, labels in zip(axes.flatten(),params.keys(), labels_data.values.T):
        X['label']=labels
        for name, group in X.groupby('label'):
            ax.scatter(group.iloc[:,0], group.iloc[:,1], label=name, s=marker_size)
            ax.set_title(title); ax.set_xlabel(X_y_label[0]); ax.set_ylabel(X_y_label[1]);
        ax.legend(loc=loc)
    fig.suptitle(fig_title, size=20)
    return labels_data

def compare_embedding(params,labels_data, figsize = (30,10),loc=(.85,0.8), marker_size=5):
    n_axes = len(params.keys())
    fig, axes = plt.subplots(1,n_axes,figsize=figsize)
    
    for ax, title, data in zip(axes.flatten(),params.keys(), params.values()):
        X = pd.DataFrame(data=[data['x'],data['y'],labels_data], index=['x','y','label']).T
        
        #if title == 'Before PCA (initial space)':
        #    return X

        for name, group in X.groupby('label'):
            ax.scatter(group['x'], group['y'], label=name, s=marker_size)
            ax.set_title(title); ax.set_xlabel(data['xlabel']); ax.set_ylabel(data['ylabel']);
        ax.legend(loc=loc)
    return None