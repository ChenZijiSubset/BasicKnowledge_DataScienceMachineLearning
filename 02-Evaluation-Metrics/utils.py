from ipywidgets import interact, IntSlider, FloatSlider
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns

import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score




def sigmoid(x):
    return 1/(1+np.exp(-x))

def calculate_proba(x, threshold):
    data = np.linspace(-5, 5, num=11)
    true_class = [0,0,1, 0, 0, 1, 0, 1, 0, 1, 1]
    proba = sigmoid(data)
    predictions = [1 if p  >= threshold else 0 for p in proba]
    
    return (data, true_class, proba, predictions)
    

def calculate_matrix(predictions, true_class):
    cf_data = list(zip(predictions, true_class))
    tp = sum([1 if d==(1, 1) else 0 for d in cf_data])
    tn = sum([1 if d==(0, 0) else 0 for d in cf_data])
    fp = sum([1 if d==(1, 0) else 0 for d in cf_data])
    fn = sum([1 if d==(0, 1) else 0 for d in cf_data])
    
    #array = [[tp,fp],[tn,fn]]
    array = [[1,2],[3,4]]
    
    df_cm = pd.DataFrame(array, index = ['Negative', 'Positive'],
                  columns = ['Negative', 'Positive'])
    
    annotations = pd.DataFrame([[f'TN\n{tn}',f'FP\n{fp}'],[f'FN\n{fn}',f'TP\n{tp}' ]])
    return df_cm, annotations

def calculate_metrics(predictions, true_class):
    precision = precision_score(true_class, predictions,zero_division=0)
    recall = recall_score(true_class, predictions)
    f1 = f1_score(true_class, predictions)
    accuracy = accuracy_score(true_class, predictions)
    return  precision, recall, accuracy, f1

def update_original(axes,threshold = 0.5):
    data, true_class, proba, predictions = calculate_proba(x, threshold)
    df_cm = calculate_matrix(predictions, true_class)
    
    axes[0][1].clear()
    axes[1][0].clear()
    sn.heatmap(df_cm, annot=True, ax=axes[0][1],cbar=False, cmap='Purples')
    threshold_line.set_ydata()
    metrics = calculate_metrics(predictions, true_class)
    
    axes[1][0].barh(['Precision','Recall'],metrics)
    
    fig.canvas.draw_idle()
  
def create_plot(threshold = 0.5):
    
    fig, axes = plt.subplots(2,2, figsize=(10, 10))
    draw_plot(axes, threshold)
    
    return fig, axes

def separate_classes(data, true_class, predictions, proba):
    all_data = zip(data, true_class, predictions, proba)
    
    tp = []
    tn = []
    fp = []
    fn = []
    
    for d in all_data:
        if d[1] == d[2] and d[1] == 0:
            tn.append(d) 
        if d[1] == d[2] and d[1] == 1:
            tp.append(d) 
        if d[1] != d[2] and d[1] == 1:
            fn.append(d) 
        if d[1] != d[2] and d[1] == 0:
            fp.append(d) 
    
    return {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn}
        

def draw_plot(axes, threshold):
    x = np.linspace(-6, 6)
    data, true_class, proba, predictions = calculate_proba(x, threshold)
    label_font = {'size':'12', 'weight':'bold'}
    
    classes = separate_classes(data, true_class, predictions, proba)
    
    axes[0][0].axhspan(-.18, threshold, facecolor='dimgrey', alpha=0.2)
    axes[0][0].axhspan(threshold, 1.18, facecolor='powderblue', alpha=0.2)
    
    sigmoid_curve, = axes[0][0].plot(x, sigmoid(x))
    threshold_line, = axes[0][0].plot(x, np.ones(len(x))*threshold, c='r', linestyle='--')
    axes[0][0].set_xlim(-6,6)
    axes[0][0].set_ylim(-.18,1.18)
    axes[0][0].set_xlabel("Feature Space", fontdict=label_font)
    axes[0][0].set_ylabel("Probability", fontdict=label_font)
    

    
    
    axes[0][0].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    
    #axes[0][0] = axes[0][0].scatter(data, sigmoid(data), marker = symbols, s=400, c=true_class)
    
    colors = {'tp':'darkorchid', 'tn':'mediumpurple', 'fp':'lightcoral', 'fn':'lightsalmon'}
    
    for k, v in classes.items():
        if k == 'tp' or k == 'fn':
            m = 's'
            s = 300
        else:
            m = 'o'
            s = 400
        c = colors.get(k)
        
        values = np.array([d[0] for d in v])
        
        axes[0][0].scatter(values,sigmoid(values), marker=m,s=s, color=c)
    
    precision, recall, accuracy, f1 = calculate_metrics(predictions, true_class)
        
    axes[1][0].barh(['Precision','Recall'],[precision, recall], color=['darkgreen','coral'])
    axes[1][1].barh(['Accuracy','F1-Score'],[accuracy, f1], color=['mediumturquoise','plum'])
    axes[1][1].yaxis.set_label_position("right")
    axes[1][1].yaxis.tick_right()
    
    axes[1][1].annotate(  f'{f1:.02}', (f1/2, 1), size=12, weight='bold', color='white')
    axes[1][1].annotate(f'{accuracy:.01%}', (accuracy/2, 0),size=12, weight='bold', color='white')
    axes[1][0].annotate(  f'{precision:.01%}', (precision/2, 0), size=12, weight='bold', color='white')
    axes[1][0].annotate(f'{recall:.01%}', (recall/2, 1),size=12, weight='bold', color='white')
    
    axes[1][0].set_xlim(0,1)
    axes[1][1].set_xlim(0,1)
    df_cm, annotations = calculate_matrix(predictions, true_class)
    
    #create a discrete color mapping
    colors = ['darkorchid', 'lightcoral','lightsalmon','mediumpurple']
    levels = [0,1,2,3,4]
    cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors)

    cfm = sns.heatmap(df_cm,square=True, annot=annotations,fmt='', 
                      ax=axes[0][1],cbar=False, linewidth=.5, linecolor='k',
                      annot_kws={"size": 14,'weight':'bold'}, cmap=cmap)
    
    axes[0][1].set_xlabel('Predicted labels', fontdict=label_font);
    axes[0][1].set_ylabel('Actual labels', fontdict=label_font);
    
    
    
    
    
    return None

def update_plot(axes, fig, threshold = 0.5):
    
    for ax in axes.flatten():
        ax.clear()
    
    draw_plot(axes, threshold)
    
    fig.canvas.draw_idle()
    


def draw_roc_plot(select=5, auc=False, compare=False):
    
    fpr= np.array([0,.03,.06,.11,.2,.3,.5,1])
    tpr =np.array([0,.3,.55,.8,.9,.95,.98,1])
    ratio = tpr[1:]/fpr[1:]
    text_y_pos = [0,0.01,.2, .35, .4, .45, .6, .5]
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    
    if auc or compare:
        pts = np.array(list(zip(fpr,tpr))+[[1,0]])
        p = Polygon(pts, closed=False, color='gold', alpha=.2)
        ax.add_patch(p)
        ax.annotate(  f'AUC = 0.79', (.4, 0.93), size=20, weight='bold', color='maroon');
    else:
        ax.plot(np.ones(5)*fpr[select], np.linspace(0,tpr[select],5), color='gold',alpha=.2, linestyle='-', linewidth=40);
        ax.annotate(  f'TPR/FPR: {ratio[select-1]:.01f}', (fpr[select]-.01,text_y_pos[select]), size=16,rotation=90, weight='bold', color='maroon');
    
    if compare:
        fpr_b= np.array([0,.031,.09,.15,.23,.4,.6,1])
        tpr_b =np.array([0,.2,.5,.6,.71,.85,.91,1])
        pts = np.array(list(zip(fpr_b,tpr_b))+[[1,0]])
        p = Polygon(pts, closed=False, color='darkgreen', alpha=.5)
        ax.add_patch(p)
        ax.annotate(  f'AUC = 0.68', (.4, 0.75), size=20, weight='bold', color='white');
        ax.plot(fpr_b, tpr_b, color='b');
        ax.scatter(fpr_b, tpr_b, color='b', s=180);
        
        
        
    ax.plot(fpr, tpr, color='k');
    ax.scatter(fpr, tpr, color='k', s=180);
    ax.plot(np.linspace(0,1,5), np.linspace(0,1,5), color='r', linestyle='--');
    label_font = {'size':'12', 'weight':'bold'}
    ax.set_xlabel('False Positive Rate\n(1-Specificity)',fontdict=label_font);
    ax.set_ylabel('True Positive Rate\n(Sensitivity/Recall)',fontdict=label_font);
    ax.set_xlim(0, 1.02);
    ax.set_ylim(0, 1.02);
    return None  