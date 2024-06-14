import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

class GradientDescent:
    def __init__(self, initial_beta0, eta, nb_epochs, beta1, X, y, skmodel_intercept, skmodel_slope):
        self.beta0 = initial_beta0
        self.X = X
        self.y = y
        self.beta1=beta1
        self.beta0-initial_beta0
        self.eta = eta
        self.nb_epochs = nb_epochs
        self.fig, axes = plt.subplots(1,2, figsize=(16,7))
    
        self.grad_ax = axes[0]
        self.reg_ax = axes[1]
        
        self.skmodel_intercept = skmodel_intercept
        self.skmodel_slope = skmodel_slope
        self.gradients = []
        self.losses = []
        self.betazeros = []
        
        self.train()
        
        return None
    
    def train(self):
        
        for epoch in range(self.nb_epochs):
            gradient = self.compute_gradient()
            self.gradients.append(gradient)
            
            loss = self.compute_loss()
            self.losses.append(loss)
            
            self.betazeros.append(self.beta0)
            self.beta0 = self.beta0-self.eta*gradient
      
        return None
    
    def compute_loss(self, beta0=None):
        if beta0 is None:
            beta0 = self.beta0
        loss = mean_squared_error(self.y,self.beta1*self.X+beta0)
        return loss
        
    def plot_loss(self):
        ax = self.grad_ax
        losses = [self.compute_loss(i) for i in np.linspace(-2, 6, 100)]
        ax.plot(np.linspace(-2, 6, 100), losses, c='teal', linewidth=6, label='MSE Loss')
        ax.set_ylabel('MSE Loss', size = 14)
        ax.set_xlabel(r'$\beta_{0}$ (Intercept)', size = 14)
        ax.set_xlim(-2,6)
        ax.set_ylim(-1,15)
        ax.legend()
    
        return None

    def plot_reg(self):
        ax = self.reg_ax
        ax.set_ylabel('Fish Height', size = 14)
        ax.set_xlabel(r'Normalized Fish Width', size = 14)
        ax.set_xlim(0,1)
        ax.set_ylim(-2,14)
        plt.scatter(self.X, self.y, c='blue', alpha=.2)
        ax.plot(np.linspace(0,1,10), self.skmodel_intercept+self.skmodel_slope*np.linspace(0,1,10), c='purple',
                linewidth=4, alpha=0.5,linestyle='-.', 
                 label='Sklearn Linear Model')
        
        self.text = ax.annotate(text=f' Sklearn Intercept:{self.skmodel_intercept:+.02f}\n GD Intercept:{self.beta0:+.02f}',
                    xy=(.4,1.1), color='blue')
        #ax.scatter([0], [self.skmodel_intercept], c='purple', marker='X', s = 400, label='Sklearn Intercept')
        ax.legend()
    
        return None
    
    def compute_gradient(self):
        return np.mean(-2*(self.y-self.beta1*self.X.squeeze()-self.beta0))
        
    def plot_gradient(self):
        ax = self.grad_ax
        gradient = self.compute_gradient()
        loss = self.compute_loss()
        min_grad = loss+(gradient*(-2))
        max_grad = loss+(gradient*(+2)) 
        self.grad_line,  = ax.plot(np.linspace(self.beta0-5,self.beta0+5, 3),np.linspace(min_grad,max_grad,3), c='r', linewidth=2, label='Gradient')
        ax.legend()

        return None

    def plot_reg_line(self):
        ax = self.reg_ax
        values = np.linspace(0,1,10)
        y_pred = self.beta1*values+self.beta0
        self.model, = ax.plot(values, y_pred, c='r', linewidth=2, linestyle='--', label='Gradient Descent Approximation')
        ax.legend()
    
        return self
    
    def regression_vs_gradient(self):
        self.plot_loss()
        self.plot_reg()
        
        self.plot_gradient();
        self.plot_reg_line();
    
        return self
    
    def update_epoch(self, epoch):
        delta_beta0=2
        gradient = self.gradients[epoch]
        loss = self.losses[epoch]
        beta0 = self.betazeros[epoch]
        min_grad = loss-(gradient*delta_beta0)
        max_grad = loss+(gradient*delta_beta0)
        y_pred = self.skmodel_slope*np.linspace(0,1,10)+beta0
        
        updated_text=f' Sklearn Intercept:{self.skmodel_intercept:+.02f}\n Gradient Descent Intercept:{beta0:+.02f}'
        
        self.text.set(text=updated_text)
        
        #grad_point = ax.scatter([beta0], [loss], c='r', marker='o', s = 200)
    
        self.grad_line.set_data([beta0-delta_beta0,beta0,beta0+delta_beta0],[min_grad,loss,max_grad]) 
        self.grad_ax.scatter(self.betazeros[:epoch+1],self.losses[:epoch+1],c='r', marker='o', alpha=.3, s = 200)
    
        self.model.set_ydata(y_pred)
        

        
        self.fig.canvas.draw_idle()
        
        return None