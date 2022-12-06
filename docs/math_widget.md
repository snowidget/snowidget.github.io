## Widget to plot prob density estimate provided a distribution and smoothing param
Adaoted from: https://jakevdp.github.io/blog/2013/12/05/static-interactive-widgets/#SymPy-Matha


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, FloatSlider, widgets, interactive
from IPython.display import display
from sklearn.neighbors import KernelDensity
```


```python
# tee up density estimate
np.random.seed(100)
x = np.concatenate([np.random.normal(0, 1, 1000), 
                   np.random.normal(1.5, 0.2, 300)])

def plot_KDE_estimate(kernel, b):
    bandwidth = 10 ** (0.1 * b)
    x_grid = np.linspace(-3, 3, 1000)
    kde = KernelDensity(bandwidth=bandwidth,
                        kernel=kernel)
    kde.fit(x[:, None])
    pdf = np.exp(kde.score_samples(x_grid[:, None]))
    
    fig, ax = plt.subplots(figsize=(4, 3),
                           subplot_kw={'fc':'#EEEEEE',
                                       'axisbelow':True})
    ax.grid(color='w', linewidth=2, linestyle='solid')
    ax.hist(x, 60, histtype='stepfilled', density=True,
            edgecolor='none', facecolor='#CCCCFF')
    ax.plot(x_grid, pdf, '-k', lw=2, alpha=0.5)
    ax.text(-2.8, 0.48,
            "kernel={0}\nbandwidth={1:.2f}".format(kernel, bandwidth),
            fontsize=14, color='gray')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 0.601)
    
    return fig
```


```python
w = interactive(plot_KDE_estimate, 
        kernel = widgets.RadioButtons(options=['gaussian', 'tophat', 'exponential', 'linear','cosine']), 
        b = widgets.IntSlider(min=-14, max=8, step=2));
```


```python
display(w)
```


    interactive(children=(RadioButtons(description='kernel', options=('gaussian', 'tophat', 'exponential', 'linear…

