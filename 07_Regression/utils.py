import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.regression.linear_model as sm
from statsmodels.graphics.gofplots import ProbPlot

#   Function to reproduce the 4 base plots of an OLS model in R
#   -----------------------------------------------------------
#   Source: https://robert-alvarez.github.io/2018-06-04-diagnostic_plots/
#   Author: Robert Alvarez
#   Date: 04.06.2018
def graph(formula, x_range, label=None):
    """
    Helper function for plotting Cook's distance lines
    """
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

def diagnostic_plots(X, y, model_fit=None):
    """
    Function to reproduce the 4 base plots of an OLS model in R.
    ---
    Inputs:
    X: A numpy array or pandas dataframe of the features to use in 
       building the linear regression model
    y: A numpy array or pandas series/dataframe of the target variable 
       of the linear regression model
    model_fit [optional]: a statsmodel.api.OLS model after regressing y on X. 
                          If not provided, will be generated from X, y
    """
    if not model_fit:
        model_fit = sm.OLS(y, sm.add_constant(X)).fit()
    
    # create dataframe from X, y for easier plot handling
    dataframe = pd.concat([X, y], axis=1)
    # model values
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]
    #-----------------------
    #   Tukey-Anscombe plot
    plot_lm_1 = plt.figure()
    plot_lm_1.axes[0] = sns.residplot(x=model_fitted_y, 
                                      y=dataframe.columns[-1], 
                                      data=dataframe, 
                                      lowess=True, 
                                      color='blue', 
                                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals');
    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    for i in abs_resid_top_3.index:
        plot_lm_1.axes[0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]));
    #----------------------
    #   Scale-location plot
    plot_lm_2 = plt.figure()
    plt.scatter(x=model_fitted_y, y=model_norm_residuals_abs_sqrt, color='blue');
    sns.regplot(x=model_fitted_y, 
                y=model_norm_residuals_abs_sqrt, 
                scatter=False,
                ci=False, 
                lowess=True, 
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
    plot_lm_2.axes[0].set_title('Scale-Location')
    plot_lm_2.axes[0].set_xlabel('Fitted values')
    plot_lm_2.axes[0].set_ylabel('$\sqrt{|Standardized~~Residuals|}$');
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for i in abs_norm_resid_top_3:
        plot_lm_2.axes[0].annotate(i, xy=(model_fitted_y[i], 
                                          model_norm_residuals_abs_sqrt[i]));
    #-----------
    #   q-q plot
    QQ = ProbPlot(model_norm_residuals)
    plot_lm_3 = QQ.qqplot(line='45', markerfacecolor='b', markeredgecolor='b')
    plot_lm_3.axes[0].set_title('Normal Q-Q')
    plot_lm_3.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_3.axes[0].set_ylabel('Standardized Residuals');
    # annotations
    for r, i in enumerate(abs_norm_resid_top_3):
        plot_lm_3.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], 
                                          model_norm_residuals[i]));
    #------------------------
    #   Residuals vs Leverage
    plot_lm_4 = plt.figure();
    plt.scatter(x=model_leverage, y=model_norm_residuals, color='blue');
    sns.regplot(x=model_leverage, 
                y=model_norm_residuals,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
    plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)
    plot_lm_4.axes[0].set_ylim(-3, 5)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals');
    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    for i in leverage_top_3:
        plot_lm_4.axes[0].annotate(i, xy=(model_leverage[i], 
                                          model_norm_residuals[i]));
    
    p = len(model_fit.params) # number of model parameters
    # 0.5 line
    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
          np.linspace(0.001, 
                      max(model_leverage), 50), 'Cook\'s distance')
    # 1 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),  
          np.linspace(0.001, max(model_leverage), 50)) 
    plot_lm_4.legend(loc='upper right');
