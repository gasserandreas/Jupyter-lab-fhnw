#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Part II: Multiple Regression
#   'Example 11.0.1+11.1.1 Construction Costs of Nuclear Power Plants, AIC, adjR2, Cp.py'
#   Author: Marcel Steiner-Curtis
#   Date: 06.09.2018    sml
#         07.03.2021    version  for Python
#         13.02.2023    sml: minor changes
#--------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.formula.api as smf
import statsmodels.regression.linear_model as sm

#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Automated Stepwise Backward and Forward Selection
#   -------------------------------------------------
#   This script is about an automated stepwise backward and forward feature selection. You can easily apply on Dataframes.
#   Functions returns not only the final features but also elimination iterations, so you can track what exactly happend at the iterations.
#   You can apply it on both Linear and Logistic problems. Eliminations can be apply with Akaike information criterion (AIC), 
#   Bayesian information criterion (BIC), R-squared (Only works with linear), Adjusted R-squared (Only works with linear). 
#   Also you don't have to worry about varchar variables, code will handle it for you.
#   Required Libraries: pandas, numpy, statmodels
#   Source: https://github.com/talhahascelik/python_stepwiseSelection
#   Author: Sinan Talha Hascelik
#   Date: 22.04.2019
def forwardSelection(X, y, model_type="linear", elimination_criteria="aic", varchar_process="dummy_dropfirst", sl=0.05):
    """
    Forward Selection is a function, based on regression models, that returns significant features and selection iterations.\n
    Required Libraries: pandas, numpy, statmodels
    
    Parameters
    ----------
    X : Independent variables (Pandas Dataframe)\n
    y : Dependent variable (Pandas Series, Pandas Dataframe)\n
    model_type : 'linear' or 'logistic'\n
    elimination_criteria : 'aic', 'bic', 'r2', 'adjr2' or None\n
                           'aic' refers Akaike information criterion\n
                           'bic' refers Bayesian information criterion\n
                           'r2' refers R-squared (Only works on linear model type)\n
                           'r2' refers Adjusted R-squared (Only works on linear model type)\n
    varchar_process : 'drop', 'dummy' or 'dummy_dropfirst'\n
                      'drop' drops varchar features\n
                      'dummy' creates dummies for all levels of all varchars\n
                      'dummy_dropfirst' creates dummies for all levels of all varchars, and drops first levels\n
    sl : Significance Level (default: 0.05)\n
    
    Returns
    -------
    columns(list), iteration_logs(str)\n\n
    Not Returns a Model
    
    Tested On
    ---------
    Python v3.6.7, Pandas v0.23.4, Numpy v1.15.04, StatModels v0.9.0
    
    See Also
    --------
    https://en.wikipedia.org/wiki/Stepwise_regression
    """
    X = __varcharProcessing__(X,varchar_process = varchar_process)
    return __forwardSelectionRaw__(X, y, model_type = model_type,elimination_criteria = elimination_criteria , sl=sl)
    
def backwardSelection(X, y, model_type="linear", elimination_criteria="aic", varchar_process="dummy_dropfirst", sl=0.05):
    """
    Backward Selection is a function, based on regression models, that returns significant features and selection iterations.\n
    Required Libraries: pandas, numpy, statmodels
    
    Parameters
    ----------
    X : Independent variables (Pandas Dataframe)\n
    y : Dependent variable (Pandas Series, Pandas Dataframe)\n
    model_type : 'linear' or 'logistic'\n
    elimination_criteria : 'aic', 'bic', 'r2', 'adjr2' or None\n
                           'aic' refers Akaike information criterion\n
                           'bic' refers Bayesian information criterion\n
                           'r2' refers R-squared (Only works on linear model type)\n
                           'r2' refers Adjusted R-squared (Only works on linear model type)\n
    varchar_process : 'drop', 'dummy' or 'dummy_dropfirst'\n
                      'drop' drops varchar features\n
                      'dummy' creates dummies for all levels of all varchars\n
                      'dummy_dropfirst' creates dummies for all levels of all varchars, and drops first levels\n
    sl : Significance Level (default: 0.05)\n
    
    Returns
    -------
    columns(list), iteration_logs(str)\n\n
    Not Returns a Model
    
    Tested On
    ---------
    Python v3.6.7, Pandas v0.23.4, Numpy v1.15.04, StatModels v0.9.0
    
    See Also
    --------
    https://en.wikipedia.org/wiki/Stepwise_regression    
    """
    X = __varcharProcessing__(X,varchar_process = varchar_process)
    return __backwardSelectionRaw__(X, y, model_type = model_type,elimination_criteria = elimination_criteria , sl=sl)

def __varcharProcessing__(X, varchar_process = "dummy_dropfirst"):
    
    dtypes = X.dtypes
    if varchar_process == "drop":   
        X = X.drop(columns = dtypes[dtypes == object].index.tolist())
        print("Character Variables (Dropped):", dtypes[dtypes == object].index.tolist())
    elif varchar_process == "dummy":
        X = pd.get_dummies(X,drop_first=False)
        print("Character Variables (Dummies Generated):", dtypes[dtypes == object].index.tolist())
    elif varchar_process == "dummy_dropfirst":
        X = pd.get_dummies(X,drop_first=True)
        print("Character Variables (Dummies Generated, First Dummies Dropped):", dtypes[dtypes == object].index.tolist())
    else: 
        X = pd.get_dummies(X,drop_first=True)
        print("Character Variables (Dummies Generated, First Dummies Dropped):", dtypes[dtypes == object].index.tolist())
    
    X["intercept"] = 1
    cols = X.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X = X[cols]
    
    return X

def __forwardSelectionRaw__(X, y, model_type ="linear", elimination_criteria = "aic", sl=0.05):

    iterations_log = ""
    cols = X.columns.tolist()
    
    def regressor(y, X, model_type=model_type):
        if model_type == "linear":
            regressor = sm.OLS(y, X).fit()
        elif model_type == "logistic":
            regressor = sm.Logit(y, X).fit()
        else:
            print("\nWrong Model Type : "+ model_type +"\nLinear model type is seleted.")
            model_type = "linear"
            regressor = sm.OLS(y, X).fit()
        return regressor
    
    selected_cols = ["intercept"]
    other_cols = cols.copy()
    other_cols.remove("intercept")
    
    model = regressor(y, X[selected_cols])
    
    if elimination_criteria == "aic":
        criteria = model.aic
    elif elimination_criteria == "bic":
        criteria = model.bic
    elif elimination_criteria == "r2" and model_type =="linear":
        criteria = model.rsquared
    elif elimination_criteria == "adjr2" and model_type =="linear":
        criteria = model.rsquared_adj
    
    for i in range(X.shape[1]):
        pvals = pd.DataFrame(columns = ["Cols","Pval"])
        for j in other_cols:
            model = regressor(y, X[selected_cols+[j]])
            pvals = pd.concat([pvals, pd.DataFrame([[j, model.pvalues[j]]], columns = ["Cols","Pval"])])
        pvals = pvals.sort_values(by = ["Pval"]).reset_index(drop=True)
        pvals = pvals[pvals.Pval<=sl]
        if pvals.shape[0] > 0:
            
            model = regressor(y, X[selected_cols+[pvals["Cols"][0]]])
            iterations_log += str("\nEntered : "+pvals["Cols"][0] + "\n")    
            iterations_log += "\n\n"+str(model.summary())+"\nAIC: "+ str(model.aic) + "\nBIC: "+ str(model.bic)+"\n\n"
                    
        
            if  elimination_criteria == "aic":
                new_criteria = model.aic
                if new_criteria < criteria:
                    print("Entered :", pvals["Cols"][0], "\tAIC :", model.aic)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("break : Criteria")
                    break
            elif  elimination_criteria == "bic":
                new_criteria = model.bic
                if new_criteria < criteria:
                    print("Entered :", pvals["Cols"][0], "\tBIC :", model.bic)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("break : Criteria")
                    break        
            elif  elimination_criteria == "r2" and model_type =="linear":
                new_criteria = model.rsquared
                if new_criteria > criteria:
                    print("Entered :", pvals["Cols"][0], "\tR2 :", model.rsquared)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("break : Criteria")
                    break           
            elif  elimination_criteria == "adjr2" and model_type =="linear":
                new_criteria = model.rsquared_adj
                if new_criteria > criteria:
                    print("Entered :", pvals["Cols"][0], "\tAdjR2 :", model.rsquared_adj)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("Break : Criteria")
                    break
            else:
                print("Entered :", pvals["Cols"][0])
                selected_cols.append(pvals["Cols"][0])
                other_cols.remove(pvals["Cols"][0])            
                
        else:
            print("Break : Significance Level")
            break
        
    model = regressor(y, X[selected_cols])
    if elimination_criteria == "aic":
        criteria = model.aic
    elif elimination_criteria == "bic":
        criteria = model.bic
    elif elimination_criteria == "r2" and model_type =="linear":
        criteria = model.rsquared
    elif elimination_criteria == "adjr2" and model_type =="linear":
        criteria = model.rsquared_adj
    
    print(model.summary())
    print("AIC: "+str(model.aic))
    print("BIC: "+str(model.bic))
    print("Final Variables:", selected_cols)

    return selected_cols, iterations_log

def __backwardSelectionRaw__(X, y, model_type ="linear",elimination_criteria = "aic", sl=0.05):
    
    iterations_log = ""
    last_eleminated = ""    
    cols = X.columns.tolist()

    def regressor(y, X, model_type=model_type):
        if model_type =="linear":
            regressor = sm.OLS(y, X).fit()
        elif model_type == "logistic":
            regressor = sm.Logit(y, X).fit()
        else:
            print("\nWrong Model Type : "+ model_type +"\nLinear model type is seleted.")
            model_type = "linear"
            regressor = sm.OLS(y, X).fit()
        return regressor
    for i in range(X.shape[1]):
        if i != 0 :          
            if elimination_criteria == "aic":
                criteria = model.aic
                new_model = regressor(y,X)
                new_criteria = new_model.aic
                if criteria < new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n"+str(new_model.summary())+"\nAIC: "+ str(new_model.aic) + "\nBIC: "+ str(new_model.bic)+"\n"
                    iterations_log += str("\n\nRegained : "+last_eleminated + "\n\n")
                    break  
            elif elimination_criteria == "bic":
                criteria = model.bic
                new_model = regressor(y, X)
                new_criteria = new_model.bic
                if criteria < new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n"+str(new_model.summary())+"\nAIC: "+ str(new_model.aic) + "\nBIC: "+ str(new_model.bic)+"\n"
                    iterations_log += str("\n\nRegained : "+last_eleminated + "\n\n")
                    break  
            elif elimination_criteria == "adjr2" and model_type =="linear":
                criteria = model.rsquared_adj
                new_model = regressor(y, X)
                new_criteria = new_model.rsquared_adj
                if criteria > new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n"+str(new_model.summary())+"\nAIC: "+ str(new_model.aic) + "\nBIC: "+ str(new_model.bic)+"\n"
                    iterations_log += str("\n\nRegained : "+last_eleminated + "\n\n")
                    break  
            elif elimination_criteria == "r2" and model_type =="linear":
                criteria = model.rsquared
                new_model = regressor(y, X)
                new_criteria = new_model.rsquared
                if criteria > new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n"+str(new_model.summary())+"\nAIC: "+ str(new_model.aic) + "\nBIC: "+ str(new_model.bic)+"\n"
                    iterations_log += str("\n\nRegained : "+last_eleminated + "\n\n")
                    break   
            else: 
                new_model = regressor(y,X)
            model = new_model
            iterations_log += "\n"+str(model.summary())+"\nAIC: "+ str(model.aic) + "\nBIC: "+ str(model.bic)+"\n"
        else:
            model = regressor(y,X)
            iterations_log += "\n"+str(model.summary())+"\nAIC: "+ str(model.aic) + "\nBIC: "+ str(model.bic)+"\n"
        maxPval = max(model.pvalues)
        cols = X.columns.tolist()
        if maxPval > sl:
            for j in cols:
                if (model.pvalues[j] == maxPval):
                    print("Eliminated :" ,j)
                    iterations_log += str("\n\nEliminated : "+j+ "\n\n")
                    
                    del X[j]
                    last_eleminated = j
        else:
            break
    print(str(model.summary())+"\nAIC: "+ str(model.aic) + "\nBIC: "+ str(model.bic))
    print("Final Variables:", cols)
    iterations_log += "\n"+str(model.summary())+"\nAIC: "+ str(model.aic) + "\nBIC: "+ str(model.bic)+"\n"
    return cols, iterations_log

#--------------------------------------------------------------------------------------------------------------------------------------------------
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
    X: A numpy array or pandas dataframe of the features to use in building the linear regression model
    y: A numpy array or pandas series/dataframe of the target variable of the linear regression model
    
    model_fit [optional]: a statsmodel.api.OLS model after regressing y on X. If not provided, will be
                          generated from X, y
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
    #----------------------
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
    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, color='blue');
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
        plot_lm_2.axes[0].annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]));
    #-----------
    #   q-q plot
    QQ = ProbPlot(model_norm_residuals)
    plot_lm_3 = QQ.qqplot(line='45', markerfacecolor='b', markeredgecolor='b')
    plot_lm_3.axes[0].set_title('Normal Q-Q')
    plot_lm_3.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_3.axes[0].set_ylabel('Standardized Residuals');
    # annotations
    for r, i in enumerate(abs_norm_resid_top_3):
        plot_lm_3.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], model_norm_residuals[i]));
    #------------------------
    #   Residuals vs Leverage
    plot_lm_4 = plt.figure();
    plt.scatter(model_leverage, model_norm_residuals, color='blue');
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
        plot_lm_4.axes[0].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]));
    
    p = len(model_fit.params) # number of model parameters
    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), np.linspace(0.001, max(model_leverage), 50), 'Cook\'s distance') # 0.5 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),  np.linspace(0.001, max(model_leverage), 50)) # 1 line
    plot_lm_4.legend(loc='upper right');
    

#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Example 11.1.1 Construction Costs of Nuclear Power Plants (NPP)
#--------------------------------------------------------------------------------------------------------------------------------------------------
#   read data
data = pd.read_csv('../04 Datasets/NPP.dat', delimiter='\t')
data.head()
data.columns.values

#   define new transformed variables and delete the original variables afterwards
data['Clog']  = np.log10(data.C)
data['Slog']  = np.log10(data.S)
data['Nsqrt'] =  np.sqrt(data.N)
data = data.drop(columns=['C', 'S', 'N'])
data.columns.values
#   array(['D', 'T1', 'T2', 'PR', 'NE', 'CT', 'BW', 'PT', 'Clog', 'Slog', 'Nsqrt'], dtype=object)

#   order columns as in original data set
data = data.reindex(['Clog', 'D', 'T1', 'T2', 'Slog', 'PR', 'NE', 'CT', 'BW', 'Nsqrt', 'PT'], axis=1)
data.columns.values
#   array(['Clog', 'D', 'T1', 'T2', 'Slog', 'PR', 'NE', 'CT', 'BW', 'Nsqrt', 'PT'], dtype=object)

#   Dependent and Independent Variables
X = data.drop(columns='Clog')
y = data.Clog


#--------------------------------------------------------------------------------------------------------------------------------------------------
#  pairs plot
sns.pairplot(data, hue='PT')


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Linear regression with full model
modfull = smf.ols(formula='Clog ~ D + T1 + T2 + Slog + PR + NE + CT + BW + Nsqrt + PT', data=data).fit()
print(modfull.summary())
#                                OLS Regression Results                            
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.868
#    Model:                            OLS   Adj. R-squared:                  0.806
#    Method:                 Least Squares   F-statistic:                     13.85
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           3.98e-07
#    Time:                        17:06:15   Log-Likelihood:                 45.372
#    No. Observations:                  32   AIC:                            -68.74
#    Df Residuals:                      21   BIC:                            -52.62
#    Df Model:                          10                                         
#    Covariance Type:            nonrobust                                         
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    Intercept     -6.0259      2.347     -2.567      0.018     -10.907      -1.144
#    D              0.0952      0.036      2.661      0.015       0.021       0.170
#    T1             0.0026      0.010      0.276      0.785      -0.017       0.022
#    T2             0.0023      0.002      1.155      0.261      -0.002       0.006
#    Slog           0.6925      0.137      5.050      0.000       0.407       0.978
#    PR            -0.0457      0.036     -1.284      0.213      -0.120       0.028
#    NE             0.1105      0.034      3.257      0.004       0.040       0.181
#    CT             0.0534      0.030      1.798      0.087      -0.008       0.115
#    BW             0.0128      0.045      0.282      0.781      -0.082       0.107
#    Nsqrt         -0.0300      0.018     -1.684      0.107      -0.067       0.007
#    PT            -0.0995      0.056     -1.789      0.088      -0.215       0.016
#    ==============================================================================
#    Omnibus:                        0.947   Durbin-Watson:                   2.318
#    Prob(Omnibus):                  0.623   Jarque-Bera (JB):                0.831
#    Skew:                          -0.368   Prob(JB):                        0.660
#    Kurtosis:                       2.715   Cond. No.                     1.73e+04
#    ==============================================================================
#    
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 1.73e+04. This might indicate that there are
#    strong multicollinearity or other numerical problems.


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Linear regression trivial model
modsmall = smf.ols(formula='Clog ~ 1', data=data).fit()
print(modsmall.summary())
#                                OLS Regression Results                            
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.000
#    Model:                            OLS   Adj. R-squared:                  0.000
#    Method:                 Least Squares   F-statistic:                       nan
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):                nan
#    Time:                        17:09:01   Log-Likelihood:                 12.927
#    No. Observations:                  32   AIC:                            -23.85
#    Df Residuals:                      31   BIC:                            -22.39
#    Df Model:                           0                                         
#    Covariance Type:            nonrobust                                         
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    Intercept      2.6349      0.029     90.810      0.000       2.576       2.694
#    ==============================================================================
#    Omnibus:                        1.935   Durbin-Watson:                   1.378
#    Prob(Omnibus):                  0.380   Jarque-Bera (JB):                1.184
#    Skew:                          -0.126   Prob(JB):                        0.553
#    Kurtosis:                       2.092   Cond. No.                         1.00
#    ==============================================================================
#    
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   2. Akaike information criterion
#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Choose a model by AIC in a stepwise algorithm (backward)
final_vars, iterations_logs = backwardSelection(X, y, model_type='linear', elimination_criteria='aic')
#    Character Variables (Dummies Generated, First Dummies Dropped): []
#    Eliminated : T1
#    Eliminated : BW
#    Eliminated : PR
#    Regained :  PR
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.868
#    Model:                            OLS   Adj. R-squared:                  0.822
#    Method:                 Least Squares   F-statistic:                     18.85
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           2.19e-08
#    Time:                        17:54:06   Log-Likelihood:                 45.288
#    No. Observations:                  32   AIC:                            -72.58
#    Df Residuals:                      23   BIC:                            -59.38
#    Df Model:                           8
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept     -6.5221      1.522     -4.285      0.000      -9.671      -3.374
#    D              0.1032      0.021      4.901      0.000       0.060       0.147
#    T2             0.0024      0.002      1.433      0.165      -0.001       0.006
#    Slog           0.6856      0.130      5.282      0.000       0.417       0.954
#    PR            -0.0452      0.032     -1.433      0.165      -0.111       0.020
#    NE             0.1112      0.032      3.432      0.002       0.044       0.178
#    CT             0.0518      0.028      1.852      0.077      -0.006       0.110
#    Nsqrt         -0.0297      0.016     -1.804      0.084      -0.064       0.004
#    PT            -0.0939      0.051     -1.855      0.076      -0.199       0.011
#    ==============================================================================
#    Omnibus:                        1.181   Durbin-Watson:                   2.273
#    Prob(Omnibus):                  0.554   Jarque-Bera (JB):                0.954
#    Skew:                          -0.408   Prob(JB):                        0.621
#    Kurtosis:                       2.781   Cond. No.                     1.16e+04
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 1.16e+04. This might indicate that there are
#    strong multicollinearity or other numerical problems.
#    AIC: -72.5758960498367
#    BIC: -59.38427292463916
#    Final Variables: ['intercept', 'D', 'T2', 'Slog', 'PR', 'NE', 'CT', 'Nsqrt', 'PT']

print(iterations_logs)
#
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.868
#    Model:                            OLS   Adj. R-squared:                  0.806
#    Method:                 Least Squares   F-statistic:                     13.85
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           3.98e-07
#    Time:                        17:51:19   Log-Likelihood:                 45.372
#    No. Observations:                  32   AIC:                            -68.74
#    Df Residuals:                      21   BIC:                            -52.62
#    Df Model:                          10
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept     -6.0259      2.347     -2.567      0.018     -10.907      -1.144
#    D              0.0952      0.036      2.661      0.015       0.021       0.170
#    T1             0.0026      0.010      0.276      0.785      -0.017       0.022
#    T2             0.0023      0.002      1.155      0.261      -0.002       0.006
#    Slog           0.6925      0.137      5.050      0.000       0.407       0.978
#    PR            -0.0457      0.036     -1.284      0.213      -0.120       0.028
#    NE             0.1105      0.034      3.257      0.004       0.040       0.181
#    CT             0.0534      0.030      1.798      0.087      -0.008       0.115
#    BW             0.0128      0.045      0.282      0.781      -0.082       0.107
#    Nsqrt         -0.0300      0.018     -1.684      0.107      -0.067       0.007
#    PT            -0.0995      0.056     -1.789      0.088      -0.215       0.016
#    ==============================================================================
#    Omnibus:                        0.947   Durbin-Watson:                   2.318
#    Prob(Omnibus):                  0.623   Jarque-Bera (JB):                0.831
#    Skew:                          -0.368   Prob(JB):                        0.660
#    Kurtosis:                       2.715   Cond. No.                     1.73e+04
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 1.73e+04. This might indicate that there are
#    strong multicollinearity or other numerical problems.
#    AIC: -68.74319042137313
#    BIC: -52.620095490576134
#
#
#    Eliminated : T1
#
#
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.868
#    Model:                            OLS   Adj. R-squared:                  0.814
#    Method:                 Least Squares   F-statistic:                     16.06
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           9.82e-08
#    Time:                        17:51:19   Log-Likelihood:                 45.314
#    No. Observations:                  32   AIC:                            -70.63
#    Df Residuals:                      22   BIC:                            -55.97
#    Df Model:                           9
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept     -6.5016      1.559     -4.171      0.000      -9.734      -3.269
#    D              0.1030      0.022      4.781      0.000       0.058       0.148
#    T2             0.0022      0.002      1.150      0.262      -0.002       0.006
#    Slog           0.6872      0.133      5.172      0.000       0.412       0.963
#    PR            -0.0433      0.034     -1.282      0.213      -0.113       0.027
#    NE             0.1111      0.033      3.358      0.003       0.042       0.180
#    CT             0.0527      0.029      1.820      0.082      -0.007       0.113
#    BW             0.0076      0.040      0.188      0.852      -0.076       0.092
#    Nsqrt         -0.0305      0.017     -1.761      0.092      -0.066       0.005
#    PT            -0.0962      0.053     -1.810      0.084      -0.206       0.014
#    ==============================================================================
#    Omnibus:                        1.122   Durbin-Watson:                   2.300
#    Prob(Omnibus):                  0.571   Jarque-Bera (JB):                0.893
#    Skew:                          -0.396   Prob(JB):                        0.640
#    Kurtosis:                       2.793   Cond. No.                     1.16e+04
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 1.16e+04. This might indicate that there are
#    strong multicollinearity or other numerical problems.
#    AIC: -70.62739561420676
#    BIC: -55.970036586209496
#
#
#    Eliminated : BW
#
#
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.868
#    Model:                            OLS   Adj. R-squared:                  0.822
#    Method:                 Least Squares   F-statistic:                     18.85
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           2.19e-08
#    Time:                        17:51:19   Log-Likelihood:                 45.288
#    No. Observations:                  32   AIC:                            -72.58
#    Df Residuals:                      23   BIC:                            -59.38
#    Df Model:                           8
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept     -6.5221      1.522     -4.285      0.000      -9.671      -3.374
#    D              0.1032      0.021      4.901      0.000       0.060       0.147
#    T2             0.0024      0.002      1.433      0.165      -0.001       0.006
#    Slog           0.6856      0.130      5.282      0.000       0.417       0.954
#    PR            -0.0452      0.032     -1.433      0.165      -0.111       0.020
#    NE             0.1112      0.032      3.432      0.002       0.044       0.178
#    CT             0.0518      0.028      1.852      0.077      -0.006       0.110
#    Nsqrt         -0.0297      0.016     -1.804      0.084      -0.064       0.004
#    PT            -0.0939      0.051     -1.855      0.076      -0.199       0.011
#    ==============================================================================
#    Omnibus:                        1.181   Durbin-Watson:                   2.273
#    Prob(Omnibus):                  0.554   Jarque-Bera (JB):                0.954
#    Skew:                          -0.408   Prob(JB):                        0.621
#    Kurtosis:                       2.781   Cond. No.                     1.16e+04
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 1.16e+04. This might indicate that there are
#    strong multicollinearity or other numerical problems.
#    AIC: -72.5758960498367
#    BIC: -59.38427292463916
#
#
#    Eliminated : PR
#
#
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.856
#    Model:                            OLS   Adj. R-squared:                  0.814
#    Method:                 Least Squares   F-statistic:                     20.36
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           1.18e-08
#    Time:                        17:51:19   Log-Likelihood:                 43.920
#    No. Observations:                  32   AIC:                            -71.84
#    Df Residuals:                      24   BIC:                            -60.11
#    Df Model:                           7
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept     -6.0057      1.511     -3.975      0.001      -9.124      -2.888
#    D              0.0965      0.021      4.600      0.000       0.053       0.140
#    T2             0.0012      0.001      0.836      0.411      -0.002       0.004
#    Slog           0.6840      0.133      5.158      0.000       0.410       0.958
#    NE             0.1070      0.033      3.247      0.003       0.039       0.175
#    CT             0.0598      0.028      2.135      0.043       0.002       0.118
#    Nsqrt         -0.0281      0.017     -1.672      0.107      -0.063       0.007
#    PT            -0.1073      0.051     -2.113      0.045      -0.212      -0.002
#    ==============================================================================
#    Omnibus:                        0.668   Durbin-Watson:                   2.609
#    Prob(Omnibus):                  0.716   Jarque-Bera (JB):                0.460
#    Skew:                          -0.286   Prob(JB):                        0.794
#    Kurtosis:                       2.865   Cond. No.                     1.12e+04
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 1.12e+04. This might indicate that there are
#    strong multicollinearity or other numerical problems.
#    AIC: -71.83917791206872
#    BIC: -60.113290689670904
#
#
#    Regained : PR
#
#
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.868
#    Model:                            OLS   Adj. R-squared:                  0.822
#    Method:                 Least Squares   F-statistic:                     18.85
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           2.19e-08
#    Time:                        17:51:19   Log-Likelihood:                 45.288
#    No. Observations:                  32   AIC:                            -72.58
#    Df Residuals:                      23   BIC:                            -59.38
#    Df Model:                           8
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept     -6.5221      1.522     -4.285      0.000      -9.671      -3.374
#    D              0.1032      0.021      4.901      0.000       0.060       0.147
#    T2             0.0024      0.002      1.433      0.165      -0.001       0.006
#    Slog           0.6856      0.130      5.282      0.000       0.417       0.954
#    PR            -0.0452      0.032     -1.433      0.165      -0.111       0.020
#    NE             0.1112      0.032      3.432      0.002       0.044       0.178
#    CT             0.0518      0.028      1.852      0.077      -0.006       0.110
#    Nsqrt         -0.0297      0.016     -1.804      0.084      -0.064       0.004
#    PT            -0.0939      0.051     -1.855      0.076      -0.199       0.011
#    ==============================================================================
#    Omnibus:                        1.181   Durbin-Watson:                   2.273
#    Prob(Omnibus):                  0.554   Jarque-Bera (JB):                0.954
#    Skew:                          -0.408   Prob(JB):                        0.621
#    Kurtosis:                       2.781   Cond. No.                     1.16e+04
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 1.16e+04. This might indicate that there are
#    strong multicollinearity or other numerical problems.
#    AIC: -72.5758960498367
#    BIC: -59.38427292463916

final_vars
#   ['intercept', 'D', 'T2', 'Slog', 'PR', 'NE', 'CT', 'Nsqrt', 'PT']

#   fit reduced model again
modbackward = smf.ols(formula='Clog ~ D + T2 + Slog + PR + NE + CT + Nsqrt + PT', data=data).fit()
print(modbackward.summary())
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.868
#    Model:                            OLS   Adj. R-squared:                  0.822
#    Method:                 Least Squares   F-statistic:                     18.85
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           2.19e-08
#    Time:                        18:00:43   Log-Likelihood:                 45.288
#    No. Observations:                  32   AIC:                            -72.58
#    Df Residuals:                      23   BIC:                            -59.38
#    Df Model:                           8
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    Intercept     -6.5221      1.522     -4.285      0.000      -9.671      -3.374
#    D              0.1032      0.021      4.901      0.000       0.060       0.147
#    T2             0.0024      0.002      1.433      0.165      -0.001       0.006
#    Slog           0.6856      0.130      5.282      0.000       0.417       0.954
#    PR            -0.0452      0.032     -1.433      0.165      -0.111       0.020
#    NE             0.1112      0.032      3.432      0.002       0.044       0.178
#    CT             0.0518      0.028      1.852      0.077      -0.006       0.110
#    Nsqrt         -0.0297      0.016     -1.804      0.084      -0.064       0.004
#    PT            -0.0939      0.051     -1.855      0.076      -0.199       0.011
#    ==============================================================================
#    Omnibus:                        1.181   Durbin-Watson:                   2.273
#    Prob(Omnibus):                  0.554   Jarque-Bera (JB):                0.954
#    Skew:                          -0.408   Prob(JB):                        0.621
#    Kurtosis:                       2.781   Cond. No.                     1.16e+04
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 1.16e+04. This might indicate that there are
#    strong multicollinearity or other numerical problems.

#   diagnostic tools
diagnostic_plots(X, y, model_fit=modbackward)

#   REMARKS: The scale-location plot shows a strong increase which is not extrem compared to the simulated curves.
#            => Check residuals versus individual explanatory variables.


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Choose a model by AIC in a stepwise algorithm (forward)
final_vars, iterations_logs = forwardSelection(X, y, model_type='linear', elimination_criteria='aic')
#    Character Variables (Dummies Generated, First Dummies Dropped): []
#    Entered : PT 	AIC : -41.250652068982845
#    Entered : Slog 	AIC : -54.48701722697288
#    Entered : D 	AIC : -63.4615235999384
#    Entered : NE 	AIC : -68.20404386127106
#    Break : Significance Level
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.805
#    Model:                            OLS   Adj. R-squared:                  0.776
#    Method:                 Least Squares   F-statistic:                     27.91
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           3.04e-09
#    Time:                        18:03:06   Log-Likelihood:                 39.102
#    No. Observations:                  32   AIC:                            -68.20
#    Df Residuals:                      27   BIC:                            -60.88
#    Df Model:                           4
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept     -3.4612      1.146     -3.021      0.005      -5.812      -1.110
#    PT            -0.1844      0.042     -4.347      0.000      -0.271      -0.097
#    Slog           0.6629      0.129      5.120      0.000       0.397       0.928
#    D              0.0610      0.016      3.821      0.001       0.028       0.094
#    NE             0.0831      0.033      2.516      0.018       0.015       0.151
#    ==============================================================================
#    Omnibus:                        2.387   Durbin-Watson:                   2.513
#    Prob(Omnibus):                  0.303   Jarque-Bera (JB):                1.222
#    Skew:                          -0.179   Prob(JB):                        0.543
#    Kurtosis:                       3.888   Cond. No.                     5.74e+03
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 5.74e+03. This might indicate that there are
#    strong multicollinearity or other numerical problems.
#    AIC: -68.20404386127106
#    BIC: -60.87536434727242
#    Final Variables: ['intercept', 'PT', 'Slog', 'D', 'NE']

print(iterations_logs)
#    Entered : PT
#
#
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.455
#    Model:                            OLS   Adj. R-squared:                  0.436
#    Method:                 Least Squares   F-statistic:                     25.00
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           2.33e-05
#    Time:                        18:03:06   Log-Likelihood:                 22.625
#    No. Observations:                  32   AIC:                            -41.25
#    Df Residuals:                      30   BIC:                            -38.32
#    Df Model:                           1
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept      2.6873      0.024    111.195      0.000       2.638       2.737
#    PT            -0.2791      0.056     -5.000      0.000      -0.393      -0.165
#    ==============================================================================
#    Omnibus:                        0.158   Durbin-Watson:                   2.053
#    Prob(Omnibus):                  0.924   Jarque-Bera (JB):                0.359
#    Skew:                          -0.096   Prob(JB):                        0.836
#    Kurtosis:                       2.518   Cond. No.                         2.67
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    AIC: -41.250652068982845
#    BIC: -38.319180263383394
#
#
#    Entered : Slog
#
#
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.661
#    Model:                            OLS   Adj. R-squared:                  0.638
#    Method:                 Least Squares   F-statistic:                     28.29
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           1.53e-07
#    Time:                        18:03:06   Log-Likelihood:                 30.244
#    No. Observations:                  32   AIC:                            -54.49
#    Df Residuals:                      29   BIC:                            -50.09
#    Df Model:                           2
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept      0.6795      0.478      1.422      0.166      -0.298       1.657
#    PT            -0.2897      0.045     -6.464      0.000      -0.381      -0.198
#    Slog           0.6920      0.165      4.205      0.000       0.355       1.028
#    ==============================================================================
#    Omnibus:                        1.294   Durbin-Watson:                   2.014
#    Prob(Omnibus):                  0.524   Jarque-Bera (JB):                0.641
#    Skew:                          -0.338   Prob(JB):                        0.726
#    Kurtosis:                       3.152   Cond. No.                         89.1
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    AIC: -54.48701722697288
#    BIC: -50.0898095185737
#
#
#    Entered : D
#
#
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.760
#    Model:                            OLS   Adj. R-squared:                  0.734
#    Method:                 Least Squares   F-statistic:                     29.48
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           8.25e-09
#    Time:                        18:03:06   Log-Likelihood:                 35.731
#    No. Observations:                  32   AIC:                            -63.46
#    Df Residuals:                      28   BIC:                            -57.60
#    Df Model:                           3
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept     -3.3122      1.249     -2.653      0.013      -5.870      -0.755
#    PT            -0.2129      0.045     -4.772      0.000      -0.304      -0.122
#    Slog           0.6710      0.141      4.752      0.000       0.382       0.960
#    D              0.0589      0.017      3.385      0.002       0.023       0.095
#    ==============================================================================
#    Omnibus:                        2.000   Durbin-Watson:                   2.398
#    Prob(Omnibus):                  0.368   Jarque-Bera (JB):                0.892
#    Skew:                          -0.146   Prob(JB):                        0.640
#    Kurtosis:                       3.764   Cond. No.                     5.73e+03
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 5.73e+03. This might indicate that there are
#    strong multicollinearity or other numerical problems.
#    AIC: -63.4615235999384
#    BIC: -57.598579988739495
#
#
#    Entered : NE
#
#
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.805
#    Model:                            OLS   Adj. R-squared:                  0.776
#    Method:                 Least Squares   F-statistic:                     27.91
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           3.04e-09
#    Time:                        18:03:06   Log-Likelihood:                 39.102
#    No. Observations:                  32   AIC:                            -68.20
#    Df Residuals:                      27   BIC:                            -60.88
#    Df Model:                           4
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    intercept     -3.4612      1.146     -3.021      0.005      -5.812      -1.110
#    PT            -0.1844      0.042     -4.347      0.000      -0.271      -0.097
#    Slog           0.6629      0.129      5.120      0.000       0.397       0.928
#    D              0.0610      0.016      3.821      0.001       0.028       0.094
#    NE             0.0831      0.033      2.516      0.018       0.015       0.151
#    ==============================================================================
#    Omnibus:                        2.387   Durbin-Watson:                   2.513
#    Prob(Omnibus):                  0.303   Jarque-Bera (JB):                1.222
#    Skew:                          -0.179   Prob(JB):                        0.543
#    Kurtosis:                       3.888   Cond. No.                     5.74e+03
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 5.74e+03. This might indicate that there are
#    strong multicollinearity or other numerical problems.
#    AIC: -68.20404386127106
#    BIC: -60.87536434727242

final_vars
#   ['intercept', 'PT', 'Slog', 'D', 'NE']

#   fit reduced model again
modforward = smf.ols(formula='Clog ~ PT + Slog + D + NE', data=data).fit()
print(modforward.summary())
#                                OLS Regression Results
#    ==============================================================================
#    Dep. Variable:                   Clog   R-squared:                       0.805
#    Model:                            OLS   Adj. R-squared:                  0.776
#    Method:                 Least Squares   F-statistic:                     27.91
#    Date:                Sun, 07 Mar 2021   Prob (F-statistic):           3.04e-09
#    Time:                        18:05:17   Log-Likelihood:                 39.102
#    No. Observations:                  32   AIC:                            -68.20
#    Df Residuals:                      27   BIC:                            -60.88
#    Df Model:                           4
#    Covariance Type:            nonrobust
#    ==============================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
#    ------------------------------------------------------------------------------
#    Intercept     -3.4612      1.146     -3.021      0.005      -5.812      -1.110
#    PT            -0.1844      0.042     -4.347      0.000      -0.271      -0.097
#    Slog           0.6629      0.129      5.120      0.000       0.397       0.928
#    D              0.0610      0.016      3.821      0.001       0.028       0.094
#    NE             0.0831      0.033      2.516      0.018       0.015       0.151
#    ==============================================================================
#    Omnibus:                        2.387   Durbin-Watson:                   2.513
#    Prob(Omnibus):                  0.303   Jarque-Bera (JB):                1.222
#    Skew:                          -0.179   Prob(JB):                        0.543
#    Kurtosis:                       3.888   Cond. No.                     5.74e+03
#    ==============================================================================
#
#    Notes:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#    [2] The condition number is large, 5.74e+03. This might indicate that there are
#    strong multicollinearity or other numerical problems.


#   diagnostic tools
diagnostic_plots(X, y, model_fit=modforward)

#   REMARK: Good model.



#--------------------------------------------------------------------------------------------------------------------------------------------------
