#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Part II: Multiple Regression
#   'Example 9.0.1-9.1.6 Blasting.py'
#   Author: Marcel Steiner-Curtis
#   Date: 26.08.2018    sml
#         09.03.2019    version  for Python
#         13.02.2023    sml: minor changes
#--------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import scipy as scp


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Example 9.0.1 Blasting
#--------------------------------------------------------------------------------------------------------------------------------------------------
#   read data
data = pd.read_csv('../04 Datasets/blasting.dat', delimiter='\t')
data.head()

#   define new log10-transformed variables
data['Chargelog']    = np.log10(data.Charge)
data['Distancelog']  = np.log10(data.Distance)
data['Vibrationlog'] = np.log10(data.Vibration)


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Scatter diagram: Vibration versus Distance, thermometers=Charge
plt.close()
Chargeprop = (data.Charge-min(data.Charge))/max(data.Charge-min(data.Charge))
plt.xlim([0,200])
plt.ylim([0,12])
plt.scatter(data.Distance, data.Vibration, marker='o', color='blue', edgecolor='black', s=200*(Chargeprop+0.1))
plt.grid()            
plt.axhline(y=0, color='black', linewidth=0.5)
plt.axvline(x=0, color='black', linewidth=0.5)
plt.title('Blasting: Vibration versus Distance')
plt.xlabel('Distance')
plt.ylabel('Vibration')
plt.show()


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Example 9.1.1 Estimation
#--------------------------------------------------------------------------------------------------------------------------------------------------
#   estimation of the parameters (explicit formulae)
n = data.shape[0];   n
##  48
#   vector of the observations
y = data.Vibrationlog
len(y)
##  48
#   matrix of the levels of the regressor variables
X = np.vstack(([1]*n, data.Distancelog, data.Chargelog)).T
X.shape
##  (48, 3)
X
#    array([[1.        , 2.27415785, 0.33845649],
#           [1.        , 2.26245109, 0.52244423],
#           [1.        , 2.24797327, 0.52244423],
#           [1.        , 1.72427587, 0.52244423],
#          
#           [1.        , 1.56820172, 0.33845649],
#           [1.        , 1.5797836 , 0.41497335]])

#   least-squares estimator
betahat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),y);   betahat
#   array([ 2.83225525, -1.51071284,  0.80834385])


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   estimation of the parameters (smf.ols)
mod = smf.ols(formula='Vibrationlog ~ Distancelog + Chargelog', data=data).fit()
print(mod.summary())
#                                OLS Regression Results                            
#    ==============================================================================
#    Dep. Variable:           Vibrationlog   R-squared:                       0.805
#    Model:                            OLS   Adj. R-squared:                  0.796
#    Method:                 Least Squares   F-statistic:                     92.79
#    Date:                Sat, 09 Mar 2019   Prob (F-statistic):           1.08e-16
#    Time:                        07:58:18   Log-Likelihood:                 23.579
#    No. Observations:                  48   AIC:                            -41.16
#    Df Residuals:                      45   BIC:                            -35.54
#    Df Model:                           2                                         
#    Covariance Type:            nonrobust                                         
#    ===============================================================================
#                      coef    std err          t      P>|t|      [0.025      0.975]
#    -------------------------------------------------------------------------------
#    Intercept       2.8323      0.223     12.707      0.000       2.383       3.281
#    Distancelog    -1.5107      0.111    -13.592      0.000      -1.735      -1.287
#    Chargelog       0.8083      0.304      2.658      0.011       0.196       1.421
#    ==============================================================================
#    Omnibus:                        1.002   Durbin-Watson:                   1.143
#    Prob(Omnibus):                  0.606   Jarque-Bera (JB):                0.342
#    Skew:                          -0.098   Prob(JB):                        0.843
#    Kurtosis:                       3.364   Cond. No.                         31.4
#    ==============================================================================
#    
#    Warnings:
#    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Example 9.1.2 Tests on the Significance of any Individual Regression Coefficient
#--------------------------------------------------------------------------------------------------------------------------------------------------
#   hat matrix
H = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(X.T,X))), X.T)
H.shape
#   (48, 48)
#   mean of all leverages is p/n (cf. Chap. 10.2)
np.diag(H).mean()-(1+2)/data.shape[0]
#   fitted values
Vibrationloghat = np.matmul(H, y)
#   residuals
res = data.Vibrationlog - Vibrationloghat
#   error sum of squares
sigma2hat = sum(res**2)/(data.shape[0]-3)
#   residual standard error
np.sqrt(sigma2hat)
#   0.15291019632128042
#   three standard errors
sebeta = np.sqrt(sigma2hat) * np.sqrt(np.diag(np.linalg.inv(np.matmul(X.T,X))));   sebeta
#    array([0.2228918 , 0.11114721, 0.30417236])

#   null hypothesis
beta0 = [0]*3
#   three test statistics
Testbeta = (betahat-beta0)/sebeta;   Testbeta
#   array([ 12.70686176, -13.59200002,   2.65751909])

#   three P-values
2*(1-scp.stats.t.cdf(np.abs(Testbeta), df=n-3))
#   array([2.22044605e-16, 0.00000000e+00, 1.08572500e-02])


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Example 9.1.3 Confidence Intervals on the Regression Coefficient
#--------------------------------------------------------------------------------------------------------------------------------------------------
#   significance level
alpha = 0.05
#   three 95% confidence intervals (explicit formulae)
tcrit = scp.stats.t.ppf(1-alpha/2, df=data.shape[0]-3);   tcrit
##  2.0141033848332923
CI = np.vstack((betahat - tcrit*sebeta, betahat + tcrit*sebeta))
CI
#    {'lower CI': array([ 2.38332812, -1.73457481,  0.19570927]),
#     'upper CI': array([ 3.28118237, -1.28685088,  1.42097843])}

#   three 95% confidence intervals (smf.ols)
mod.conf_int(alpha=alpha)
#                        0         1
#    Intercept    2.383328  3.281182
#    Distancelog -1.734575 -1.286851
#    Chargelog    0.195709  1.420978


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Example 9.1.4 Confidence Interval of the Response
#--------------------------------------------------------------------------------------------------------------------------------------------------
#   confidence interval of the response (explicit formulae)
x0 = np.array([1, np.log10(50), np.log10(3.0)])
y0hat  = np.matmul(x0, betahat)
seconf = np.sqrt(sigma2hat * np.matmul(np.matmul(x0, np.linalg.inv(np.matmul(X.T,X))), x0))
10**(y0hat + np.array([0,-1,1])*scp.stats.t.ppf(1-alpha/2, df=data.shape[0]-3)*seconf)
#    array([4.47999438, 3.91036534, 5.13260218])

#   define new point
datanew = pd.DataFrame(data={'Distancelog': [np.log10(50)], 'Chargelog': [np.log10(3.0)]})
#   predict (smf.ols)
VibrationlogConf = mod.get_prediction(datanew).summary_frame(alpha=0.05)
10**VibrationlogConf
#           mean   mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
#    0  4.479994  1.069851       3.910365       5.132602       2.17615      9.222873


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Example 9.1.5 Prediction Interval
#--------------------------------------------------------------------------------------------------------------------------------------------------
#   prediction interval of the response (explicit formulae)
sepred = np.sqrt(sigma2hat + seconf**2)
10**(y0hat + np.array([0,-1,1])*scp.stats.t.ppf(1-alpha, df=data.shape[0]-3)*sepred)
#   array([4.47999438, 2.45355949, 8.18009497])

#   predict (smf.ols) (one-sided, thereforealpha=0.1)
VibrationlogConf = mod.get_prediction(datanew).summary_frame(alpha=0.1)
10**VibrationlogConf
#           mean   mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
#    0  4.479994  1.069851       3.999734       5.017921      2.453559      8.180095


#--------------------------------------------------------------------------------------------------------------------------------------------------
#   Example 9.1.6 Coefficient of Determination, Multiple R-squared
#--------------------------------------------------------------------------------------------------------------------------------------------------
#   multiple R-squared (explicit formulae)
np.corrcoef(data.Vibrationlog, mod.fittedvalues.values)**2
#    array([[1.       , 0.8048385],
#           [0.8048385, 1.       ]])
#   multiple R-squared (smf.ols)
mod.rsquared
#    0.8048384966875008


#--------------------------------------------------------------------------------------------------------------------------------------------------
