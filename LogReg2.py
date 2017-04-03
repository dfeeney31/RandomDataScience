
# coding: utf-8
#This is from the excellent y hat tutorial on Logistic Regression at http://blog.yhat.com/posts/logistic-regression-python-rodeo.html
#First time trying python for logitreg

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# read the data in and look at first five rows
df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
df.head()

np_df = np.array(df)

# rename the 'rank' column because there is also a DataFrame method called 'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]
print df.columns
print df.describe()

# frequency table cutting presitge and whether or not someone was admitted
print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])


df.hist()
pl.show()


# dummify rank. We want a dummy variable for Prestige
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
print dummy_ranks.head()


#Make a clean dataframe with admit, gre, and gpa plus the dummy variables
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
print data.head()


#Adding the intercept for the Logit Reg
data['intercept'] = 1.0


#Training dataset for GRE, GPA, and prestige dummy variables
train_cols = data.columns[1:]
#logit from the stats model package does logistic regression with sm.Logit(dependentVar, predictors)
logit = sm.Logit(data['admit'], data[train_cols])

# fit the model with logit.fit
result = logit.fit()



#Summary output; similar to R, coefficients, z scores, p values and confint's
print result.summary()


# look at the confidence interval of each coeffecient- more sig figs then above
print result.conf_int()


# Calculate the odds ratios by exponentiating the coefficients of the predictors. 
# Tells you how much a one unit increase/decrease affects the odds of being admitted
print np.exp(result.params)
# From this, you can see being at a school with prestige of 2 decreases odds of admission by 50%


# odds ratios and 95% CI with more manipulation
params = result.params
#confidence intervals
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf)

# Generate many possible values of GRE and GPA, so we can evaluate the predictability
# Use linspace to make an evenly spaced range of 10 values from the min to the max
gres = np.linspace(data['gre'].min(), data['gre'].max(), 10)
print gres
gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 10)
print gpas


#define the cartesian function, which will make linear combinations 
def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


#Enumerate possible combos
combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3, 4], [1.]]))

# recreate the dummy variables
combos.columns = ['gre', 'gpa', 'prestige', 'intercept']
dummy_ranks = pd.get_dummies(combos['prestige'], prefix='prestige')
dummy_ranks.columns = ['prestige_1', 'prestige_2', 'prestige_3', 'prestige_4']


# keep only what we need for making predictions
cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
combos = combos[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])

# make predictions on the enumerated dataset
combos['admit_pred'] = result.predict(combos[train_cols])

print combos.head()


def isolate_and_plot(variable):
    # isolate gre and class rank
    grouped = pd.pivot_table(combos, values=['admit_pred'], index=[variable, 'prestige'],
                  aggfunc=np.mean)
    # make a plot
    colors = 'rbgyrbgy'
    for col in combos.prestige.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0), plt_data['admit_pred'], color=colors[int(col)])

    pl.xlabel(variable)
    pl.ylabel("P(admit=1)")
    pl.legend(['1', '2', '3', '4'], loc='upper left', title='Prestige')
    pl.title("Prob(admit=1) isolating " + variable + " and presitge")
    pl.show()

isolate_and_plot('gre')
isolate_and_plot('gpa')




