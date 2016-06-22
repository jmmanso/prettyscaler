# PrettyScaler

### This python module uses input data to create a condensed isomorphic transformation ![alt text](http://mathurl.com/hdprl5n.png)

Do you usually use sklearn.preprocessing.StandardScaler to normalize your data? Sometimes, the distribution of the data can be so assymetric and heterogeneous that a better transformation is necessary. PrettyScaler acts in the same way as StandardScaler; fits a mapping from raw to normalized distributions. However, PrettyScaler will neatly condense the metric, squeezing all reals into the bound domain (-1,1) and producing a uniform density distribution.

Basically, it will map each ![alt text](http://mathurl.com/go6xs4l.png) to the cumulative function of the data. Since the training data is finite, PrettyScaler will add some wings to that cumulative function so that it is defined in the entire domain. 
The wing boundaries are defined by the wing_percentile argument, which represents the p-value enclosed by each of the 2 wings. These wings are exponential functions, and are glued to the main body by enforcing that the function value and gradient match and the boundary. Thus, this **makes the extrema of the data to be tucked into (-1,1), in an asymptotic fashion**.

The result of this transformation is a uniform distribution, but PrettyScaler can also output one that generates a bulge-like distribution. Depending on the "kind" argument in the fit() method, the probability density distributions will be:
* ![alt text](http://mathurl.com/zv7az78.png)
* ![alt text](http://mathurl.com/jhu6gqh.png)

The mapping constructed by PrettyScaler meets the following requirements:
* ![alt text](http://mathurl.com/zk3ck4m.png)
* ![alt text](http://mathurl.com/zlurmyc.png), monotonically increasing
* ![alt text](http://mathurl.com/hvdh7lp.png) is non-zero, finite and positive everywhere

When to use PrettyScaler:
* The main use case is to preprocess data for neural nets and SVMs
* It is not helpful for CART-based methods (decision trees, random forest, GBM), although it will not hurt either
* It will completely wash out 1-D clustering signal, so don't use it for this purpose


## Usage example


```python
# Make some basic imports
import matplotlib.pyplot as plt;
%matplotlib inline
import numpy as np;
from prettyscaler import prettyscaler;
```


```python
# let's create some ugly mock data. This will be a mixture
# of a few truncated log-normal distributions:
def create_lognormal_truncated_sample(logmu,logsig,N,sign):
    random_sample = np.random.normal(logmu,logsig,size=N)
    random_sample = random_sample[sign*random_sample>sign*logmu]
    return 10**random_sample

random_sample1 = create_lognormal_truncated_sample(1,1,5000,1)
random_sample2 = create_lognormal_truncated_sample(5,.3,5000,1)
random_sample3 = create_lognormal_truncated_sample(8,.25,5000,-1)
random_sample = np.r_[random_sample1,random_sample2, random_sample3]
```


```python
# Take a look at the monster we've created in a log plot
plt.hist(np.log10(random_sample[random_sample>0]),bins=50);
```


![png](images/output_4_0.png)



```python
# Break the random sample into train and test sets:
np.random.shuffle(random_sample)
test_data, train_data = random_sample[random_sample>np.median(random_sample)],\
    random_sample[random_sample<np.median(random_sample)]
```


```python
# Initialize
pts = prettyscaler.PrettyScaler()
# Fit the data. The most important argument you need
# to care about is the wing_percentile. Roughly,
# set it to the fraction of your data that you would 
# consider quasi-outliers. For example, wing_percentile=0.01 
# will result in fitting the data directly with 98% of the points,
# and a 1% on each side is discarded (the reason for a non-zero 
# wing_percentile is that we need a robust computation of the gradient
# at the wing boundary, and there needs to be enough data density). 
# The argument sample_size determines the size of the grid
# that will be used to map the distribution densities. As a rule
# of thumb, try to have wing_percentile*sample_size > 10. If sample_size
# happens to be larger that the size of the training data, it will revert
# to the latter.
# Finally, ftol is the kind of the minimum difference enforced between
# consecutive values in the cumulative density array. It is there to make sure
# that the gradient of that array not close to being (numerically) flat anywhere.
#
# This might be a lot to take, but for most cases you can just
# use the default values and not worry about it.
#
# So, let's fit some data:
pts.fit(test_data, wing_percentile=0.01,ftol=1e-5,sample_size=1e3)
# If you want to load these params later, you should save
# them now to file:
pts.save_pickle('fitted_params.pkl')
# To load this state from scratch, you would just have to do:
# pts = prettydata.PrettyData()
# pts.load_pickle('fitted_params.pkl')
```


```python
# Transform the test data into a flat and bulge spaces:
test_data_flat = pts.transform(test_data, kind='flat')
test_data_bulge = pts.transform(test_data, kind='bulge')
```

    Mapping to bulge distribution



```python
# Plot the histogram of the flat representation
plt.hist(test_data_flat, bins=20);
```


![png](images/output_8_0.png)



```python
# And a histogram of the bulge representation
plt.hist(test_data_bulge, bins=20);
```


![png](images/output_9_0.png)


### NOTE
In the bulge representation plot, gaps might appear toward the edges. This is because the test data used in that plot is finite, but the regions close to -1 and 1 are "reserved" for mappings from the $(-\infty,+\infty)$ original domain.

## Issues

If you have strong outliers in the data, those might be mapped so close to the boundary limits (like 0.9999...) that python will numerically round them up. In these regions, the non-flat gradient requirement of the transformation is broken. You might end up with a transformed data set with more than one value being equal to 1. This is probably not an issue, since such values are outliers anyway, but it is important to keep in mind.
