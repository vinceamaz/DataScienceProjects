# Anomaly Detection with Bearing Dataset

## 1. Introduction

Working in the manufacturing industry, I've always heard people talking about SPC (statistical process control) and how powerful it is back in the old days. However, it seems strange to me that it is somehow used much less compared with how frequent it is being brought up. What's stopping people from using it nowadays?

SPC is the most widely used traditional method to monitor the stability of a process/ equipment in manufacturing industry. The common practice is to identify an individual parameter of interest and perform some analysis to establish upper and lower control limits when the process/ equipment is initially set up. The upper and lower control limits are usually derived from the 3 standard deviations from the mean value of the parameter from the sampled lot. In the on-going production phase, the common approach is to keep monitoring the parameter and an alarm will be triggered if a measurement is outside of the control limits or some abnormal trend is observed within the limits.

One of the major drawbacks of SPC is that it only monitors an individual parameter at a time, which means:

* It is not able to capture the interactions among multiple parameters. However, in practice It is difficult to identify a parameter that is able to individually indicate the health of a process/ equipment. Our best chance to detect an issue is to have a model that considers all the parameters and their interactions.
* An SPC chart is developed for each parameter, which is cumbersome if we want to monitor a large number of parameters for a single process/ equipment. In real world, it is not uncommon that we have a large number of assets of each equipment. If we have n assets and each asset has m parameters to be monitored, in total n x m charts will be created and monitored.

Fortunately, in recent years many novel methods have been introduced to overcome the disadvantage of SPC, thanks to the rapid development of machine learning and deep learning. I've come across a really great article written by Vegard Flovik on https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7. He introduced two different types of unsupervised methods to solve anomaly detection problem and provided a practical example with code:

* PCA (principal component analysis) (implemented with scikit-learn) with Mahalanobis distance (implemented with numpy)
* Autoencoder (implemented with keras)

For the benefit of understanding the concepts introduced in the great article, I've followed along his code to reproduce the results and provided a different Pytorch version of autoencoder implementation.

## 2. Data

The bearing dataset was downloaded from the NASA Prognostics Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing). There are 3 datasets in total and each dataset consists of individual files that are 1-second vibration signal snapshots recorded at specific intervals. Vegard Flovik selected the second one to perform this study. 

For the second dataset, there are 984 data files with a measurement interval of 10 minutes between each two adjacent data file, and each data file has 20,480 measurements. This means we conduct a measurement exercise of the equipment every 10 minutes and in each exercise we take 20,480 measurements.

For each measurement, there are 4 different parameters recorded, representing 4 different types of bearings of the equipment. This project is essentially studying the interactions between the 4 bearings. For each data file, Vegard Flovik took an average of all 20,480 measurements and the final dataset has 984 rows and 4 columns. I believe the purpose of taking the average is for simplicity and the reduction of noise among individual measurements. However, one can actually train a more sophisticated model by using the large number of individual measurements without taking the average.

The processed data look like below:

|                     | Bearing 1 | Bearing 2 | Bearing 3 | Bearing 4 |
| ------------------- | --------- | --------- | --------- | --------- |
| 2004-02-12 10:32:39 | 0.058333  | 0.071832  | 0.083242  | 0.043067  |
| 2004-02-12 10:42:39 | 0.058995  | 0.074006  | 0.084435  | 0.044541  |
| 2004-02-12 10:52:39 | 0.060236  | 0.074227  | 0.083926  | 0.044443  |
| 2004-02-12 11:02:39 | 0.061455  | 0.073844  | 0.084457  | 0.045081  |
| 2004-02-12 11:12:39 | 0.061361  | 0.075609  | 0.082837  | 0.045118  |

If we plot all 984 data points of the 4 bearings individually, we can see the chart below. It looks like Bearing 1 started to have some abnormal trend after 2004-02-16 while others still behaved quite normal until after 2004-02-18. In traditional SPC, if we are unlucky, we may not have selected Bearing 1 to be monitored and may not have detected any issue before 2004-02-19, when a breakdown event actually occurred.



For the purpose of model training, the data was split into training set (222 observations) and test set (760 observations).



## 3. Multivariate statistical analysis (PCA with Mahalanobis distance)

### 3.1 PCA

PCA performs a linear mapping of the data to a lower-dimensional space in such a way that the variance of the data in the low-dimensional representation is maximised. There are often two use cases:

* Dimensionality reduction of data for visualisation due to human beings are only able to understand visualisations of up to 3 dimensions.
* Dimensionality reduction of data for feature selection. In some research areas it is difficult to obtain a large amount of data and each observation may have a large number of features (dimensions). In some extreme cases, the number of features can be even more than the number of observations. In some scenarios, PCA will be a good method to reduce the dimensionality of data to allow a model to be trained. 

Mathematically, the covariance matrix of the data is constructed and the eigenvectors of this matrix are computed. The eigenvectors that correspond to the largest eigenvalues (the principal components) can now be used to reconstruct a large fraction of the variance of the original data. The original feature space has now been reduced (with some data loss, but hopefully retaining the most important variance) to the space spanned by a few eigenvectors.

### 3.2 Mahalanobis distance

Consider the problem of estimating the probability that a data point belongs to a distribution, as described above. Our first step would be to find the centroid or center of mass of the sample points. Intuitively, the closer the point in question is to this center of mass, the more likely it is to belong to the set. However, we also need to know if the set is spread out over a large range or a small range, so that we can decide whether a given distance from the center is noteworthy or not. The simplistic approach is to estimate the `standard deviation` of the distances of the sample points from the center of mass. By plugging this into the normal distribution we can derive the probability of the data point belonging to the same distribution.

The drawback of the above approach was that we assumed that the sample points are distributed about the center of mass in a spherical manner. Were the distribution to be decidedly non-spherical, for instance `ellipsoidal`, then we would expect the probability of the test point belonging to the set to depend not only on the distance from the center of mass, but also on the direction. In those directions where the ellipsoid has a short axis the test point must be closer, while in those where the axis is long the test point can be further away from the center. Putting this on a mathematical basis, the ellipsoid that best represents the set’s probability distribution can be estimated by calculating the covariance matrix of the samples. The `Mahalanobis distance (MD)` is the distance of the test point from the center of mass divided by the width of the ellipsoid in the direction of the test point.

`Mahalonobis distance` is the distance between a point and a distribution. And not between two distinct points. It is effectively a multivariate equivalent of the Euclidean distance.
* It transforms the columns into uncorrelated variables
* Scale the columns to make their variance equal to 1
* Finally, it calculates the Euclidean distance

The formula to compute `Mahalonobis distance` is:

$$
D^2 = (x - m)^T \cdot C^{-1} \cdot (x - m)
$$

where

* $D^2$ is the square of the Mahalonobis distance
* $x$ is the vector of the observation (row in a dataset)
* $m$ is the vector of mean values of indepedent variables (mean of each column)
* $C^{-1}$ is the inverse covariance matrix of independent variables
* $(x - m)$ is essentially the distance of the vector from the mean
* $(x - m)^T \cdot C^{-1}$ is essentially a multivariate equivalent of the regular standardisation $z = \frac{x - \mu}{\sigma} = \frac{(\text{x vector}) - (\text{mean vector})}{\text{covariance matrix}}$

The effect of dividing by the covariance is:

* If the variables in your dataset are strongly correlated, then, the covariance will be high. Dividing by a large covariance will effectively reduce the distance.
* If the variables are not correlated, then the covariance is not high and the distance is not reduced much.
* Effectively, it addresses both the problems of scale as well as the correlation of the variables.

In order to use the MD to classify a test point as belonging to one of N classes, one first estimates the `covariance matrix` of each class, usually based on samples known to belong to each class. In our case, as we are only interested in classifying “normal” vs “anomaly”, we use training data that only contains normal operating conditions (measured when an equipment is new and stable) to calculate the `covariance matrix`. Then, given a test sample, we compute the MD to the “normal” class, and classify the test point as an “anomaly” if the distance is above a certain threshold.

Note of caution: Use of the MD implies that inference can be done through the mean and covariance matrix — and that is a property of the `normal distribution` alone. This criteria is not necessarily fulfilled in our case, as the input variables might not be normal distributed. However, we try anyway and see how well it works!

### 3.3 Mahalanobis distance distribution of training data

From the distribution below, we can set the anomaly threshold to be 4 standard deviations from mean of training data's Mahalanobis distances. I actually set the limit based on 4 standard deviations instead of 3 based on the observed distribution of training data to avoid false alarms due to overly tight control limit. The concept is similar to SPC, where the threshold of 4 standard deviations from mean is served as an upper control limit. 



### 3.3 Mahalanobis distances of all data

The chart below shows that we can detect anomaly between 2004-02-16 to 2004-02-17 without the risk of missing Bearing 1 as opposed to SPC. That's 2 days before when the breakdown actually occurred, on 2004-02-19.



## 4. Autoencoder model for anomaly detection

### 4.1  Autoencoder concept

An autoencoder is a type of artificial neural network used to learn efficient data encodings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction. Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input.



In our use case, what we would do is to plot the distribution of the reconstruction loss of the autoencoder for training data to identify where normally reconstruction loss lies and come up with a threshold as the upper control limit. One can also compute the 3 standard deviations from the mean of reconstruction loss to determine an appropriate upper control limit, like what we did in PCA with Mahalanobis distance modelling.

## 4.2 Distribution of loss function in the training set

By plotting the distribution of the calculated loss in the training set, one can use this to identify a suitable threshold value for identifying an anomaly. In doing this, one can make sure that this threshold is set above the “noise level”, and that any flagged anomalies should be statistically significant above the noise background.

From our result, we can use a threshold of 0.3 for flagging an anomaly. We can then calculate the loss in the test set, to check when the output crosses the anomaly threshold.

## 4.3 Autoencoder reconstruction loss of all data

Similar to the PCA with Mahalanobis distance method, the chart below shows that we can detect anomaly in 2004-02-16. One can see that it detects anomaly slightly earlier than the PCA with Mahalanobis distance model. In real world issues, Autoencoder may perform better too since it doesn't have the assumption that the input data follows Gaussian distribution, which is a constraint of Mahalanobis distance.





