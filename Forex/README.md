# Australian dollar exchange rate prediction

## Objective

The main objective of this project is to explore the feasibility of using a simple LSTM based Seq2Seq model with attention mechanism to predict Australian dollar exchange rate.

I my recent study, Seq2Seq model with attention mechanism is commonly used in NLP and has gain massive success in both academic world and in the industry thanks to the elimination of memory loss issue in long input sequence. In particular, pretrained models like transformer further pushed NLP models' performances to the next level.

For an NLP problem such as named entity recognition, we commonly use various features to represent the input embedding of each token including word2vec, POS tags, dependency labels, etc. Therefore, I was thinking I could apply the same idea of using some Australian dollar related features as model input to build a Seq2Seq model for Australian dollar exchange rate prediction.

## Input Features

Without having much experience with currency trading, I had to do some research online from scratch to find indices related to Australian dollar currency. The good thing is that there are quite a lot of articles online analysing Australian dollar and its fluctuations. It's also not too difficult to acquire some historical data including the Australian dollar exchange rate itself and some other related indices. The features I finally decided to use are below mainly due to their high correlation to AUD and accessibility online:

* Australian dollar exchange rate
* Brent Oil Futures
* Centre bank interest rates
* Dow Jones Industrial Average
* Gold price
* Iron Ore Fines 62% Fe CFR Futures
* S&P ASX 200

## Exploratory Feature Analysis

The most interesting finding in EDA is that people usually think that a country's currency shall be mostly correlated to its centre bank interest rate. Surprisingly for Australian dollar, the currency is correlated more to US centre bank interest rate and China centre bank interest rate. More specifically, there is little correlation between Australian dollar exchange rate and China centre bank interest rate from 2000 to 2009, but much more from 2010 to 2020. This shows that Australian dollar and possibly Australian economy started to have a much closer relationship with China from 2010.

It is also a bit surprising to see the correlation between AUD and crude oil price outweighs iron price, as we know Australia is world's leading iron ore exporter. One explanation can be that oil price influences iron ore price because the latter usually includes shipping costs, which indirectly makes crude oil price impacts AUD. It also indicates that Australia economy, or at least AUD is very vulnerable to the fluctuations of iron price.

**2000 - 2009:**

![image-20200725211545706](D:\GitHub\DataScienceProjects\Forex\Images\2000-2009 Correlation.png)

**2010-2020:**

![image-20200725211613044](D:\GitHub\DataScienceProjects\Forex\Images\2010-2020 Correlation.png)

**Line plot of highly correlated features:**

![image-20200725211652975](D:\GitHub\DataScienceProjects\Forex\Images\Trend of features.png)

## Model Structures

For comparison purposes, I've trained two models for this project. 

The first model is a simple Seq2Seq model with Bi-LSTM as encoder and LSTM as decoder. The encoder network takes input of concatenation of features and labels, while the decoder network takes input of only the label (in training state) or output from the previous time step (in inference state).

The second model is similar to the first one except for the addition of the attention mechanism. More specifically, the attention mechanism is implemented as below:

1. Compute the projection of encoder hidden states $H$ by $H_{project} = W^H_{project} \cdot H$ 
2. Compute the attention scores between current decoder hidden state $s_t$ and encoder hidden states projection $H_{project}$ by $e_t = H_{project} \cdot s_t$
3. Compute the attention weights between current decoder hidden state and encoder hidden states projection by applying softmax $a_t = \text{softmax}(e_t)$
4. Compute the attention output by $c_t = a_t \cdot H$
5. Concatenate the attention output and current decoder hidden state by $[c_t;s_t]$
6. Pass the concatenated output through a linear layer to be the final output of the attention mechanism by $W^{output}_{project} \cdot [c_t;s_t]$

## Model Performances

Both models perform quite poor in terms of accurately predicting the future AUD exchange rate. The design of the model is to take 30 days features and AUD exchange rate as encoder input, and predict the AUD exchange rate in the next 1 day or next 5 days. The model with attention yields a slightly lower test error compared with that without attention. However, the overall performances between the models are quite similar. I will only show the output of the model with attention as example below.

For 1 day prediction, the model is only able to align with the overall trending of the ground truth, but fail to accurately predict each upward and downward inflection points. Sometimes the output of the model is completely opposite of the ground truth.

**1 day prediction of model with attention:**

![image-20200725214146139](D:\GitHub\DataScienceProjects\Forex\Images\1 day prediction output)

For 5 day prediction, I've sampled a few random dates in history to test the model. It turns out the model's outputs are consistently disappointing compared with the ground truth. One of the samples is shown below:

**5 day prediction of model with attention:**

![image-20200725214500577](D:\GitHub\DataScienceProjects\Forex\Images\5 day prediction output)

## Conclusion

In this project, I have implemented Seq2Seq model with BiLSTM and attention mechanism using Pytorch. The objective of the model is to use features correlated to AUD exchange rate and AUD exchange rate in the past to predict AUD exchange rate in the future.

LSTM was introduced in the 90s and currency prediction with LSTM is nothing noval. Researchers found that a simple LSTM network generally fails to effectively predict currency. 

The main purpose of this project is to explore how much performance boost we can get by incorporating the attention mechanism. It turns out the performance boost from the attention mechanism is quite limited. The main advantage of the attention machanism for a LSTM model is to get rid of the vanishing gradient issue. If the reason of the failure of using a simple LSTM model to predict currency is not due to vanishing gradient, then it is not surprising that the attention mechanism does not add significant value to the model performance.