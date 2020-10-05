# Cryptocurrencies
######  
Unsupervised machine learning, PCA, KMeans, etc.

---

###
Resources
---  
######   
Python, K-Means, scikit-Learn, Pandas, Plotly Express, hvPlot 
######  

---
### Summary
---
######

Accountability Accounting is interested in offering a new cryptocurrencies investment portfolio for its customers. The company, is seeking a report of what cryptocurrencies are on the trading market and how cryptocurrencies could be grouped toward creating a classification for developing a new investment product.

Created an unsupervised machine learning model to analyze data on the cryptocurrencies traded on the market.  The provided [data set](https://min-api.cryptocompare.com/data/all/coinlist) was processed with this Python code and prepared for Principle Component Analysis to reduce data set dimensions to three.  Cryptocurrency clusters were predicted using the K-means algorithm with an elbow-curve technique to select the best K-value.

Finally, the cryptcurrency data was visualized in a number of ways.  The code plotted the data with each principle component on an axis in a 3D hvPlot.  Then it created a filterable table of cryptocurrencies.  Lastly, the data was visualized with a logarithmic 2D chart comparing coin supply to coins mined.
