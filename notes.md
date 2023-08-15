Taweret takes an MCMC context (these are fairly easy to setup, and I can give examples)
The context has to have a variable in called `mixed_likelihood` which the sample method then calls to do the sampling.

Questions: 
- How to specify priors, for example, for plotting predictive prior distributions
- How to extract predictive prior and posterior distributions (are the chains post sampling readily available?) for weights? For models?

Functions for Linear Model Mixing
- set_prior
- predict
- prior_predict
- predict_weights
- evaluate
- mix_likelihood
- train

Functions to add:
- corner_plot
