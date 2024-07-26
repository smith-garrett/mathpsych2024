# Combining likelihood-based and Bayesian methods for model identifiability

## Importance of linking models to data

- Mechanistic models allow us to more directly test hypotheses about cognition and behavior
- Can often go beyond purely statistical models by illustrating possible causal mechanisms
  behind behavior [@mcelreath2020statistical] (?)
- [@roberts2000how]: ?

## Importance of model identifiability

- As important as mechanistic models are, they are only truly useful when we can actually
  fit their free parameters with the data available
- Models can be underspecified by the data
- Models can also have dependencies between parameters that are not apparent from the model
  equations alone
- Finding unidentifiable parameters can help with model simplification
- Fundamentally, it's about falsifiability: Is it possible to rule out different variants (=
  parameter settings) of our model given the experimental data we have? If the model is
  (practically) unidentifiable, we cannot really falsify it. So this is really fundamental.

## ToySWIFT

## Profile likelihoods for model identifiability

- Basically, find the MLE while holding one of the parameters constant ⇝ optimization
  problem
- Can derive approximate confidence intervals for each parameter individually
- Identifiability: there should be a single, clear peak in the profile likelihood for each
  parameter
    - Keine Garantie, but it suggests at least that there might be one set of truly optimal
      paramters for the model given the data

## Using priors to make unidentifiable parameters identifiable

- ToySWIFT profile likelihoods
    - Both with and without parsing
    - ToySWIFT will probably be mostly identifiable
    - Simplified SEAM is definitely not, at least in ζ
- What do? Add priors!
- Bayes' rule shows us that the posterior/joint is just the likelihood weighted by the
  prior, so we can use the prior to downweight unrealistic parameter values
    - Muss man mit Bedacht machen! But domain knowledge is *always* relevant in modeling!

## Results

## Conclusions

