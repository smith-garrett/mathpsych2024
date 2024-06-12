# Investigating parameter identifiability and "sloppiness" in a dynamical model of reading

- Stage setting: A common task is to design a model and fit it to data. The model embodies a set of hypotheses about some cognitive process that we want to test by evaluating the fit to the data
- Often though, we run into problems when fitting the model, e.g., using max. lik. or Bayesian methods
  - Numerical optimization of the likelihood function is inefficient, inconsistent, or slow
  - Or MCMC chains don't converge or have lots of divergent transitions
- State folk theorem
- I will present a set of methods that stumbled across when trying to fix computational problems with a model of eye-movement control during reading
  - Borrowed from physics and econometrics, they have not seen much application in cognitive science to my knowledge
- I will first introduce the model, and then I will introduce and demonstrate ideas related to parameter identifiability, redundancy, and sloppiness, the problems they can cause, and what to do when you encounter them

## SWIFT

### (SWIFT's) Likelihood function(s)

- Intro likelihood function
- Discuss SWIFT's & parameters

## (Practical) Identifiability

- Identifiability (Cole 2020): identifiable if different parameter settings lead to different model outputs
- Practical identifiability (Raue et al., 2009): Finite confidence/credible interval
  - Can be identified using profile likelihoods [DEFINE]

## Sloppiness

- There are parameters that can be varied over orders of magnitude without substantially changing model output
- Approximately evenly spaced eigenvalues of observed information matrix that span many orders of magnitude

## Methods

### Profile likelihoods

- A method of visually inspecting cross-sections of the whole likelihood function for individual parameters
- Calculated by iteratively fixing values of the parameter of interest, and then maximizing the likelihood wrt. the rest of the parameters
- Result is a set of plots showing the approximate shape of the likelihood function

- Interpreting:
  - Flatness
  - Practical identifiability: bounded CI, calculated by inverting the likelihood ratio test
  - [EXAMPLES]

### Bias and coefficient of variation

### Observed information matrix/Hessian method

- Another useful tool is the observed information matrix
  - Matrix of partial second derivatives of the likelihood wrt. each of the parameters
  - Hessian gives curvature, negative of that is OIM, which quantifies how much information the data carry about each of the parameters and their interactions
  - High values on the diagonal suggest ease of identifiability
- Eigenvalue analysis
  - Large range of eigenvalues coupled with small ones = sloppy
  - Small (log) eigenvalues correspond eigenparameters that can be changed a lot without greatly changing the likelihood, i.e., the model's predictions

#### Near-redundancy

## Solutions

### More data

### Reparameterize model

- Sloppiness is responsive to scale of parameters, so if possible, have them all on approx. same order of magnitude, or log-transform

#### Fix parameter to reasonable values

#### Minimally disruptive curves to eliminate parameter (combinations)

### Slap on some priors

## Results with SWIFT

## Conclusion

