# Likelihood profiles with Toy SWIFT

## 29.01.2024

Progress to remember:
- Model implemented in Turing
- Automatic differentiation works! Opens so many doors...

It really seems like there are serious identifiability problems with Toy SWIFT. First, the
Hessian matrix of the likelihood function evaluated at the MLE is nearly singular,
indicating flat ridge(s) in the likelihood. This, combined with the actual likelihood
profiles, suggests that there are both structurally and practically non-identifiable
parameters. Basically, only the mean fixation duration is identifiable; all other parameters
either have flat profiles (= structural non-identifiability) or likelihood profiles that are
only bounded on one side (= practical non-identifiability).

Some of this should have been obvious: $r$, $ν$, and $μ$ all get multiplied directly
together, $ι$ and $γ$ are basically a single quotient, $r$ and $β$ interact pretty directly,
and $η$ and $γ$ are related exponentially.

Responses to possible reactions:
- "But the likelihood profiles in paper XXX looked good": Sure, but that was with data
  generated *by the model* and with the non-profiled parameters held at *the known true
  values*. It's a more than ideal situation. When we evaluate on real data, we can't expect
  nearly as good a fit compared to simulated data. Also, [@cole2020parameter, 4.1.2] states
  that doing identifiability analysis with simulated data can only detect
  intrinsic/model-internal identifiability. The results of (profile) likelihood-based
  parameter recovery can also depend on the particular parameter values used to simulate
  data, as well as the initial values used in the MLE recovery algorithm *and* the MLE
  recovery algorithm itself. In particular, starting the MLE recovery search *at* the known
  MLEs can actually hide non-identifiability present even with simulated data.


## 21.02. Trying to reconcile R & Julia versions

Errors in Ralf's code
- Product of saliencies used for `leftsal` in generating data, sum used in likelihood
- In the likelihood fn., the spatial likelihood is calculated with the *next* word, while
  the part of the fixation duration with ι is calculated with the *current* word's
  activation. This isn't necessarily an error on its own, but in the same line of code, the
  *next* word's fixation duration is used in `dgamma`
- Also, κ gets multiplied in twice, once in the calculation of `leftact` and once again in
  the denominator of `rate2` in toyLL.R.

Even with these errors, the model is still sloppy, although the eigenvalues are much larger
than I've typically seen.

When the likelihood and data generating code match, the likelihood profiles (calculated
incorrectly) look worse. It's only with the mismatch that they look good.

