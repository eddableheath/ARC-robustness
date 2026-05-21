# Visualising Formalisms for Investigating Robustness to Evasion Attacks

The goal here is to try and get some understanding of which formalism would be a good setting to prove some theorems with.

## Initial plan

The initial plan is as follows:

1. Investigate visulations of the `MNIST` dataset using various kinds of manifold visualisation techniques.
2. Train a simple classifier for the `MNIST` dataset - just use a dense model in torch.
3. Extract the features at each layer to visualise the homomorphic transformations to the space.
4. Apply an evasion attack, and try to see if a particular formalism characterising the relationship to the adversarial images and the original images predicts a lack of stochastic separability (i.e. only discrete Ricci-Hamilton flow _with surgery_ will provide robustness).

This will start as a series of notebooks, and become a series of scripts based on an underlying package for visualisation.
