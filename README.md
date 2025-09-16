# Compare different quadrature rules for integration

There are two examples provided for calculating the weights and abscissas for gaussian quadrature rules, try:

```
make
./gqconstants
```

or

```
python gqconstants.py
```

You can also use the C++ example as a guide to build your own executable

There is no need to look at rules >~25 for Gaussian quadrature.  And you can also stop at ~ 1000 divisions for the trapezoidal and Simpson's rules.  If you run much longer you'll see the numerical errors bevome visible for the trapezoidal, but hyou'll need to think about how to code efficiently or the running time may be very long.


4) Based on my plots, we see a slope of -2 for the trapezoidal rule and a slope of around -4 for the simpson's rule.

- Hard to integrate functions were functions that were either extremely oscilatory (created more room for error when approximating segments) or functions that had holes (like sqrt(x^2) from -1 to 1), which I believe was because the numerical methods require you to call the functions themselves and they can't be called around the hole. 

- An approach that may work to fix the calculation is a solution that follows a dynamic programming rule. Essentially, we save common solutions in a table of values and call that first before attempting numerical integration. We could also add conditional functionality that breaks integrals up around a hole and then solves both sides and adds them together. 
