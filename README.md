# `function`

## Notes
- `jax` CMB likelihoods: [clipy](https://github.com/benabed/clipy)
- How to denote derivatives? Trailing underscore for tau derivative?
$$
\frac{\mathrm d\log T}{\mathrm d\log a} = \frac{a}{T} \frac{\mathrm d T}{\mathrm d a}
= \frac{a}{T} \frac{\mathrm d T}{\mathrm d a}\frac{\mathrm d T}{\mathrm d \tau} \frac{\mathrm d \tau}{\mathrm d a} = \frac{a}{T} \frac{T'}{a'}
$$
