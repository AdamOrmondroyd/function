# ![`function`](function.png "function")

## Notes
- `jax` CMB likelihoods: [clipy](https://github.com/benabed/clipy)
- How to denote derivatives? Trailing underscore for tau derivative?

```math
\frac{\mathrm d\log T}{\mathrm d\log a} = \frac{a}{T} \frac{\mathrm d T}{\mathrm d a}
= \frac{a}{T} \frac{\mathrm d T}{\mathrm d a}\frac{\mathrm d T}{\mathrm d \tau} \frac{\mathrm d \tau}{\mathrm d a} = \frac{a}{T} \frac{T'}{a'}
```

## Symbols and meanings
| maths | code | meaning |
|-------|------|---------|
| $\tau$ | `tau` | conformal time |
| $\mathcal H$ | `hubble` | Hubble parameter in conformal time $\frac 1 a \frac{\mathrm d a}{\mathrm d \tau}$ |
