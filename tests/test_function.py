import pytest
import jax
import jax.numpy as jnp
from jax import jit

# === import the routines under test ===
from function.boltzmann import (
    metric,
    Metric,
    radiation,
    Radiation,
    Baryon,
    _massless_nu_moment_rhs,
    massless_nu_rhs,
    MasslessNeutrino,
    _massive_nu_moment_rhs,
    massive_nu_rhs,
    MassiveNeutrino,
    Background,
)


# -- 1) Test metric helper in the zero‐source limit --
def test_metric_zero_sources():
    # G is hard‐coded to 1 in your metric(), so we just set deltaT00=deltaTii=0.
    a = 2.0
    hubble = 1.0
    k = 4.0
    mi = Metric(h=1.3, hdot=2.0, eta=3.0)
    deltaT00 = 0.0
    deltaTii = 0.0

    out = metric(a, hubble, mi, deltaT00, deltaTii, k)

    # etadot = (hubble*hdot/2)/k^2 = (1*2/2)/(16) = 1/16
    assert float(out.eta) == pytest.approx(1.0 / 16.0, rel=1e-12)

    # hdd = -2*hubble*hdot + 2*k^2*eta = -4 + 2*16*3 = 92
    assert float(out.hdot) == pytest.approx(92.0, rel=1e-12)

    # we expect out.h = old hdot
    assert float(out.h) == pytest.approx(2.0, rel=1e-12)


# -- 2) Test trivial photon hierarchy: no scattering, no metric driving --
@pytest.mark.parametrize("lmax", [2, 3, 5])
def test_radiation_free_streaming_zero(lmax):
    # build a zero‐perturbation radiation state
    f = jnp.zeros(lmax + 1)
    g = jnp.zeros(lmax + 1)
    r = Radiation(f=f, g=g, theta=0.0, delta=0.0)

    # baryon is only used in Thomson drag term, set theta=0 → no effect if ne*sigmaT=0
    b = Baryon(delta=0.0, theta=0.0, t=0.0)

    # zero metric driving
    mi = Metric(h=0.0, hdot=0.0, eta=0.0)

    out = radiation(
        r,
        b,
        mi,
        ne=0.0,  # no scattering
        sigmaT=0.0,
        a=1.0,
        k=1.23,
        t=1.0,
    )

    # all derivatives should remain zero
    assert isinstance(out, Radiation)
    assert out.f.shape == (lmax + 1,)
    assert out.g.shape == (lmax + 1,)
    assert jnp.allclose(out.f, 0.0)
    assert jnp.allclose(out.g, 0.0)
    assert float(out.theta) == pytest.approx(0.0)
    assert float(out.delta) == pytest.approx(0.0)


# 2b) A simple non‐zero check: if f[1]=1, k=2, we get f0' = -k * f1 = -2
def test_radiation_monopole_simple():
    lmax = 3
    f = jnp.zeros(lmax + 1).at[1].set(1.0)
    g = jnp.zeros(lmax + 1)
    r = Radiation(f=f, g=g, theta=0.0, delta=0.0)
    b = Baryon(delta=0.0, theta=0.0, t=0.0)
    mi = Metric(h=0.0, hdot=0.0, eta=0.0)

    out = radiation(r, b, mi, ne=0.0, sigmaT=0.0, a=1.0, k=2.0, t=1.0)
    assert float(out.f[0]) == pytest.approx(-2.0)
    assert float(out.g[0]) == pytest.approx(0.0)


# -- 3) Test massless neutrino moment‐RHS directly for ell=0,1,2 --
def test_massless_moment_rhs_direct():
    # build a toy f_all array length L+1=4
    f_all = jnp.array([0.0, 1.0, 0.5, 0.0])
    k = 2.0
    hdot = 0.1
    etadot = 0.2
    tau = 1.0

    # ell=0  => -k * f1 + (4/3)hdot = -2 + 4/3*0.1 = -2 + 0.13333...
    out0 = _massless_nu_moment_rhs(0, f_all[0], f_all, hdot, etadot, k, tau)
    assert float(out0) == pytest.approx(-2.0 + (4 / 3) * 0.1, rel=1e-6)

    # ell=1 => (k/3)(F0-2F2)+(4/3)k η_dot = (2/3)*(0-1.0)+(4/3)*2*0.2 = -0.6667 + 0.5333
    out1 = _massless_nu_moment_rhs(1, f_all[1], f_all, hdot, etadot, k, tau)
    expected1 = (2 / 3) * (0 - 2 * 0.5) + (4 / 3) * 2 * 0.2
    assert float(out1) == pytest.approx(expected1, rel=1e-6)


# 3b) Wrap the full massless_nu_rhs and check trivial zero‐state
def test_massless_nu_rhs_zero():
    L = 3
    k = 1.0
    f = jnp.zeros(L + 1)
    nu = MasslessNeutrino(f=f)
    mi = Metric(h=0.0, hdot=0.0, eta=0.0)

    out = massless_nu_rhs(nu, mi, k=k, tau=1.0)
    assert isinstance(out, MasslessNeutrino)
    assert out.f.shape == (L + 1,)
    assert jnp.allclose(out.f, 0.0)
    assert float(out.delta) == pytest.approx(0.0)
    assert float(out.theta(k)) == pytest.approx(0.0)


# -- 4) Test the massive‐neutrino ℓ=0 moment RHS analytically --
def test_massive_moment_rhs_ell0():
    # psi_all of length L+1 = 4, with psi1=1 others=0
    psi_all = jnp.array([0.0, 1.0, 0.0, 0.0])
    q = 0.5
    k = 2.0
    # choose m_nu=0 so eps = sqrt(q^2 + (a*m)^2) = q
    eps = q
    df0 = 1.0
    hdot = 0.1
    etadot = 0.0
    tau = 1.0

    # the formula for ℓ=0 is:  -(q k / eps)*psi1 + (1/6) hdot * df0
    expected = -(q * k / eps) * 1.0 + (1.0 / 6.0) * hdot * df0
    out0 = _massive_nu_moment_rhs(
        0, psi_all[0], psi_all, k, q, eps, df0, hdot, etadot, tau
    )
    assert float(out0) == pytest.approx(expected, rel=1e-6)


# 4b) Test massive_nu_rhs in the trivial zero‐psi limit
def test_massive_nu_rhs_zero():
    nq = 2
    L = 3
    psi = jnp.zeros((nq, L + 1))
    # pick a non‐zero background so we don't divide by zero:
    bg = Background(
        rho_r=0.0, rho_b=0.0, rho_l=0.0, rho_nu=1.0, p_nu=0.5, rho_mnu=0
    )  # not used in massive_nu_rhs

    mi = Metric(h=0.0, hdot=0.0, eta=0.0)
    # zero momentum grid and weights → all quadrature integrals vanish
    q_arr = jnp.zeros(nq)
    w_arr = jnp.zeros(nq)
    df0dlnq = jnp.zeros(nq)

    out = massive_nu_rhs(
        mnu=MassiveNeutrino(psi=psi, delta_rho=0.0, delta_p=0.0, theta=0.0, sigma=0.0),
        bg=bg,
        metric_dot=mi,
        k=1.0,
        a=1.0,
        m_nu=0.1,
        q_arr=q_arr,
        w_arr=w_arr,
        df0dlnq_arr=df0dlnq,
        tau=1.0,
    )

    assert isinstance(out, MassiveNeutrino)
    # shape of dpsi same as psi
    assert out.psi.shape == psi.shape
    # all derivatives & fluid moments should be zero
    assert jnp.allclose(out.psi, 0.0)
    assert float(out.delta_rho) == pytest.approx(0.0)
    assert float(out.delta_p) == pytest.approx(0.0)
    assert float(out.theta) == pytest.approx(0.0)
    assert float(out.sigma) == pytest.approx(0.0)


# -- 5) Finally test that everything is JIT‐compilable --
@pytest.mark.parametrize(
    "fn,args",
    [
        (metric, (2.0, 1.0, Metric(1.0, 2.0, 3.0), 0.0, 0.0, 4.0)),
        (
            radiation,
            (
                Radiation(jnp.zeros(4), jnp.zeros(4), 0.0, 0.0),
                Baryon(0.0, 0.0, 0.0),
                Metric(0.0, 0.0, 0.0),
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
            ),
        ),
        (
            massless_nu_rhs,
            (MasslessNeutrino(jnp.zeros(4)), Metric(0.0, 0.0, 0.0), 1.0, 1.0),
        ),
        (
            massive_nu_rhs,
            (
                MassiveNeutrino(jnp.zeros((2, 4)), 0.0, 0.0, 0.0, 0.0),
                Background(rho_r=0, rho_b=0, rho_l=0, rho_nu=0, rho_mnu=0, p_nu=0),
                Metric(0.0, 0.0, 0.0),
                1.0,
                1.0,
                0.1,
                jnp.zeros(2),
                jnp.zeros(2),
                jnp.zeros(2),
                1.0,
            ),
        ),
    ],
)
def test_jit_compile(fn, args):
    jitted = jit(fn)
    # first call will compile, second call should be instant and equal
    out1 = jitted(*args)
    out2 = fn(*args)
    # We just check that no errors were raised, and the tree‐structure matches
    assert jax.tree.structure(out1) == jax.tree.structure(out2)
