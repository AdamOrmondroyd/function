import jax
import jax.numpy as jnp
from jax.numpy import pi, ndarray
from jax.dataclasses import pytree_dataclass
from functools import partial

# TODO: Thomson scattering cross-section for electrons - check units


@pytree_dataclass
class Metric:
    h: float
    hdot: float
    eta: float


@pytree_dataclass
class CDM:
    delta: float


@pytree_dataclass
class Baryon:
    delta: float
    theta: float
    t: float  # temperature


@pytree_dataclass
class Radiation:
    f: ndarray  # (lmax+1,)
    g: ndarray  # (lmax+1,)
    theta: float
    delta: float


@pytree_dataclass
class MasslessNeutrino:
    f: ndarray  # (lmax+1,)
    delta: float
    theta: float


@pytree_dataclass
class MassiveNeutrino:
    psi: ndarray  # shape (nq, lmax+1)
    # compute delta/theta/sigma via quadrature later


@pytree_dataclass
class Background:
    rho_r: float
    rho_m: float
    rho_l: float


@pytree_dataclass
class Params:
    h0: float  # Hubble constant today
    omm: float  # Omega_matter
    omb: float  # Omega_baryon
    omr: float  # Omega_radiation
    omnu: float  # Omega_massless_neutrino


@pytree_dataclass
class Cosmology:
    a: float
    hubble: float
    metric: Metric
    cdm: CDM
    b: Baryon
    r: Radiation
    nu: MasslessNeutrino
    mnu: MassiveNeutrino


def metric(a, hubble, metric: Metric, deltaT00, deltaTii, k):
    """Einstein equations."""
    G = 1
    _, hdot, eta = metric.h, metric.hdot, metric.eta  # h currently unused
    etadot = (4 * pi * G * a**2 * deltaT00 + hubble * hdot / 2) / k**2
    hdd = -2 * hubble * hdot + 2 * k**2 * eta - 8 * pi * G * a**2 * deltaTii

    return Metric(hdot, hdd, etadot)


# CDM
def cdm(metric: Metric):
    """δ_c' = -h'/2."""
    return CDM(-metric.h_ / 2)


# Baryons
def baryon(a, hubble, b: Baryon, r: Radiation, ne, sigmaT, rho_r, rho_b, mu, k):
    adot = hubble * a
    KB = 1
    ME = 1
    delta_bdot = -b.theta - metric.hdot / 2

    t_bdot = -2 * metric.h * b.t + 8 / 3 * mu / ME * rho_r / rho_b * a * ne * sigmaT * (
        r.t - b.t
    )

    c_s2 = KB * b.t / mu * (1 - b.t * adot / (3 * t_bdot * a))
    drag = 4 / 3 * rho_r / rho_b * a * ne * sigmaT * (r.theta - b.theta)
    theta_bdot = -hubble * b.theta + c_s2 * k**2 * b.delta + drag

    return Baryon(delta_bdot, theta_bdot, t_bdot)


# Radiation
def _photon_moment(
    ell, fl, gl, f_all, g_all, theta_r, hdot, etadot, ne, sigmaT, a, k, t
):
    L = f_all.shape[0] - 1

    f_prev = jnp.where(ell > 0, f_all[ell - 1], 0.0)
    f_next = jnp.where(ell < L, f_all[ell + 1], 0.0)
    g_prev = jnp.where(ell > 0, g_all[ell - 1], 0.0)
    g_next = jnp.where(ell < L, g_all[ell + 1], 0.0)

    # ℓ = 0: monopole
    def case0(_):
        # F₀′ = −k F₁ + (4/3) h′
        fp = -k * f_next + (4 / 3) * hdot

        # G₀′ = −k G₁ + a n_e σ_T [−G₀ + ½(F₂ + G₀ + G₂)]
        s0 = 0.5 * (f_all[2] + g_all[0] + g_all[2])
        gp = -k * g_next + a * ne * sigmaT * (-g_all[0] + s0)
        return fp, gp

    # ℓ = 1: dipole
    def case1(_):
        # F₁′ = (k/3)(F₀ − 2F₂) + k G₀ + (4/3)k η′ − a n_eσ_T F₁
        fp = (
            (k / 3) * (f_all[0] - 2 * f_all[2])
            + k * g_all[0]
            + (4 / 3) * k * etadot
            - a * ne * sigmaT * f_all[1]
        )

        # G₁′ = (k/3)(G₀ − 2G₂) − a n_eσ_T G₁
        gp = (k / 3) * (g_all[0] - 2 * g_all[2]) - a * ne * sigmaT * g_all[1]
        return fp, gp

    # ℓ = 2: quadrupole / shear
    def case2(_):
        # F₂′ = 8/15 θ_γ
        #      − (3/5) k F₃
        #      + (4/15) h′
        #      + (8/5) η′
        #      − (9/5) a n_eσ_T σ_γ  + (1/10) a n_eσ_T (G₀+G₂)
        # but σ_γ = F₂/2 ⇒ −(9/5) a n_eσ_T σ_γ = −(9/10)a n_eσ_T F₂
        fp = (
            (8 / 15) * theta_r
            - (3 / 5) * k * f_all[3]
            + (4 / 15) * hdot
            + (8 / 5) * etadot
            - (9 / 10) * a * ne * sigmaT * f_all[2]
            + (1 / 10) * a * ne * sigmaT * (g_all[0] + g_all[2])
        )

        # G₂′ = (k/5)(2G₁ − 3G₃)
        #      + a n_eσ_T [ −G₂ + ½(F₂ + G₀ + G₂)*(1/5) ]
        src = 0.5 * (f_all[2] + g_all[0] + g_all[2]) * (1 / 5)
        gp = (k / 5) * (2 * g_all[1] - 3 * g_all[3]) + a * ne * sigmaT * (
            -g_all[2] + src
        )
        return fp, gp

    # 3 ≤ ℓ ≤ L−1: free‐streaming + damping
    def case_general(_):
        fp = (k / (2 * ell + 1)) * (
            ell * f_prev - (ell + 1) * f_next
        ) - a * ne * sigmaT * fl
        gp = (k / (2 * ell + 1)) * (
            ell * g_prev - (ell + 1) * g_next
        ) - a * ne * sigmaT * gl
        return fp, gp

    # ℓ = L (truncation)
    def case_truncate(_):
        fp = k * f_prev - (L + 1) / t * fl - a * ne * sigmaT * fl
        gp = k * g_prev - (L + 1) / t * gl - a * ne * sigmaT * gl
        return fp, gp

    # build branch list: [ℓ=0,1,2] + (L−2) copies of general + [truncate]
    branches = [case0, case1, case2] + [case_general] * (L - 2) + [case_truncate]

    out = jax.lax.switch(ell, branches=branches, operand=None)
    return out


def radiation(r, b, metric, ne, sigmaT, a, k, t):
    delta_rdot = -4 / 3 * r.theta - 2 / 3 * metric.hdot
    theta_rdot = k**2 * (r.delta / 4 - r.f[2] / 2) + a * ne * sigmaT * (
        b.theta - r.theta
    )

    ell = jnp.arange(r.f.shape[0])

    _moment = partial(
        _photon_moment,
        f_all=r.f,
        g_all=r.g,
        theta_r=r.theta,
        hdot=metric.hdot,
        etadot=metric.eta,
        ne=ne,
        sigmaT=sigmaT,
        a=a,
        k=k,
        t=t,
    )
    fdot, gdot = jax.vmap(_moment, in_axes=(0, 0, 0))(ell, r.f, r.g)
    return Radiation(fdot, gdot, theta_rdot, delta_rdot)


# Massless neutrinos
def _massless_nu_moment_rhs(ell, fl, f_all, h_dot, eta_dot, k, tau, theta_nu):
    # ell = multipole index, 0 .. L
    L = f_all.shape[0] - 1

    # safe neighbors
    f_prev = jnp.where(ell > 0, f_all[ell - 1], 0.0)
    f_next = jnp.where(ell < L, f_all[ell + 1], 0.0)

    # ell=0
    def case0(_):
        # dF0 = -k F1 + (4/3) h_dot
        return -k * f_next + (4.0 / 3.0) * h_dot

    # ell=1
    def case1(_):
        # dF1 = (k/3)*(F0 - 2 F2) + (4/3)*k*eta_dot
        return (k / 3.0) * (f_all[0] - 2.0 * f_all[2]) + (4.0 / 3.0) * k * eta_dot

    # ell=2
    def case2(_):
        # dF2 = (8/15)*theta_nu - (3/5)*k*F3
        #      + (4/15)*h_dot + (8/5)*eta_dot
        return (
            (8.0 / 15.0) * theta_nu
            - (3.0 / 5.0) * k * f_all[3]
            + (4.0 / 15.0) * h_dot
            + (8.0 / 5.0) * eta_dot
        )

    # general 3 <= ell < L
    def case_general(_):
        return (k / (2 * ell + 1.0)) * (ell * f_prev - (ell + 1.0) * f_next)

    # ell = L, truncated
    def case_truncate(_):
        # standard truncation term
        return k * f_prev - (L + 1.0) / tau * fl

    branches = [case0, case1, case2] + [case_general] * (L - 2) + [case_truncate]

    return jax.lax.switch(ell, branches, operand=None)


def massless_nu_rhs(nu, metric, k, t):
    """
    nu.F     : array of shape (L+1,)
    nu.delta : array scalar
    nu.theta : array scalar
    metric.h_dot   : array scalar
    metric.eta_dot : array scalar
    """
    f = nu.f
    delta = nu.delta
    theta = nu.theta
    h_dot = metric.h_dot
    eta_dot = metric.eta_dot

    # fluid moments
    delta_dot = -4.0 / 3.0 * theta - 2.0 / 3.0 * h_dot
    theta_dot = k**2 * (0.25 * delta - 0.5 * f[2])

    # prepare for vmap over ell = 0..L
    ells = jnp.arange(f.shape[0])

    moment_fun = partial(
        _massless_nu_moment_rhs,
        f_all=f,
        h_dot=h_dot,
        eta_dot=eta_dot,
        k=k,
        t=t,
        theta_nu=nu.theta,
    )

    fdot = jax.vmap(moment_fun, in_axes=(0, 0))(ells, f)

    return MasslessNeutrino(f=fdot, delta=delta_dot, theta=theta_dot)


# Massive neutrinos
def _massive_nu_moment_rhs(ell, psi_ell, psi_all, k, q, eps, df0, h_dot, eta_dot, tau):
    # ell = 0..L
    L = psi_all.shape[0] - 1

    # safe neighbours
    psi_prev = jnp.where(ell > 0, psi_all[ell - 1], 0.0)
    psi_next = jnp.where(ell < L, psi_all[ell + 1], 0.0)

    # ell = 0
    def case0(_):
        # psi0' = - (q k / eps) psi1 + (1/6) h_dot * dlnf0
        return -(q * k / eps) * psi_next + (1.0 / 6.0) * h_dot * df0

    # ell = 1
    def case1(_):
        # psi1' = (q k)/(3 eps) * (psi0 - 2 psi2)
        return (q * k) / (3.0 * eps) * (psi_all[0] - 2.0 * psi_all[2])

    # ell = 2
    def case2(_):
        # psi2' = (q k)/(5 eps)*(2 psi1 - 3 psi3)
        #        - [ (1/15)h_dot + (2/5)eta_dot ] * dlnf0
        term1 = (q * k) / (5.0 * eps) * (2.0 * psi_all[1] - 3.0 * psi_all[3])
        term2 = -((1.0 / 15.0) * h_dot + (2.0 / 5.0) * eta_dot) * df0
        return term1 + term2

    # 3 <= ell < L
    def case_general(_):
        return (
            (q * k)
            / ((2 * ell + 1.0) * eps)
            * (ell * psi_prev - (ell + 1.0) * psi_next)
        )

    # ell = L (truncate)
    def case_trunc(_):
        # psiL' = (q k/eps) psi_{L-1} - (L+1)/tau * psiL
        return (q * k / eps) * psi_prev - (L + 1.0) / tau * psi_ell

    branches = [case0, case1, case2] + [case_general] * (L - 2) + [case_trunc]
    return jax.lax.switch(ell, branches, operand=None)


def massive_nu_rhs(
    mnu,
    bg,  # background object with bg.rho_nu, bg.p_nu
    metric,  # metric.h_dot, metric.eta_dot
    k,
    a,
    m_nu,
    q_arr,  # shape (nq,)
    w_arr,  # quadrature weights shape (nq,)
    df0dlnq_arr,  # shape (nq,)
    tau,
):
    """
    mnu.psi  : array (nq, L+1) of psi^{(ell)}_q
    Returns a new mnu with fields:
      psi       : d psi / d tau, same shape
      delta_rho : scalar delta rho_nu
      delta_p   : scalar delta P_nu
      theta     : scalar theta_nu
      sigma     : scalar sigma_nu
    """
    psi = mnu.psi
    h_dot = metric.h_dot
    eta_dot = metric.eta_dot
    nq, L1 = psi.shape

    # evolve each momentum bin
    def one_bin_rhs(psi_q, q, w, df0):
        eps = jnp.sqrt(q * q + (a * m_nu) ** 2)
        ells = jnp.arange(L1)
        moment_fun = partial(
            _massive_nu_moment_rhs,
            psi_all=psi_q,
            k=k,
            q=q,
            eps=eps,
            df0=df0,
            h_dot=h_dot,
            eta_dot=eta_dot,
            tau=tau,
        )
        # vmap over ell and psi_ell
        return jax.vmap(moment_fun, in_axes=(0, 0))(ells, psi_q)

    # dpsi has shape (nq, L+1)
    dpsi = jax.vmap(one_bin_rhs, in_axes=(0, 0, 0, 0))(psi, q_arr, w_arr, df0dlnq_arr)

    # now form the fluid moments via quadrature
    eps_arr = jnp.sqrt(q_arr**2 + (a * m_nu) ** 2)
    prefac = 4.0 * jnp.pi / a**4

    # density perturbation
    delta_rho = prefac * jnp.sum(w_arr * q_arr**2 * eps_arr * psi[:, 0])

    # pressure perturbation
    delta_p = prefac / 3.0 * jnp.sum(w_arr * q_arr**4 / eps_arr * psi[:, 1])

    # (rho+P) theta
    theta_num = 4.0 * jnp.pi * k / a**4 * jnp.sum(w_arr * q_arr**3 * psi[:, 1])

    # (rho+P) sigma
    sigma_num = (
        8.0 * jnp.pi / (3.0 * a**4) * jnp.sum(w_arr * q_arr**4 / eps_arr * psi[:, 2])
    )

    rho_plus_p = bg.rho_nu + bg.p_nu

    theta = theta_num / rho_plus_p
    sigma = sigma_num / rho_plus_p

    return MassiveNeutrino(
        psi=dpsi, delta_rho=delta_rho, delta_p=delta_p, theta=theta, sigma=sigma
    )


def eb(t, state: Cosmology, k, params):
    a, hubble = state.a, state.hubble
    # background
    hubble_dot = None

    # TODO: build these
    deltaT00 = None
    deltaTii = None
    metric_dot = metric(a, hubble, state.metric, deltaT00, deltaTii, k)

    # derivatives to return
    cdm_dot = cdm(metric)

    # mu needs to come from somewhere
    mu = None
    b_dot = baryon(
        a,
        hubble,
        state.b,
        state.r,
        params.ne,
        params.sigma_t,
        params.rho_r,
        params.rho_b,
        mu,
        k,
    )

    # ne needs to come from somewhere
    ne = None
    r_dot = radiation(state.r, state.b, state.metric, ne, params.sigmaT, a, k, t)

    nu_dot = massless_nu_rhs(state.nu, state.metric, k, t)
    mnu_dot = massive_nu_rhs(
        state.mnu, state.metric, params.w(k), a, params.q, params.w, params.df0dlnq, t
    )

    Cosmology(
        a=hubble * a,
        hubble=hubble_dot,  # from background
        metric=metric_dot,
        cdm=cdm_dot,
        b=b_dot,
        r=r_dot,
        nu=nu_dot,
        mnu=mnu_dot,
    )
