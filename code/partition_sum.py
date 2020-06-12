import numpy as np
from scipy.special import chebyu, chebyt

import jax
import jax.numpy as jnp

# Implementation of Chebychev polynomials using jax to be able to differenitate below
def chebyIter(u,x):
    unew = 2. * x * u[0] - u[1]
    u = jax.ops.index_update(u,jax.ops.index[1], u[0])
    u = jax.ops.index_update(u,jax.ops.index[0], unew)
    return u, 0.

def chebyUn(n, x):
    if n == 0:
        return 1.
    if n == 1:
        return 2.*x

    init=jnp.array([2*x,1])
    xs=jnp.array([x] * (n-1))
    res,_ = jax.lax.scan(chebyIter, init, xs)

    return res[0]

def chebyTn(n, x):
    if n == 0:
        return 1.
    if n == 1:
        return x

    init=jnp.array([x,1])
    xs=jnp.array([x] * (n-1))
    res,_ = jax.lax.scan(chebyIter, init, xs)

    return res[0]

def c(l,L,T):
    return jnp.cosh(2./T) / jnp.tanh(2./T) - jnp.cos((jnp.pi * l) / L)

def logZ1(L,T):
    r = 2 * jnp.arange(0,L) + 1
    return jnp.sum(jnp.log(jax.vmap(chebyTn,in_axes=(None,0))(L//2,c(r,L,T))))

def logZ2(L,T):
    r = 2 * jnp.arange(0,L) + 1
    cr = c(r,L,T)
    return jnp.sum(jnp.log(jax.vmap(chebyUn,in_axes=(None,0))(L//2-1,cr))) + jnp.sum(jnp.log(cr[0:L//2]**2-1))

def logZ3(L,T):
    r = 2 * jnp.arange(0,L)
    return jnp.sum(jnp.log(jax.vmap(chebyTn,in_axes=(None,0))(L//2,c(r,L,T))))

def logZ4(L,T):
    r = 2 * np.arange(0,L)
    cr = c(r,L,T)
    ch = jnp.cosh(2./T)**2 - 1./jnp.tanh(2./T)**2 + 0.j
    return jnp.sum(jnp.log(jax.vmap(chebyUn,in_axes=(None,0))(L//2-1,cr))) + jnp.sum(jnp.log(cr[1:L//2]**2-1)) + jnp.log(ch)

def get_log_Z_sum(L,T):
    logZs = jnp.array([logZ1(L,T),logZ2(L,T),logZ3(L,T),logZ4(L,T)])
    logZmax = jnp.max(jnp.real(logZs))
    return jnp.log(jnp.sum(jnp.exp(logZs-logZmax))) + logZmax + L * jnp.log(2.)

# Compute finite size free energy according to https://arxiv.org/pdf/1909.10831.pdf
def free_energy(L,T):
    assert L % 2 ==0, "L has to be even."
    return - jnp.real( (0.5 - 1./L**2) * jnp.log(2) + 0.5 * jnp.log(jnp.sinh(2./T)) + get_log_Z_sum(L,T) / L**2 )

def get_thermodynamics(L,T):
    F = free_energy(L,T)
    E = -jax.grad(free_energy,argnums=1)(L,float(T)) * T**2
    S = (E/T - F) / np.log(2.)

    return (F,E,S)
