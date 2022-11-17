import numpy as np
import scipy
import jax
import jax.numpy as jnp

def chisqloss(xvals, yvals, yvariances, func, parms):
    return jnp.sum( (func(xvals, parms) - yvals)**2/yvariances )

chisqloss_grad = jax.jit(jax.value_and_grad(chisqloss, argnums = 4), static_argnums = 3)

def _chisqloss_grad_hess(xvals, yvals, yvariances, func, parms):
    def lossf(parms):
        return chisqloss(xvals, yvals, yvariances, func, parms)

    gradf = jax.grad(lossf)
    hessf = jax.jacfwd(gradf)

    loss = lossf(parms)
    grad = gradf(parms)
    hess = hessf(parms)

    return loss, grad, hess

chisqloss_grad_hess = jax.jit(_chisqloss_grad_hess, static_argnums = 3)

def _chisqloss_hessp(xvals, yvals, yvariances, func, parms, p):
    def lossf(parms):
        return chisqloss(xvals, yvals, yvariances, func, parms)

    gradf = jax.grad(lossf)
    hessp = jax.jvp(gradf, (parms,), (p,))[1]
    return hessp

chisqloss_hessp = jax.jit(_chisqloss_hessp, static_argnums = 3)

def fit_hist_jax(hist, func, parmvals, max_iter = 5, edmtol = 1e-5):

    xvals = [jnp.array(center) for center in hist.axes.centers]
    yvals = jnp.array(hist.values())
    yvariances = jnp.array(hist.variances())

    def scipy_loss(parmvals):
        parms = jnp.array(parmvals)
        loss, grad = chisqloss_grad(xvals, yvals, yvariances, func, parms)
        return np.asarray(loss).item(), np.asarray(grad)

    def scipy_hessp(parmvals, p):
        parms = jnp.array(parmvals)
        tangent = jnp.array(p)
        hessp = chisqloss_hessp(xvals, yvals, yvariances, func, parms, tangent)
        return np.asarray(hessp)

    for iiter in range(max_iter):
        res = scipy.optimize.minimize(scipy_loss, parmvals, method = "trust-krylov", jac = True, hessp = scipy_hessp)

        parms = jnp.array(res.x)
        loss, grad, hess = chisqloss_grad_hess(xvals, yvals, yvariances, func, parms)
        loss, grad, hess = np.asarray(loss).item(), np.asarray(grad), np.asarray(hess)

        eigvals = np.linalg.eigvalsh(hess)
        cov = 2.*np.linalg.inv(hess)

        gradv = grad[:, np.newaxis]
        edmval = 0.5*gradv.transpose()@cov@gradv
        edmval = edmval[0][0]

        converged = edmval < edmtol and np.abs(edmval) >= 0. and eigvals[0] > 0.
        if converged:
            break

    status = 1
    covstatus = 1
    if edmval < edmtol and np.abs(edmval) >= 0.:
        status = 0
    if eigvals[0] > 0.:
        covstatus = 0

    res = { "x" : res.x,
           "cov" : cov,
           "status" : status,
           "covstatus" : covstatus,
           "hess_eigvals" : eigvals,
           "edmval" : edmval,
           "chisqval" : loss }

    return res
