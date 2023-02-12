import numpy as np
import scipy
import tensorflow as tf
import math

@tf.function
def val_grad(func, *args, **kwargs):
    xdep = args[0]
    with tf.GradientTape() as t1:
        t1.watch(xdep)
        val = func(*args, **kwargs)
    grad = t1.gradient(val, xdep)
    return val, grad

#TODO forward-over-reverse also here?
@tf.function
def val_grad_hess(func, *args, **kwargs):
    xdep = args[0]
    with tf.GradientTape() as t2:
        t2.watch(xdep)
        with tf.GradientTape() as t1:
            t1.watch(xdep)
            val = func(*args, **kwargs)
        grad = t1.gradient(val, xdep)
    hess = t2.jacobian(grad, xdep)

    return val, grad, hess    

@tf.function
def val_grad_hessp(func, p, *args, **kwargs):
    xdep = args[0]
    with tf.autodiff.ForwardAccumulator(xdep, p) as acc:
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(xdep)
            val = func(*args, **kwargs)
        grad = grad_tape.gradient(val, xdep)
    hessp = acc.jvp(grad)
  
    return val, grad, hessp

def chisq_loss(parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes = None, *args):
    fvals = func(xvals, parms, *args)

    # exclude zero-variance bins
    variances_safe = tf.where(yvariances == 0., tf.ones_like(yvariances), yvariances)
    chisqv = (fvals - yvals)**2/variances_safe
    chisqv_safe = tf.where(yvariances == 0., tf.zeros_like(chisqv), chisqv)
    return tf.reduce_sum(chisqv_safe)

def chisq_normalized_loss(parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes = None, *args):
    fvals = func(xvals, parms, *args)
    norm = tf.reduce_sum(fvals, keepdims=True, axis = norm_axes)
    sumw = tf.reduce_sum(yvals, keepdims=True, axis = norm_axes)
    if norm_axes is None:
        for xwidth in xwidths:
            norm *= xwidth
    else:
        for norm_axis in norm_axes:
            norm *= xwidths[norm_axis]
            
    # exclude zero-variance bins
    variances_safe = tf.where(yvariances == 0., tf.ones_like(yvariances), yvariances)
    chisqv = (sumw*fvals/norm - yvals)**2/variances_safe
    chisqv_safe = tf.where(yvariances == 0., tf.zeros_like(chisqv), chisqv)
    return tf.reduce_sum(chisqv_safe)

def nll_loss(parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes = None, *args):
    fvals = func(xvals, parms, *args)

    # compute overall scaling needed to restore mean == variance condition
    yval_total = tf.reduce_sum(yvals, keepdims = True, axis = norm_axes)
    variance_total = tf.reduce_sum(yvariances, keepdims = True, axis = norm_axes)
    isnull_total = variance_total == 0.
    variance_total_safe = tf.where(isnull_total, tf.ones_like(variance_total), variance_total)
    scale_total = yval_total/variance_total_safe
    scale_total_safe = tf.where(isnull_total, tf.ones_like(scale_total), scale_total)

    # skip likelihood calculation for empty bins to avoid inf or nan
    # compute per-bin scaling needed to restore mean == variance condition, falling
    # back to overall scaling for empty bins
    isnull = tf.logical_or(yvals == 0., yvariances == 0.)
    variances_safe = tf.where(isnull, tf.ones_like(yvariances), yvariances)
    scale = yvals/variances_safe
    scale_safe = tf.where(isnull, scale_total_safe*tf.ones_like(scale), scale)

    norm = tf.reduce_sum(scale_safe*fvals, keepdims=True, axis = norm_axes)
    if norm_axes is None:
        for xwidth in xwidths:
            norm *= xwidth
    else:
        for norm_axis in norm_axes:
            norm *= xwidths[norm_axis]

    fvalsnorm = fvals/norm

    fvalsnorm_safe = tf.where(isnull, tf.ones_like(fvalsnorm), fvalsnorm)
    nllv = -scale_safe*yvals*tf.math.log(scale_safe*fvalsnorm_safe)
    nllv_safe = tf.where(isnull, tf.zeros_like(nllv), nllv)
    nllsum = tf.reduce_sum(nllv_safe)
    return nllsum

def nll_loss_bin_integrated(parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes = None, *args):
    #TODO reduce code duplication with nll_loss_bin

    #FIXME this is only defined in 1D for now
    cdfvals = func(xedges, parms, *args)
    bin_integrals = cdfvals[1:] - cdfvals[:-1]
    bin_integrals = tf.maximum(bin_integrals, tf.zeros_like(bin_integrals))

    fvals = bin_integrals

    # compute overall scaling needed to restore mean == variance condition
    yval_total = tf.reduce_sum(yvals, keepdims = True, axis = norm_axes)
    variance_total = tf.reduce_sum(yvariances, keepdims = True, axis = norm_axes)
    isnull_total = variance_total == 0.
    variance_total_safe = tf.where(isnull_total, tf.ones_like(variance_total), variance_total)
    scale_total = yval_total/variance_total_safe
    scale_total_safe = tf.where(isnull_total, tf.ones_like(scale_total), scale_total)

    # skip likelihood calculation for empty bins to avoid inf or nan
    # compute per-bin scaling needed to restore mean == variance condition, falling
    # back to overall scaling for empty bins
    isnull = tf.logical_or(yvals == 0., yvariances == 0.)
    variances_safe = tf.where(isnull, tf.ones_like(yvariances), yvariances)
    scale = yvals/variances_safe
    scale_safe = tf.where(isnull, scale_total_safe*tf.ones_like(scale), scale)

    norm = tf.reduce_sum(scale_safe*fvals, keepdims=True, axis = norm_axes)

    fvalsnorm = fvals/norm

    fvalsnorm_safe = tf.where(isnull, tf.ones_like(fvalsnorm), fvalsnorm)
    nllv = -scale_safe*yvals*tf.math.log(scale_safe*fvalsnorm_safe)
    nllv_safe = tf.where(isnull, tf.zeros_like(nllv), nllv)
    nllsum = tf.reduce_sum(nllv_safe)
    return nllsum

def chisq_loss_bin_integrated(parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes = None, *args):
    #FIXME this is only defined in 1D for now
    cdfvals = func(xedges, parms, *args)
    bin_integrals = cdfvals[1:] - cdfvals[:-1]
    bin_integrals = tf.maximum(bin_integrals, tf.zeros_like(bin_integrals))

    fvals = bin_integrals

    # exclude zero-variance bins
    variances_safe = tf.where(yvariances == 0., tf.ones_like(yvariances), yvariances)
    chisqv = (fvals - yvals)**2/variances_safe
    chisqv_safe = tf.where(yvariances == 0., tf.zeros_like(chisqv), chisqv)
    chisqsum = tf.reduce_sum(chisqv_safe)

    return chisqsum


def fit_hist(hist, func, initial_parmvals, max_iter = 5, edmtol = 1e-5, mode = "chisq", norm_axes = None, args = ()):

    dtype = tf.float64

    xvals = [tf.constant(center, dtype=dtype) for center in hist.axes.centers]
    xwidths = [tf.constant(width, dtype=dtype) for width in hist.axes.widths]
    xedges = [tf.constant(edge, dtype=dtype) for edge in hist.axes.edges]
    yvals = tf.constant(hist.values(), dtype=dtype)
    yvariances = tf.constant(hist.variances(), dtype=dtype)
    
    covscale = 1.
    if mode == "chisq":
        floss = chisq_loss
        covscale = 2.
    elif mode == "nll":
        floss = nll_loss
    elif mode == "nll_bin_integrated":
        floss = nll_loss_bin_integrated
    elif mode == "chisq_normalized":
        floss = chisq_normalized_loss
        covscale = 2.
    elif mode == "chisq_loss_bin_integrated":
        floss = chisq_loss_bin_integrated
        covscale = 2.
    elif mode == "nll_extended":
        raise Exception("Not Implemented")
    else:
        raise Exception("unsupported mode")

    def scipy_loss(parmvals, *args):
        parms = tf.constant(parmvals, dtype=dtype)
        loss, grad = val_grad(floss, parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes, *args)
        return loss.numpy(), grad.numpy()

    def scipy_hessp(parmvals, p, *args):
        parms = tf.constant(parmvals, dtype=dtype)
        loss, grad, hessp = val_grad_hessp(floss, p, parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes, *args)
        return hessp.numpy()

    current_parmvals = initial_parmvals
    for iiter in range(max_iter):

        res = scipy.optimize.minimize(scipy_loss, current_parmvals, method = "trust-krylov", jac = True, hessp = scipy_hessp, args = args)

        current_parmvals = res.x

        parms = tf.constant(current_parmvals, dtype=dtype)
        loss, grad, hess = val_grad_hess(floss, parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes, *args)
        loss, grad, hess = loss.numpy(), grad.numpy(), hess.numpy()
        
        try:
            eigvals = np.linalg.eigvalsh(hess)
            gradv = grad[:, np.newaxis]
            edmval = 0.5*gradv.transpose()@np.linalg.solve(hess, gradv)
            edmval = edmval[0][0]
        except np.linalg.LinAlgError:
            eigvals = np.zeros_like(grad)
            edmval = 99.

        converged = edmval < edmtol and np.abs(edmval) >= 0. and eigvals[0] > 0.
        if converged:
            break

    status = 1
    covstatus = 1
    
    if edmval < edmtol and edmval >= -0.:
        status = 0
    if eigvals[0] > 0.:
        covstatus = 0

    try:
        cov = covscale*np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.zeros_like(hess)
        covstatus = 1

    res = { "x" : current_parmvals,
           "cov" : cov,
           "status" : status,
           "covstatus" : covstatus,
           "hess_eigvals" : eigvals,
           "edmval" : edmval,
           "loss_val" : loss }

    return res

