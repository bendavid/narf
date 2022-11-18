import numpy as np
import scipy
import tensorflow as tf

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

def chisq_loss(parms, xvals, xwidths, yvals, yvariances, func, norm_axes = None):
    fvals = func(xvals, parms)

    # exclude zero-variance bins
    variances_safe = tf.where(yvariances == 0., tf.ones_like(yvariances), yvariances)
    chisqv = (fvals - yvals)**2/variances_safe
    chisqv_safe = tf.where(yvariances == 0., tf.zeros_like(chisqv), chisqv)
    return tf.reduce_sum(chisqv_safe)

def chisq_normalized_loss(parms, xvals, xwidths, yvals, yvariances, func, norm_axes = None):
    fvals = func(xvals, parms)
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

def nll_loss(parms, xvals, xwidths, yvals, yvariances, func, norm_axes = None):
    fvals = func(xvals, parms)
    norm = tf.reduce_sum(fvals, keepdims=True, axis = norm_axes)
    if norm_axes is None:
        for xwidth in xwidths:
            norm *= xwidth
    else:
        for norm_axis in norm_axes:
            norm *= xwidths[norm_axis]
            
    return -tf.reduce_sum(yvals*tf.math.log(fvals/norm))

def fit_hist(hist, func, initial_parmvals, max_iter = 5, edmtol = 1e-5, mode = "chisq", norm_axes = None):

    dtype = tf.float64

    xvals = [tf.constant(center, dtype=dtype) for center in hist.axes.centers]
    xwidths = [tf.constant(width, dtype=dtype) for width in hist.axes.widths]
    yvals = tf.constant(hist.values(), dtype=dtype)
    yvariances = tf.constant(hist.variances(), dtype=dtype)
    
    covscale = 1.
    if mode == "chisq":
        floss = chisq_loss
        covscale = 2.
    elif mode == "nll":
        floss = nll_loss
    elif mode == "chisq_normalized":
        floss = chisq_normalized_loss
        covscale = 2.
    elif mode == "nll_extended":
        raise Exception("Not Implemented")
    else:
        raise Exception("unsupported mode")

    def scipy_loss(parmvals):
        parms = tf.constant(parmvals, dtype=dtype)
        loss, grad = val_grad(floss, parms, xvals, xwidths, yvals, yvariances, func, norm_axes)
        return loss.numpy(), grad.numpy()

    def scipy_hessp(parmvals, p):
        parms = tf.constant(parmvals, dtype=dtype)
        loss, grad, hessp = val_grad_hessp(floss, p, parms, xvals, xwidths, yvals, yvariances, func, norm_axes)
        return hessp.numpy()

    current_parmvals = initial_parmvals
    for iiter in range(max_iter):
        res = scipy.optimize.minimize(scipy_loss, current_parmvals, method = "trust-krylov", jac = True, hessp = scipy_hessp)

        current_parmvals = res.x

        parms = tf.constant(current_parmvals, dtype=dtype)
        loss, grad, hess = val_grad_hess(floss, parms, xvals, xwidths, yvals, yvariances, func, norm_axes)
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
    
    if edmval < edmtol and np.abs(edmval) >= 0.:
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

