import numpy as np
import scipy
import tensorflow as tf

@tf.function
def chisqloss(xvals, yvals, yvariances, func, parms):
    return tf.reduce_sum( (func(xvals, parms) - yvals)**2/yvariances )

@tf.function
def chisqloss_grad(xvals, yvals, yvariances, func, parms):
    loss = chisqloss(xvals, yvals, yvariances, func, parms)
    with tf.GradientTape() as t1:
        loss = chisqloss(xvals, yvals, yvariances, func, parms)
    grad = t1.gradient(loss, parms)
    return loss, grad

@tf.function
def chisqloss_grad_hess(xvals, yvals, yvariances, func, parms):
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            loss = chisqloss(xvals, yvals, yvariances, func, parms)

        grad = t1.gradient(loss, parms)
    hess = t2.jacobian(grad, parms)

    return loss, grad, hess

@tf.function
def chisqloss_hessp(xvals, yvals, yvariances, func, parms, p):
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            loss = chisqloss(xvals, yvals, yvariances, func, parms)

        grad = t1.gradient(loss, parms)
        gradp = tf.reduce_sum(grad*p)
    hessp = t2.gradient(gradp, parms)

    return hessp

def fit_hist(hist, func, parmvals, max_iter = 5, edmtol = 1e-5):

    xvals = [tf.constant(center) for center in hist.axes.centers]
    yvals = tf.constant(hist.values())
    yvariances = tf.constant(hist.variances())
    parms = tf.Variable(parmvals)

    def scipy_loss(parmvals):
        parms.assign(parmvals)
        loss, grad = chisqloss_grad(xvals, yvals, yvariances, func, parms)
        return loss.numpy(), grad.numpy()

    def scipy_hessp(parmvals, p):
        parms.assign(parmvals)
        hessp = chisqloss_hessp(xvals, yvals, yvariances, func, parms, p)
        return hessp.numpy()

    for iiter in range(max_iter):
        res = scipy.optimize.minimize(scipy_loss, parmvals, method = "trust-krylov", jac = True, hessp = scipy_hessp)

        parms.assign(res.x)
        loss, grad, hess = chisqloss_grad_hess(xvals, yvals, yvariances, func, parms)
        loss, grad, hess = loss.numpy(), grad.numpy(), hess.numpy()

        eigvals = np.linalg.eigvalsh(hess)
        cov = np.linalg.inv(hess)

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

    res = { "x" : parms.numpy(),
           "cov" : cov,
           "status" : status,
           "covstatus" : covstatus,
           "hess_eigvals" : eigvals,
           "edmval" : edmval,
           "chisqval" : loss }

    return res



