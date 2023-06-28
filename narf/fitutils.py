import numpy as np
import scipy
import tensorflow as tf
import math

from numpy import (zeros, where, diff, floor, minimum, maximum, array, concatenate, logical_or, logical_xor,
                   sqrt)

def pchip_interpolate(xi, yi, x, axis=-1):
    '''
        Functionality:
            1D PCHP interpolation
        Authors:
            Michael Taylor <mtaylor@atlanticsciences.com>
            Mathieu Virbel <mat@meltingrocks.com>
        Link:
            https://gist.github.com/tito/553f1135959921ce6699652bf656150d
            https://github.com/tensorflow/tensorflow/issues/46609#issuecomment-774573667
    '''

    tensors = [xi, yi]
    nelems = [tensor.shape.num_elements() for tensor in tensors]

    max_nelems = max(nelems)
    broadcast_shape = tensors[nelems.index(max_nelems)].shape

    ndim = len(broadcast_shape)

    if xi.shape.num_elements() < max_nelems:
        xi = tf.broadcast_to(xi, broadcast_shape)
    if yi.shape.num_elements() < max_nelems:
        yi = tf.broadcast_to(yi, broadcast_shape)

    # # permutation to move the selected axis to the end
    selaxis = axis
    if axis < 0:
        selaxis = ndim + axis

    permfwd = list(range(ndim))
    permfwd.remove(selaxis)
    permfwd.append(selaxis)

    # reverse permutation to restore the original axis order
    permrev = list(range(ndim))
    permrev.remove(ndim-1)
    permrev.insert(selaxis, ndim-1)

    xi = tf.transpose(xi, permfwd)
    yi = tf.transpose(yi, permfwd)
    x = tf.transpose(x, permfwd)
    axis = -1

    xi_steps = tf.experimental.numpy.diff(xi, axis=axis)


    x_steps = tf.experimental.numpy.diff(x, axis=axis)

    idx_zero_constant = tf.constant(0, dtype=tf.int64)
    float64_zero_constant = tf.constant(0., dtype=tf.float64)

    x_compare = x[...,None] < xi[..., None, :]
    x_compare_all = tf.math.reduce_all(x_compare, axis=-1)
    x_compare_none = tf.math.reduce_all(tf.logical_not(x_compare), axis=-1)
    x_index = tf.argmax(x_compare, axis = -1) - 1

    x_index = tf.where(x_compare_all, idx_zero_constant, x_index)
    x_index = tf.where(x_compare_none, tf.constant(xi.shape[axis]-2, dtype=tf.int64), x_index)

    # Calculate gradients d
    h = tf.experimental.numpy.diff(xi, axis=axis)

    d = tf.zeros_like(xi)

    delta = tf.experimental.numpy.diff(yi, axis=axis) / h
    # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
    # recipe

    slice01 = [slice(None)]*ndim
    slice01[axis] = slice(0,1)
    slice01 = tuple(slice01)

    slice0m1 = [slice(None)]*ndim
    slice0m1[axis] = slice(0,-1)
    slice0m1 = tuple(slice0m1)

    slice1 = [slice(None)]*ndim
    slice1[axis] = slice(1,None)
    slice1 = tuple(slice1)

    slicem1 = [slice(None)]*ndim
    slicem1[axis] = slice(-1,None)
    slicem1 = tuple(slicem1)

    d = tf.concat(
        (delta[slice01], 3 * (h[slice0m1] + h[slice1]) / ((h[slice0m1] + 2 * h[slice1]) / delta[slice0m1] +
                                                (2 * h[slice0m1] + h[slice1]) / delta[slice1]), delta[slicem1]), axis=axis)

    false_shape = [*xi.shape]
    false_shape[axis] = 1
    false_const = tf.fill(false_shape, False)

    mask = tf.concat((false_const, tf.math.logical_xor(delta[slice0m1] > 0, delta[slice1] > 0), false_const), axis=axis)
    d = tf.where(mask, float64_zero_constant, d)

    mask = tf.math.logical_or(tf.concat((false_const, delta == 0), axis=axis), tf.concat((delta == 0, false_const), axis=axis))
    d = tf.where(mask, float64_zero_constant, d)

    xiperm = xi
    yiperm = yi
    dperm = d
    hperm = h

    nbatch = ndim - 1

    # # in principle could use tf.gather here instead but this doesn't play nice with onnx
    #
    x_index = x_index[..., None]

    xi_xidx = tf.gather_nd(xiperm, x_index, batch_dims=nbatch)
    xi_1pxidx = tf.gather_nd(xiperm, 1 + x_index, batch_dims=nbatch)
    yi_xidx = tf.gather_nd(yiperm, x_index, batch_dims=nbatch)
    yi_1pxidx = tf.gather_nd(yiperm, 1 + x_index, batch_dims=nbatch)
    d_xidx = tf.gather_nd(dperm, x_index, batch_dims=nbatch)
    d_1pxidx = tf.gather_nd(dperm, 1 + x_index, batch_dims=nbatch)
    h_xidx = tf.gather_nd(hperm, x_index, batch_dims=nbatch)

    dxxi = x - xi_xidx
    dxxid = x - xi_1pxidx
    dxxi2 = tf.math.pow(dxxi, 2)
    dxxid2 = tf.math.pow(dxxid, 2)

    y = (2 / tf.math.pow(h_xidx, 3) *
            (yi_xidx * dxxid2 * (dxxi + h_xidx / 2) - yi_1pxidx * dxxi2 *
            (dxxid - h_xidx / 2)) + 1 / tf.math.pow(h_xidx, 2) *
            (d_xidx * dxxid2 * dxxi + d_1pxidx * dxxi2 * dxxid))

    y = tf.transpose(y, permrev)

    return y

def pchip_interpolate_np(xi, yi, x, mode="mono", verbose=False):
    '''
        Functionality:
            1D PCHP interpolation
        Authors:
            Michael Taylor <mtaylor@atlanticsciences.com>
            Mathieu Virbel <mat@meltingrocks.com>
        Link:
            https://gist.github.com/tito/553f1135959921ce6699652bf656150d
    '''

    if mode not in ("mono", "quad"):
        raise ValueError("Unrecognized mode string")

    # Search for [xi,xi+1] interval for each x
    xi = xi.astype("double")
    yi = yi.astype("double")

    x_index = zeros(len(x), dtype="int")
    xi_steps = diff(xi)
    if not all(xi_steps > 0):
        raise ValueError("x-coordinates are not in increasing order.")

    x_steps = diff(x)
    if xi_steps.max() / xi_steps.min() < 1.000001:
        # uniform input grid
        if verbose:
            print("pchip: uniform input grid")
        xi_start = xi[0]
        xi_step = (xi[-1] - xi[0]) / (len(xi) - 1)
        x_index = minimum(maximum(floor((x - xi_start) / xi_step).astype(int), 0), len(xi) - 2)

        # Calculate gradients d
        h = (xi[-1] - xi[0]) / (len(xi) - 1)
        d = zeros(len(xi), dtype="double")
        if mode == "quad":
            # quadratic polynomial fit
            d[[0]] = (yi[1] - yi[0]) / h
            d[[-1]] = (yi[-1] - yi[-2]) / h
            d[1:-1] = (yi[2:] - yi[0:-2]) / 2 / h
        else:
            # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
            # recipe
            delta = diff(yi) / h
            d = concatenate((delta[0:1], 2 / (1 / delta[0:-1] + 1 / delta[1:]), delta[-1:]))
            d[concatenate((array([False]), logical_xor(delta[0:-1] > 0, delta[1:] > 0), array([False])))] = 0
            d[logical_or(concatenate((array([False]), delta == 0)), concatenate(
                (delta == 0, array([False]))))] = 0
        # Calculate output values y
        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = pow(dxxi, 2)
        dxxid2 = pow(dxxid, 2)
        y = (2 / pow(h, 3) * (yi[x_index] * dxxid2 * (dxxi + h / 2) - yi[1 + x_index] * dxxi2 *
                              (dxxid - h / 2)) + 1 / pow(h, 2) *
             (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))
    else:
        # not uniform input grid
        if (x_steps.max() / x_steps.min() < 1.000001 and x_steps.max() / x_steps.min() > 0.999999):
            # non-uniform input grid, uniform output grid
            if verbose:
                print("pchip: non-uniform input grid, uniform output grid")
            x_decreasing = x[-1] < x[0]
            if x_decreasing:
                x = x[::-1]
            x_start = x[0]
            x_step = (x[-1] - x[0]) / (len(x) - 1)
            x_indexprev = -1
            for xi_loop in range(len(xi) - 2):
                x_indexcur = max(int(floor((xi[1 + xi_loop] - x_start) / x_step)), -1)
                x_index[1 + x_indexprev:1 + x_indexcur] = xi_loop
                x_indexprev = x_indexcur
            x_index[1 + x_indexprev:] = len(xi) - 2
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        elif all(x_steps > 0) or all(x_steps < 0):
            # non-uniform input/output grids, output grid monotonic
            if verbose:
                print("pchip: non-uniform in/out grid, output grid monotonic")
            x_decreasing = x[-1] < x[0]
            if x_decreasing:
                x = x[::-1]
            x_len = len(x)
            x_loop = 0
            for xi_loop in range(len(xi) - 1):
                while x_loop < x_len and x[x_loop] < xi[1 + xi_loop]:
                    x_index[x_loop] = xi_loop
                    x_loop += 1
            x_index[x_loop:] = len(xi) - 2
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        else:
            # non-uniform input/output grids, output grid not monotonic
            if verbose:
                print("pchip: non-uniform in/out grids, " "output grid not monotonic")
            for index in range(len(x)):
                loc = where(x[index] < xi)[0]
                if loc.size == 0:
                    x_index[index] = len(xi) - 2
                elif loc[0] == 0:
                    x_index[index] = 0
                else:
                    x_index[index] = loc[0] - 1
        # Calculate gradients d
        h = diff(xi)
        d = zeros(len(xi), dtype="double")
        delta = diff(yi) / h
        if mode == "quad":
            # quadratic polynomial fit
            d[[0, -1]] = delta[[0, -1]]
            d[1:-1] = (delta[1:] * h[0:-1] + delta[0:-1] * h[1:]) / (h[0:-1] + h[1:])
        else:
            # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
            # recipe
            d = concatenate(
                (delta[0:1], 3 * (h[0:-1] + h[1:]) / ((h[0:-1] + 2 * h[1:]) / delta[0:-1] +
                                                      (2 * h[0:-1] + h[1:]) / delta[1:]), delta[-1:]))
            d[concatenate((array([False]), logical_xor(delta[0:-1] > 0, delta[1:] > 0), array([False])))] = 0
            d[logical_or(concatenate((array([False]), delta == 0)), concatenate(
                (delta == 0, array([False]))))] = 0
        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = pow(dxxi, 2)
        dxxid2 = pow(dxxid, 2)
        y = (2 / pow(h[x_index], 3) *
             (yi[x_index] * dxxid2 * (dxxi + h[x_index] / 2) - yi[1 + x_index] * dxxi2 *
              (dxxid - h[x_index] / 2)) + 1 / pow(h[x_index], 2) *
             (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))
    return y



def pchip_interpolate_np_forced(xi, yi, x, mode="mono", verbose=False):
    '''
        Functionality:
            1D PCHP interpolation
        Authors:
            Michael Taylor <mtaylor@atlanticsciences.com>
            Mathieu Virbel <mat@meltingrocks.com>
        Link:
            https://gist.github.com/tito/553f1135959921ce6699652bf656150d
    '''

    if mode not in ("mono", "quad"):
        raise ValueError("Unrecognized mode string")

    # Search for [xi,xi+1] interval for each x
    xi = xi.astype("double")
    yi = yi.astype("double")

    x_index = zeros(len(x), dtype="int")
    xi_steps = diff(xi)
    if not all(xi_steps > 0):
        raise ValueError("x-coordinates are not in increasing order.")

    x_steps = diff(x)
    # if xi_steps.max() / xi_steps.min() < 1.000001:
    if False:
        # uniform input grid
        if verbose:
            print("pchip: uniform input grid")
        xi_start = xi[0]
        xi_step = (xi[-1] - xi[0]) / (len(xi) - 1)
        x_index = minimum(maximum(floor((x - xi_start) / xi_step).astype(int), 0), len(xi) - 2)

        # Calculate gradients d
        h = (xi[-1] - xi[0]) / (len(xi) - 1)
        d = zeros(len(xi), dtype="double")
        if mode == "quad":
            # quadratic polynomial fit
            d[[0]] = (yi[1] - yi[0]) / h
            d[[-1]] = (yi[-1] - yi[-2]) / h
            d[1:-1] = (yi[2:] - yi[0:-2]) / 2 / h
        else:
            # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
            # recipe
            delta = diff(yi) / h
            d = concatenate((delta[0:1], 2 / (1 / delta[0:-1] + 1 / delta[1:]), delta[-1:]))
            d[concatenate((array([False]), logical_xor(delta[0:-1] > 0, delta[1:] > 0), array([False])))] = 0
            d[logical_or(concatenate((array([False]), delta == 0)), concatenate(
                (delta == 0, array([False]))))] = 0
        # Calculate output values y
        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = pow(dxxi, 2)
        dxxid2 = pow(dxxid, 2)
        y = (2 / pow(h, 3) * (yi[x_index] * dxxid2 * (dxxi + h / 2) - yi[1 + x_index] * dxxi2 *
                              (dxxid - h / 2)) + 1 / pow(h, 2) *
             (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))
    else:
        # not uniform input grid
        # if (x_steps.max() / x_steps.min() < 1.000001 and x_steps.max() / x_steps.min() > 0.999999):
        if False:
            # non-uniform input grid, uniform output grid
            if verbose:
                print("pchip: non-uniform input grid, uniform output grid")
            x_decreasing = x[-1] < x[0]
            if x_decreasing:
                x = x[::-1]
            x_start = x[0]
            x_step = (x[-1] - x[0]) / (len(x) - 1)
            x_indexprev = -1
            for xi_loop in range(len(xi) - 2):
                x_indexcur = max(int(floor((xi[1 + xi_loop] - x_start) / x_step)), -1)
                x_index[1 + x_indexprev:1 + x_indexcur] = xi_loop
                x_indexprev = x_indexcur
            x_index[1 + x_indexprev:] = len(xi) - 2
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        # elif all(x_steps > 0) or all(x_steps < 0):
        elif True:
            # non-uniform input/output grids, output grid monotonic
            if verbose:
                print("pchip: non-uniform in/out grid, output grid monotonic")
            # x_decreasing = x[-1] < x[0]
            x_decreasing = False
            if x_decreasing:
                x = x[::-1]
            x_len = len(x)
            x_loop = 0
            for xi_loop in range(len(xi) - 1):
                while x_loop < x_len and x[x_loop] < xi[1 + xi_loop]:
                    x_index[x_loop] = xi_loop
                    x_loop += 1
            x_index[x_loop:] = len(xi) - 2

            print("np_forced x_index", x_index)
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        else:
            # non-uniform input/output grids, output grid not monotonic
            if verbose:
                print("pchip: non-uniform in/out grids, " "output grid not monotonic")
            for index in range(len(x)):
                loc = where(x[index] < xi)[0]
                if loc.size == 0:
                    x_index[index] = len(xi) - 2
                elif loc[0] == 0:
                    x_index[index] = 0
                else:
                    x_index[index] = loc[0] - 1
        # Calculate gradients d
        h = diff(xi)
        d = zeros(len(xi), dtype="double")
        delta = diff(yi) / h
        if mode == "quad":
            # quadratic polynomial fit
            d[[0, -1]] = delta[[0, -1]]
            d[1:-1] = (delta[1:] * h[0:-1] + delta[0:-1] * h[1:]) / (h[0:-1] + h[1:])
        else:
            # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
            # recipe
            d = concatenate(
                (delta[0:1], 3 * (h[0:-1] + h[1:]) / ((h[0:-1] + 2 * h[1:]) / delta[0:-1] +
                                                      (2 * h[0:-1] + h[1:]) / delta[1:]), delta[-1:]))
            d[concatenate((array([False]), logical_xor(delta[0:-1] > 0, delta[1:] > 0), array([False])))] = 0
            d[logical_or(concatenate((array([False]), delta == 0)), concatenate(
                (delta == 0, array([False]))))] = 0
        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = pow(dxxi, 2)
        dxxid2 = pow(dxxid, 2)
        y = (2 / pow(h[x_index], 3) *
             (yi[x_index] * dxxid2 * (dxxi + h[x_index] / 2) - yi[1 + x_index] * dxxi2 *
              (dxxid - h[x_index] / 2)) + 1 / pow(h[x_index], 2) *
             (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))
    return y

def qparms_to_quantiles(qparms, x_low = 0., x_high = 1., axis = -1):
    deltax = tf.exp(qparms)
    sumdeltax = tf.math.reduce_sum(deltax, axis=axis, keepdims=True)

    deltaxnorm = deltax/sumdeltax

    x0shape = list(deltaxnorm.shape)
    x0shape[axis] = 1
    x0 = tf.fill(x0shape, x_low)

    deltaxfull = (x_high - x_low)*deltaxnorm
    deltaxfull = tf.concat([x0, deltaxfull], axis = axis)

    quants = tf.math.cumsum(deltaxfull, axis=axis)

    return quants



def quantiles_to_qparms(quants, quant_errs = None, x_low = 0., x_high = 1., axis = -1):

    deltaxfull = tf.experimental.numpy.diff(quants, axis=axis)
    deltaxnorm = deltaxfull/(x_high - x_low)
    qparms = tf.math.log(deltaxnorm)

    if quant_errs is not None:
        quant_vars = tf.math.square(quant_errs)

        ndim = len(quant_errs.shape)

        slicem1 = [slice(None)]*ndim
        slicem1[axis] = slice(None,-1)
        slicem1 = tuple(slicem1)

        slice1 = [slice(None)]*ndim
        slice1[axis] = slice(1,None)
        slice1 = tuple(slice1)

        deltaxfull_vars = quant_vars[slice1] + quant_vars[slicem1]
        deltaxfull_errs = tf.math.sqrt(deltaxfull_vars)

        qparm_errs = deltaxfull_errs/deltaxfull

        return qparms, qparm_errs
    else:
        return qparms


def hist_to_quantiles(h, quant_cdfvals, axis = -1):
    dtype = tf.float64

    xvals = [tf.constant(center, dtype=dtype) for center in h.axes.centers]
    xwidths = [tf.constant(width, dtype=dtype) for width in h.axes.widths]
    xedges = [tf.constant(edge, dtype=dtype) for edge in h.axes.edges]
    yvals = tf.constant(h.values(), dtype=dtype)

    if not isinstance(quant_cdfvals, tf.Tensor):
        quant_cdfvals = tf.constant(quant_cdfvals, tf.float64)

    x_flat = tf.reshape(xedges[axis], (-1,))
    x_low = x_flat[0]
    x_high = x_flat[-1]

    hist_cdfvals = tf.cumsum(yvals, axis=axis)/tf.reduce_sum(yvals, axis=axis, keepdims=True)

    x0shape = list(hist_cdfvals.shape)
    x0shape[axis] = 1
    x0 = tf.zeros(x0shape, dtype = dtype)

    hist_cdfvals = tf.concat([x0, hist_cdfvals], axis=axis)

    quants = pchip_interpolate(hist_cdfvals, xedges[axis], quant_cdfvals, axis=axis)

    quants = tf.where(quant_cdfvals == 0., x_low, quants)
    quants = tf.where(quant_cdfvals == 1., x_high, quants)

    ntot = tf.math.reduce_sum(yvals, axis=axis, keepdims=True)

    quant_cdf_bar = ntot/(1.+ntot)*(quant_cdfvals + 0.5/ntot)
    quant_cdfval_errs = ntot/(1.+ntot)*tf.math.sqrt(quant_cdfvals*(1.-quant_cdfvals)/ntot + 0.25/ntot/ntot)

    quant_cdfvals_up = quant_cdf_bar + quant_cdfval_errs
    quant_cdfvals_up = tf.clip_by_value(quant_cdfvals_up, 0., 1.)

    quant_cdfvals_down = quant_cdf_bar - quant_cdfval_errs
    quant_cdfvals_down = tf.clip_by_value(quant_cdfvals_down, 0., 1.)

    quants_up = pchip_interpolate(hist_cdfvals, xedges[axis], quant_cdfvals_up, axis=axis)
    quants_up = tf.where(quant_cdfvals_up == 0., x_low, quants_up)
    quants_up = tf.where(quant_cdfvals_up == 1., x_high, quants_up)

    quants_down = pchip_interpolate(hist_cdfvals, xedges[axis], quant_cdfvals_down, axis=axis)
    quants_down = tf.where(quant_cdfvals_down == 0., x_low, quants_down)
    quants_down = tf.where(quant_cdfvals_down == 1., x_high, quants_down)

    quant_errs = 0.5*(quants_up - quants_down)

    zero_const = tf.constant(0., dtype)

    quant_errs = tf.where(quant_cdfvals == 0., zero_const, quant_errs)
    quant_errs = tf.where(quant_cdfvals == 1., zero_const, quant_errs)

    return quants.numpy(), quant_errs.numpy()

def func_cdf_for_quantile_fit(xvals, xedges, qparms, quant_cdfvals, axis=-1, transform = None):
    x_flat = tf.reshape(xedges[axis], (-1,))
    x_low = x_flat[0]
    x_high = x_flat[-1]

    quants = qparms_to_quantiles(qparms, x_low = x_low, x_high = x_high, axis = axis)

    spline_edges = xedges[axis]

    ndim = len(xvals)

    if transform is not None:
        transform_cdf, transform_quantile = transform

        slicelim = [slice(None)]*ndim
        slicelim[axis] = slice(1, -1)
        slicelim = tuple(slicelim)

        quants = quants[slicelim]
        quant_cdfvals = quant_cdfvals[slicelim]

        quant_cdfvals = transform_quantile(quant_cdfvals)

    cdfvals = pchip_interpolate(quants, quant_cdfvals, spline_edges, axis=axis)

    if transform is not None:
        cdfvals = transform_cdf(cdfvals)

    slicefirst = [slice(None)]*ndim
    slicefirst[axis] = slice(None, 1)
    slicefirst = tuple(slicefirst)

    slicelast = [slice(None)]*ndim
    slicelast[axis] = slice(-1, None)
    slicelast = tuple(slicelast)

    cdfvals = (cdfvals - cdfvals[slicefirst])/(cdfvals[slicelast] - cdfvals[slicefirst])

    return cdfvals

def func_constraint_for_quantile_fit(xvals, xedges, qparms, axis=-1):
    constraints = 0.5*tf.math.square(tf.math.reduce_sum(tf.exp(qparms), axis=axis) - 1.)
    constraint = tf.math.reduce_sum(constraints)
    return constraint

@tf.function
def val_grad(func, *args, **kwargs):
    xdep = kwargs["parms"]
    with tf.GradientTape() as t1:
        t1.watch(xdep)
        val = func(*args, **kwargs)
    grad = t1.gradient(val, xdep)
    return val, grad

#TODO forward-over-reverse also here?
@tf.function
def val_grad_hess(func, *args, **kwargs):
    xdep = kwargs["parms"]
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
    xdep = kwargs["parms"]
    with tf.autodiff.ForwardAccumulator(xdep, p) as acc:
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(xdep)
            val = func(*args, **kwargs)
        grad = grad_tape.gradient(val, xdep)
    hessp = acc.jvp(grad)
  
    return val, grad, hessp

def loss_with_constraint(func_loss, parms, func_constraint = None, args_loss = (), extra_args_loss=(), args_constraint = (), extra_args_constraint = ()):
    loss = func_loss(parms, *args_loss, *extra_args_loss)
    if func_constraint is not None:
        loss += func_constraint(*args_constraint, parms, *extra_args_constraint)

    return loss

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

    norm_axis = 0
    if norm_axes is not None:
        if len(norm_axes) > 1:
            raise ValueError("Only 1 nomralization access supported for bin-integrated nll")
        norm_axis = norm_axes[0]

    cdfvals = func(xvals, xedges, parms, *args)

    slices_low = [slice(None)]*len(cdfvals.shape)
    slices_low[norm_axis] = slice(None,-1)

    slices_high = [slice(None)]*len(cdfvals.shape)
    slices_high[norm_axis] = slice(1,None)

    # bin_integrals = cdfvals[1:] - cdfvals[:-1]
    bin_integrals = cdfvals[tuple(slices_high)] - cdfvals[tuple(slices_low)]
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


def fit_hist(hist, func, initial_parmvals, max_iter = 5, edmtol = 1e-5, mode = "chisq", norm_axes = None, func_constraint = None,  args = (), args_constraint=()):

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

    val_grad_args = { "func_loss" : floss,
                    "func_constraint" : func_constraint,
                    "args_loss" : (xvals, xwidths, xedges, yvals, yvariances, func, norm_axes),
                    "extra_args_loss" : args,
                    "args_constraint" : (xvals, xedges),
                    "extra_args_constraint" : args_constraint}

    def scipy_loss(parmvals, *args):
        parms = tf.constant(parmvals, dtype=dtype)

        # loss, grad = val_grad(floss, parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes, *args)
        loss, grad = val_grad(loss_with_constraint, parms=parms, **val_grad_args)
        return loss.numpy(), grad.numpy()

    def scipy_hessp(parmvals, p, *args):
        parms = tf.constant(parmvals, dtype=dtype)

        # loss, grad, hessp = val_grad_hessp(floss, p, parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes, *args)
        loss, grad, hessp = val_grad_hessp(loss_with_constraint, p, parms=parms, **val_grad_args)
        return hessp.numpy()

    current_parmvals = initial_parmvals
    for iiter in range(max_iter):

        res = scipy.optimize.minimize(scipy_loss, current_parmvals, method = "trust-krylov", jac = True, hessp = scipy_hessp, args = args)

        current_parmvals = res.x

        parms = tf.constant(current_parmvals, dtype=dtype)

        # loss, grad, hess = val_grad_hess(floss, parms, xvals, xwidths, xedges, yvals, yvariances, func, norm_axes, *args)
        loss, grad, hess = val_grad_hess(loss_with_constraint, parms=parms, **val_grad_args)
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
           "hess" : hess,
           "cov" : cov,
           "status" : status,
           "covstatus" : covstatus,
           "hess_eigvals" : eigvals,
           "edmval" : edmval,
           "loss_val" : loss }

    return res

