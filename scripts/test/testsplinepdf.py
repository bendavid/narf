import narf
import narf.fitutils

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
import hist
import math

np.random.seed(1234)

nevt = 100000

rgaus = np.random.normal(size=(nevt,))

print(rgaus.dtype)
print(rgaus)

axis0 = hist.axis.Regular(100, -5., 5.)

htest = hist.Hist(axis0)
htest.fill(rgaus)

print(htest)


quant_cdfvals = tf.constant([0.0, 1e-3, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0-1e-3, 1.0], tf.float64)

nquants = quant_cdfvals.shape.num_elements()

def func_transform_cdf(quantile):
    const_sqrt2 = tf.constant(math.sqrt(2.), quantile.dtype)
    return 0.5*(1. + tf.math.erf(quantile/const_sqrt2))

def func_transform_quantile(cdf):
    const_sqrt2 = tf.constant(math.sqrt(2.), cdf.dtype)
    return const_sqrt2*tf.math.erfinv(2*cdf - 1.)



def func_cdf(xvals, xedges, parms, quant_cdfvals):
    qparms = parms

    cdf = narf.fitutils.func_cdf_for_quantile_fit(xvals, xedges, qparms, quant_cdfvals, transform = (func_transform_cdf, func_transform_quantile))

    return cdf


#this is just for plotting
def func_pdf(h, parms):
    dtype = tf.float64
    xvals = [tf.constant(center, dtype=dtype) for center in h.axes.centers]
    xedges = [tf.constant(edge, dtype=dtype) for edge in h.axes.edges]

    tfparms = tf.constant(parms)

    cdf = func_cdf(xvals, xedges, tfparms, quant_cdfvals)

    pdf = cdf[1:] - cdf[:-1]
    pdf = tf.maximum(pdf, tf.zeros_like(pdf))

    return pdf

nparms = nquants-1


initial_parms = np.array([np.log(1./nparms)]*nparms)

res = narf.fitutils.fit_hist(htest, func_cdf, initial_parms, mode="nll_bin_integrated", func_constraint=narf.fitutils.func_constraint_for_quantile_fit, args = (quant_cdfvals,))

print(res)


parmvals = res["x"]


pdfvals = func_pdf(htest, parmvals)
pdfvals *= htest.sum()/np.sum(pdfvals)

#
plot = plt.figure()
plt.yscale("log")
htest.plot()
plt.plot(htest.axes[0].centers, pdfvals)
# plt.show()
plot.savefig("test.png")



