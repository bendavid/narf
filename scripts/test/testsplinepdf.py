import narf
import narf.fitutils

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
import hist

np.random.seed(1234)

nevt = 100000

rgaus = np.random.normal(size=(nevt,))

print(rgaus.dtype)
print(rgaus)

axis0 = hist.axis.Regular(100, -5., 5.)

htest = hist.Hist(axis0)
htest.fill(rgaus)

print(htest)


xmin = -5.
xmax = 5.

quantvals = [0.0, 1e-3, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0-1e-3, 1.0]
nquants = len(quantvals)


def func_cdf(xvals, xedges, parms):

    quants = tf.constant(quantvals, dtype=tf.float64)

    x0 = tf.constant(xmin, dtype=tf.float64)
    deltax = tf.concat([[x0], tf.exp(parms)], axis=0)

    xquants = tf.cumsum(deltax)


    print("xquants", xquants)
    print("quants", quants)
    print("xedges[1]", xedges[0])

    cdfvals = narf.fitutils.pchip_interpolate(xquants, quants, xedges[0])

    print("cdfvals", cdfvals)


    return cdfvals



#this is just for plotting
def func_pdf(h, parms):
    dtype = tf.float64
    xvals = [tf.constant(center, dtype=dtype) for center in h.axes.centers]
    xedges = [tf.constant(edge, dtype=dtype) for edge in h.axes.edges]

    tfparms = tf.constant(parms)

    cdf = func_cdf(xvals, xedges, tfparms)

    pdf = cdf[1:] - cdf[:-1]
    pdf = tf.maximum(pdf, tf.zeros_like(pdf))

    return pdf

nparms = nquants-1


initial_parms = np.array([np.log((xmax-xmin)/nparms)]*nparms)

res = narf.fitutils.fit_hist(htest, func_cdf, initial_parms, mode="nll_bin_integrated")

print(res)


parmvals = res["x"]


pdfvals = func_pdf(htest, parmvals)
pdfvals *= htest.sum()/np.sum(pdfvals)

#
# plot = plt.figure()
# htest.plot()
# plt.plot(htest.axes[0].centers, pdfvals)
# # plt.show()
# plot.savefig("test.png")



