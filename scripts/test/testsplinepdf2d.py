import narf
import narf.fitutils

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
import hist

np.random.seed(1234)

nevt = 100000

runiform = np.random.random((nevt,))
rgaus = np.random.normal(size=(nevt,))

data = np.stack([runiform, rgaus], axis=-1)

# "pt"-dependent mean and sigma
data[:,1] = -0.1 + 0.1*data[:,0] + (1. + 0.2*data[:,0])*data[:,1]


# print(rgaus.dtype)
# print(rgaus)

axis0 = hist.axis.Regular(50, 0., 1., name="pt")
axis1 = hist.axis.Regular(100, -5., 5., name="recoil")

htest = hist.Hist(axis0, axis1)

print("data.shape", data.shape)
htest.fill(data[:,0], data[:, 1])




quant_cdfvals = tf.constant([0.0, 1e-3, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0-1e-3, 1.0], dtype = tf.float64)
nquants = quant_cdfvals.shape.num_elements()

print("nquants", nquants)

#cdf is in terms of axis1, so shapes need to be compatible
quant_cdfvals = quant_cdfvals[None, :]


# get quantiles from histogram, e.g. to help initialize the parameters for the fit (not actually used here)

hist_quantiles, hist_quantile_errs = narf.fitutils.hist_to_quantiles(htest, quant_cdfvals, axis=1)

print(hist_quantiles)
print(hist_quantile_errs)

hist_qparms, hist_qparm_errs = narf.fitutils.quantiles_to_qparms(hist_quantiles, hist_quantile_errs)

print(hist_qparms)
print(hist_qparm_errs)

def parms_to_qparms(xvals, parms):

    parms_2d = tf.reshape(parms, (-1, 2))
    parms_const = parms_2d[:,0]
    parms_slope = parms_2d[:,1]

    #cdf is in terms of axis1, so shapes need to be compatible
    parms_const = parms_const[None, :]
    parms_slope = parms_slope[None, :]

    qparms = parms_const + parms_slope*xvals[0]

    return qparms



def func_cdf(xvals, xedges, parms):
    qparms = parms_to_qparms(xvals, parms)
    return narf.fitutils.func_cdf_for_quantile_fit(xvals, xedges, qparms, quant_cdfvals, axis=1)

def func_constraint(xvals, xedges, parms):
    qparms = parms_to_qparms(xvals, parms)
    return narf.fitutils.func_constraint_for_quantile_fit(xvals, xedges, qparms)

#this is just for plotting
def func_pdf(h, parms):
    dtype = tf.float64
    xvals = [tf.constant(center, dtype=dtype) for center in h.axes.centers]
    xedges = [tf.constant(edge, dtype=dtype) for edge in h.axes.edges]

    tfparms = tf.constant(parms)

    cdf = func_cdf(xvals, xedges, tfparms)

    pdf = cdf[:,1:] - cdf[:,:-1]
    pdf = tf.maximum(pdf, tf.zeros_like(pdf))

    return pdf

nparms = nquants-1


# print("edges", htest.edges)

# assert(0)

initial_parms_const = np.array([np.log(1./nparms)]*nparms)
initial_parms_slope = np.zeros_like(initial_parms_const)

initial_parms = np.stack([initial_parms_const, initial_parms_slope], axis=-1)
initial_parms = np.reshape(initial_parms, (-1,))

res = narf.fitutils.fit_hist(htest, func_cdf, initial_parms, mode="nll_bin_integrated", norm_axes=[1], func_constraint=func_constraint)

print(res)


parmvals = res["x"]


pdfvals = func_pdf(htest, parmvals)
pdfvals *= htest.sum()/np.sum(pdfvals)


# hplot = htest[5]

plot = plt.figure()
htest[5,:].plot()
plt.plot(htest.axes[1].centers, pdfvals[5])
# plt.show()
plot.savefig("test.png")



