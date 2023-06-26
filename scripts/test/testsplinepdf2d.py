import narf
import narf.fitutils

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
import hist
import math

import onnx
import tf2onnx

np.random.seed(1234)

nevt = 20000

runiform = np.random.random((nevt,))
rgaus = np.random.normal(size=(nevt,))

data = np.stack([runiform, rgaus], axis=-1)

# "pt"-dependent mean and sigma
data[:,1] = -0.1 + 0.1*data[:,0] + (1. + 0.2*data[:,0])*data[:,1]


# print(rgaus.dtype)
# print(rgaus)

axis0 = hist.axis.Regular(50, 0., 1., name="pt")
axis1 = hist.axis.Regular(100, -5., 5., name="recoil")

htest_data = hist.Hist(axis0, axis1)
htest_mc = hist.Hist(axis0, axis1)

# print("data.shape", data.shape)
htest_data.fill(data[:nevt//2,0], data[:nevt//2, 1])
htest_mc.fill(data[nevt//2:,0], data[nevt//2:, 1])




quant_cdfvals = tf.constant([0.0, 1e-3, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0-1e-3, 1.0], dtype = tf.float64)
nquants = quant_cdfvals.shape.num_elements()

print("nquants", nquants)

#cdf is in terms of axis1, so shapes need to be compatible
quant_cdfvals = quant_cdfvals[None, :]


# get quantiles from histogram, e.g. to help initialize the parameters for the fit (not actually used here)

# hist_quantiles, hist_quantile_errs = narf.fitutils.hist_to_quantiles(htest, quant_cdfvals, axis=1)
#
# print(hist_quantiles)
# print(hist_quantile_errs)
#
# hist_qparms, hist_qparm_errs = narf.fitutils.quantiles_to_qparms(hist_quantiles, hist_quantile_errs)
#
# print(hist_qparms)
# print(hist_qparm_errs)

def parms_to_qparms(xvals, parms):

    parms_2d = tf.reshape(parms, (-1, 2))
    parms_const = parms_2d[:,0]
    parms_slope = parms_2d[:,1]

    #cdf is in terms of axis1, so shapes need to be compatible
    parms_const = parms_const[None, :]
    parms_slope = parms_slope[None, :]

    qparms = parms_const + parms_slope*xvals[0]

    return qparms


def func_transform_cdf(quantile):
    const_sqrt2 = tf.constant(math.sqrt(2.), quantile.dtype)
    return 0.5*(1. + tf.math.erf(quantile/const_sqrt2))

def func_transform_quantile(cdf):
    const_sqrt2 = tf.constant(math.sqrt(2.), cdf.dtype)
    return const_sqrt2*tf.math.erfinv(2.*cdf - 1.)

# def func_transform_cdf(quantile):
#     return tf.math.log(quantile/(1.-quantile))
#
# def func_transform_quantile(cdf):
#     return tf.math.sigmoid(cdf)

def func_cdf(xvals, xedges, parms):
    qparms = parms_to_qparms(xvals, parms)
    # return narf.fitutils.func_cdf_for_quantile_fit(xvals, xedges, qparms, quant_cdfvals, axis=1)

    return narf.fitutils.func_cdf_for_quantile_fit(xvals, xedges, qparms, quant_cdfvals, axis=1, transform = (func_transform_cdf, func_transform_quantile))

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

res_data = narf.fitutils.fit_hist(htest_data, func_cdf, initial_parms, mode="nll_bin_integrated", norm_axes=[1])

res_mc = narf.fitutils.fit_hist(htest_mc, func_cdf, initial_parms, mode="nll_bin_integrated", norm_axes=[1])

print(res_data)


parmvals_data = tf.constant(res_data["x"], tf.float64)
parmvals_mc = tf.constant(res_mc["x"], tf.float64)

hess_data = res_data["hess"]
hess_mc = res_mc["hess"]

def get_scaled_eigenvectors(hess, num_null = 2):
    e,v = np.linalg.eigh(hess)

    # remove the null eigenvectors
    e = e[None, num_null:]
    v = v[:, num_null:]

    # scale the eigenvectors
    vscaled = v/np.sqrt(e)

    return vscaled

vscaled_data = tf.constant(get_scaled_eigenvectors(hess_data), tf.float64)
vscaled_mc = tf.constant(get_scaled_eigenvectors(hess_data), tf.float64)

print("vscaled_data.shape", vscaled_data.shape)

ut_flat = np.reshape(htest_data.axes.edges[1], (-1,))
ut_low = tf.constant(ut_flat[0], tf.float64)
ut_high = tf.constant(ut_flat[-1], tf.float64)

def func_cdf_mc(pt, ut):
    pts = tf.reshape(pt, (1,1))
    uts = tf.reshape(ut, (1,1))

    xvals = [pts, None]
    xedges = [None, uts]

    parms = parmvals_mc

    qparms = parms_to_qparms(xvals, parms)

    ut_axis = 1

    quants = narf.fitutils.qparms_to_quantiles(qparms, x_low = ut_low, x_high = ut_high, axis = ut_axis)
    spline_edges = xedges[ut_axis]

    cdfvals = narf.fitutils.pchip_interpolate(quants, quant_cdfvals, spline_edges, axis=ut_axis)

    return cdfvals

def func_cdfinv_data(pt, quant):
    pts = tf.reshape(pt, (1,1))
    quant_outs = tf.reshape(quant, (1,1))

    xvals = [pts, None]
    xedges = [None, quant_outs]

    parms = parmvals_data

    qparms = parms_to_qparms(xvals, parms)

    ut_axis = 1

    quants = narf.fitutils.qparms_to_quantiles(qparms, x_low = ut_low, x_high = ut_high, axis = ut_axis)
    spline_edges = xedges[ut_axis]

    cdfinvvals = narf.fitutils.pchip_interpolate(quant_cdfvals, quants, spline_edges, axis=ut_axis)

    return cdfinvvals

def func_cdfinv_pdf_data(pt, quant):
    with tf.GradientTape() as t:
        t.watch(quant)
        cdfinv = func_cdfinv_data(pt, quant)
    pdfreciprocal = t.gradient(cdfinv, quant)
    pdf = 1./pdfreciprocal
    return cdfinv, pdf

scalar_spec = tf.TensorSpec([], tf.float64)


def transform_mc(pt, ut):
    with tf.GradientTape(persistent=True) as t:
        t.watch(parmvals_mc)
        t.watch(parmvals_data)

        cdf_mc = func_cdf_mc(pt, ut)
        ut_transformed, pdf = func_cdfinv_pdf_data(pt, cdf_mc)

        ut_transformed = tf.reshape(ut_transformed, [])
        pdf = tf.reshape(pdf, [])

    pdf_grad_mc = t.gradient(pdf, parmvals_mc)
    pdf_grad_data = t.gradient(pdf, parmvals_data)

    del t

    weight_grad_mc = pdf_grad_mc/pdf
    weight_grad_data = pdf_grad_data/pdf

    weight_grad_mc = weight_grad_mc[None, :]
    weight_grad_data = weight_grad_data[None, :]

    weight_grad_mc_eig = weight_grad_mc @ vscaled_mc
    weight_grad_data_eig = weight_grad_data @ vscaled_data

    weight_grad_mc_eig = tf.reshape(weight_grad_mc_eig, [-1])
    weight_grad_data_eig = tf.reshape(weight_grad_data_eig, [-1])

    weight_grad_eig = tf.concat([weight_grad_mc_eig, weight_grad_data_eig], axis=0)

    return ut_transformed, weight_grad_eig
    # return ut_transformed

@tf.function
def transform_mc_simple(pt, ut):
    cdf_mc = func_cdf_mc(pt, ut)
    ut_transformed, pdf = func_cdfinv_pdf_data(pt, cdf_mc)

    ut_transformed = tf.reshape(ut_transformed, [])

    return ut_transformed



pt_test = tf.constant(0.2, tf.float64)
ut_test = tf.constant(1.0, tf.float64)

ut, grad = transform_mc(pt_test, ut_test)
# ut = transform_mc(pt_test, ut_test)

print("shapes", ut.shape, grad.shape)

print("ut", ut)
print("grad", grad)

input_signature = [scalar_spec, scalar_spec]

class TestMod(tf.Module):

    @tf.function(input_signature =  [scalar_spec, scalar_spec])
    def __call__(self, pt, ut):
        return transform_mc(pt, ut)

module = TestMod()
# tf.saved_model.save(module, "test")

concrete_function = module.__call__.get_concrete_function()

# Convert the model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function], module)

# converter = tf.lite.TFLiteConverter.from_saved_model("test") # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()

# print(tflite_model)

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)


# onnx_model, _ = tf2onnx.convert.from_function(transform_mc, input_signature)
# onnx.save(onnx_model, "test.onnx")


parmvals = res_data["x"]


pdfvals = func_pdf(htest_data, parmvals)
pdfvals *= htest_data.sum()/np.sum(pdfvals)


# hplot = htest[5]

plot = plt.figure()
plt.yscale("log")
htest_data[5,:].plot()
plt.plot(htest_data.axes[1].centers, pdfvals[5])
# plt.show()
plot.savefig("test.png")



