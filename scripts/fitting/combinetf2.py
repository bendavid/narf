import numpy as np
import hist
import h5py
import tensorflow as tf
import narf.combineutils
import argparse
import narf.ioutils

parser =  argparse.ArgumentParser()

parser.add_argument("filename", help="filename of the main hdf5 input")
parser.add_argument("-o","--output", default="fitresults.hdf5",  help="output file name")
parser.add_argument("-t","--toys", default=-1, type=int, help="run a given number of toys, 0 fits the data, and -1 fits the asimov toy (the default)")
parser.add_argument("--expectSignal", default=1., type=float, help="rate multiplier for signal expectation (used for fit starting values and for toys)")
parser.add_argument("--POIMode", default="mu", help="mode for POI's")
parser.add_argument("--allowNegativePOI", default=False, action='store_true', help="allow signal strengths to be negative (otherwise constrained to be non-negative)")
parser.add_argument("--POIDefault", default=1., type=float, help="mode for POI's")
parser.add_argument("--saveHists", default=False, action='store_true', help="save prefit and postfit histograms")
parser.add_argument("--computeHistErrors", default=False, action='store_true', help="propagate uncertainties to prefit and postfit histograms")
parser.add_argument("--binByBinStat", default=False, action='store_true', help="add bin-by-bin statistical uncertainties on templates (using Barlow and Beeston 'lite' method")
parser.add_argument("--externalPostfit", default=None, help="load posfit nuisance parameters and covariance from result of an external fit")

args = parser.parse_args()

indata = narf.combineutils.FitInputData(args.filename)
fitter = narf.combineutils.Fitter(indata, args)

if args.toys == -1:
    fitter.nobs.assign(fitter._compute_yields_inclusive())


results = {}

if args.saveHists:
    nbins = fitter.indata.nbins
    nbinsfull = fitter.indata.nbinsfull

    axis_obs = hist.axis.Integer(0, nbins, underflow=False, overflow=False, name="obs")
    axis_obsfull = hist.axis.Integer(0, nbinsfull, underflow=False, overflow=False, name="obsfull")
    axis_procs = hist.axis.StrCategory(fitter.indata.procs, name="processes")

    hist_data_obs = hist.Hist(axis_obs, storage=hist.storage.Weight(), name = "data_obs", label="observed number of events in data")
    hist_data_obs.values()[...] = memoryview(fitter.indata.data_obs)
    hist_data_obs.variances()[...] = hist_data_obs.values()
    results["hist_data_obs"] = narf.ioutils.H5PickleProxy(hist_data_obs)

    hist_nobs = hist.Hist(axis_obs, storage=hist.storage.Weight(), name = "nobs", label = "observed number of events for fit")
    hist_nobs.values()[...] = memoryview(fitter.nobs.value())
    hist_nobs.variances()[...] = hist_nobs.values()
    results["hist_nobs"] = narf.ioutils.H5PickleProxy(hist_nobs)

    invhessianprefit = fitter.prefit_covariance()

    # for a diagonal matrix cholesky decomposition equivalent is equal to the element-wise sqrt
    invhessianprefitchol = tf.sqrt(invhessianprefit)

    exp_pre_per_process = fitter.expected_events_per_process()

    hist_prefit = hist.Hist(axis_obsfull, axis_procs, storage=hist.storage.Weight(), name = "prefit", label = "prefit expected number of events")
    hist_prefit.values()[...] = memoryview(exp_pre_per_process)
    hist_prefit.variances()[...] = 0.
    results["hist_prefit"] = narf.ioutils.H5PickleProxy(hist_prefit)

    if args.computeHistErrors:
        exp_pre_inclusive, exp_pre_inclusive_var = fitter.expected_events_inclusive_with_variance(invhessianprefitchol)

        hist_prefit_inclusive = hist.Hist(axis_obsfull, storage=hist.storage.Weight(), name = "prefit_inclusive", label = "prefit expected number of events for all processes combined")
        hist_prefit_inclusive.values()[...] = memoryview(exp_pre_inclusive)
        hist_prefit_inclusive.variances()[...] = memoryview(exp_pre_inclusive_var)
        results["hist_prefit_inclusive"] = narf.ioutils.H5PickleProxy(hist_prefit_inclusive)




if args.toys >= 0:
    fitter.minimize()

val, grad, hess = fitter.loss_val_grad_hess()

cov = tf.linalg.inv(hess)



if args.externalPostfit is not None:
    # load results from external fit and set postfit value and covariance elements for common parameters
    with h5py.File(args.externalPostfit, "r") as fext:
        parms_ext = fext["parms"][...]

        x_ext = fext["x"][...]
        cov_ext = fext["cov"][...]

    # FIXME do this without explicit loops and ideally in tensorflow directly

    xvals = fitter.x.numpy()
    covval = invhessianprefit.numpy()

    parmmap = {}

    for iparm, parm in enumerate(fitter.parms):
        parmmap[parm] = iparm

    for iparm_ext, parmi_ext in enumerate(parms_ext):
        iparm = parmmap.get(parmi_ext)

        if iparm is None:
            continue

        xvals[iparm] = x_ext[iparm_ext]

        for jparm_ext, parmj_ext in enumerate(parms_ext):
            jparm = parmmap.get(parmj_ext)

            if jparm is None:
                continue

            covval[iparm, jparm] = cov_ext[iparm_ext, jparm_ext]

    fitter.x.assign(xvals)
    cov = tf.convert_to_tensor(covval)





if args.saveHists:

    covchol_ext = tf.linalg.cholesky(cov)

    exp_post_per_process = fitter.expected_events_per_process()

    hist_postfit = hist.Hist(axis_obsfull, axis_procs, storage=hist.storage.Weight(), name = "postfit", label = "postfit expected number of events")
    hist_postfit.values()[...] = memoryview(exp_post_per_process)
    hist_postfit.variances()[...] = 0.
    results["hist_postfit"] = narf.ioutils.H5PickleProxy(hist_postfit)

    if args.computeHistErrors:

        exp_post_inclusive, exp_post_inclusive_var = fitter.expected_events_inclusive_with_variance(covchol_ext)

        hist_postfit_inclusive = hist.Hist(axis_obsfull, storage=hist.storage.Weight(), name = "postfit_inclusive", label = "postfit expected number of events for all processes combined")
        hist_postfit_inclusive.values()[...] = memoryview(exp_post_inclusive)
        hist_postfit_inclusive.variances()[...] = memoryview(exp_post_inclusive_var)
        results["hist_postfit_inclusive"] = narf.ioutils.H5PickleProxy(hist_postfit_inclusive)

# pass meta data into output file
meta = {
    "signals": fitter.indata.signals,
    "procs": fitter.indata.procs,
    **fitter.indata.metadata
}
results["meta"] = narf.ioutils.H5PickleProxy(meta)
 
with h5py.File(args.output, "w") as fout:
    narf.ioutils.pickle_dump_h5py("results", results, fout)
