import numpy as np
import hist
import h5py
import tensorflow as tf
import narf.combineutils
import argparse
import narf.ioutils
import os

import pdb


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
parser.add_argument("--computeVariations", default=False, action='store_true', help="save postfit histograms with each noi varied up to down")
parser.add_argument("--binByBinStat", default=False, action='store_true', help="add bin-by-bin statistical uncertainties on templates (adding sumW2 on variance)")
parser.add_argument("--externalPostfit", default=None, help="load posfit nuisance parameters and covariance from result of an external fit")
parser.add_argument("--pseudoData", default=None, type=str, help="run fit on pseudo data with the given name")
parser.add_argument("--normalize", default=False, action='store_true', help="Normalize prediction and systematic uncertainties to the overall event yield in data")

args = parser.parse_args()

indata = narf.combineutils.FitInputData(args.filename, args.pseudoData, normalize=args.normalize)
fitter = narf.combineutils.Fitter(indata, args)

if args.toys == -1:
    fitter.nobs.assign(fitter.expected_events_noBBB())

if args.saveHists:

    invhessianprefit = fitter.prefit_covariance()

    # for a diagonal matrix cholesky decomposition equivalent is equal to the element-wise sqrt
    invhessianprefitchol = tf.sqrt(invhessianprefit)

    if args.computeHistErrors:
        exp_pre_inclusive, exp_pre_inclusive_var = fitter.expected_events_inclusive_with_variance_noBBB(invhessianprefitchol)
        exp_pre_per_process, exp_pre_per_process_var = fitter.expected_events_per_process_with_variance_noBBB(invhessianprefitchol)
    else:
        exp_pre_inclusive = fitter.expected_events()
        exp_pre_per_process = fitter.expected_events_per_process()

    if args.computeVariations:
        exp_pre_inclusive_var_up = {}
        exp_pre_inclusive_var_down = {}
        for var_idx, var_key in enumerate(indata.systs): 
            # get up/down variation for each variation

            var_name = fitter.parms[var_idx].decode('utf-8')

            fitter.x[var_idx].assign(1)
            exp_pre_inclusive_var_up[var_name] = fitter.expected_events()

            fitter.x[var_idx].assign(-1)
            exp_pre_inclusive_var_down[var_name] = fitter.expected_events()

            # set back to central value
            fitter.x[var_idx].assign(0)

chi2_prefit = fitter.chi2(fitter.prefit_covariance())

if args.toys >= 0 and args.externalPostfit is None:
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

chi2_postfit = fitter.chi2(cov)

results = {
    "ndf_prefit": fitter.indata.nbins - fitter.indata.normalize,
    "ndf_postfit": fitter.indata.nbins - fitter.indata.nsystnoconstraint - fitter.indata.nsignals - fitter.indata.normalize,
    "chi2_prefit": chi2_prefit,
    "chi2_postfit": chi2_postfit
}

if args.saveHists:

    covchol_ext = tf.linalg.cholesky(cov)

    if args.computeHistErrors:
        exp_post_inclusive, exp_post_inclusive_var = fitter.expected_events_inclusive_with_variance_noBBB(covchol_ext)
        exp_post_per_process, exp_post_per_process_var = fitter.expected_events_per_process_with_variance_noBBB(covchol_ext)
    else:
        exp_post_inclusive = fitter.expected_events()
        exp_post_per_process = fitter.expected_events_per_process()
    
    if args.computeVariations:
        exp_post_inclusive_var_up = {}
        exp_post_inclusive_var_down = {}
        for var_idx, var_key in enumerate(indata.systs): 
            # get up/down variation for each variation
            central = fitter.x[var_idx]
            variation = cov[var_idx,var_idx]**0.5
            var_name = fitter.parms[var_idx].decode('utf-8')

            fitter.x[var_idx].assign(central + variation)
            exp_post_inclusive_var_up[var_name] = fitter.expected_events()

            fitter.x[var_idx].assign(central - variation)
            exp_post_inclusive_var_down[var_name] = fitter.expected_events()

            # set back to central value
            fitter.x[var_idx].assign(central)

    results.update({
        "hist_data_obs":{},
        "hist_nobs":{},
        "hist_prefit":{},
        "hist_postfit":{},
        "hist_prefit_inclusive":{},
        "hist_postfit_inclusive":{},
        "hist_prefit_inclusive_variations":{},
        "hist_postfit_inclusive_variations":{},
    })

    ibin = 0
    for channel, info in fitter.indata.channel_info.items():
        axes = info["axes"]

        shape = [len(a) for a in axes]
        stop = ibin+np.product(shape)

        shape_proc = [*shape, fitter.indata.nproc]

        if "masked" not in channel:
            hist_data_obs = hist.Hist(*axes, storage=hist.storage.Weight(), name = "data_obs", label="observed number of events in data")
            hist_data_obs.values()[...] = memoryview(tf.reshape(fitter.indata.data_obs[ibin:stop], shape))
            hist_data_obs.variances()[...] = hist_data_obs.values()
            results["hist_data_obs"][channel] = narf.ioutils.H5PickleProxy(hist_data_obs)

            hist_nobs = hist.Hist(*axes, storage=hist.storage.Weight(), name = "nobs", label = "observed number of events for fit")
            hist_nobs.values()[...] = memoryview(tf.reshape(fitter.nobs.value()[ibin:stop], shape))
            hist_nobs.variances()[...] = hist_nobs.values()
            results["hist_nobs"][channel] = narf.ioutils.H5PickleProxy(hist_nobs)

        hist_prefit = hist.Hist(*axes, fitter.indata.axis_procs, storage=hist.storage.Weight(), name = "prefit", label = "prefit expected number of events")
        hist_prefit.values()[...] = memoryview(tf.reshape(exp_pre_per_process[ibin:stop,:], shape_proc))
        if args.computeHistErrors:
            hist_prefit.variances()[...] = memoryview(tf.reshape(exp_pre_per_process_var[ibin:stop], shape_proc))
        else:
            hist_prefit.variances()[...] = 0.
        results["hist_prefit"][channel] = narf.ioutils.H5PickleProxy(hist_prefit)

        hist_postfit = hist.Hist(*axes, fitter.indata.axis_procs, storage=hist.storage.Weight(), name = "postfit", label = "postfit expected number of events")
        hist_postfit.values()[...] = memoryview(tf.reshape(exp_post_per_process[ibin:stop,:], shape_proc))
        if args.computeHistErrors:
            hist_postfit.variances()[...] = memoryview(tf.reshape(exp_post_per_process_var[ibin:stop], shape_proc))
        else:
            hist_postfit.variances()[...] = 0.
        results["hist_postfit"][channel] = narf.ioutils.H5PickleProxy(hist_postfit)

        if args.computeHistErrors:
            hist_prefit_inclusive = hist.Hist(*axes, storage=hist.storage.Weight(), name = "prefit_inclusive", label = "prefit expected number of events for all processes combined")
            hist_prefit_inclusive.values()[...] = memoryview(tf.reshape(exp_pre_inclusive[ibin:stop], shape))
            hist_prefit_inclusive.variances()[...] = memoryview(tf.reshape(exp_pre_inclusive_var[ibin:stop], shape))
            results["hist_prefit_inclusive"][channel] = narf.ioutils.H5PickleProxy(hist_prefit_inclusive)

            hist_postfit_inclusive = hist.Hist(*axes, storage=hist.storage.Weight(), name = "postfit_inclusive", label = "postfit expected number of events for all processes combined")
            hist_postfit_inclusive.values()[...] = memoryview(tf.reshape(exp_post_inclusive[ibin:stop], shape))
            hist_postfit_inclusive.variances()[...] = memoryview(tf.reshape(exp_post_inclusive_var[ibin:stop], shape))
            results["hist_postfit_inclusive"][channel] = narf.ioutils.H5PickleProxy(hist_postfit_inclusive)

        if args.computeVariations:
            for fittype, up, down in [
                ("prefit", exp_pre_inclusive_var_up, exp_pre_inclusive_var_down),
                ("postfit", exp_post_inclusive_var_up, exp_post_inclusive_var_down)            
            ]:
                # save up/down variations
                hVar = hist.Hist(
                    *axes, 
                    hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar"), 
                    hist.axis.StrCategory(up.keys(), name="vars"),
                    storage = hist.storage.Double(), 
                    name = f"{fittype}_inclusive_variations", 
                    label = f"{fittype} expected number with variations of events for all processes combined",
                    )

                for var in up.keys():
                    var_idx = hVar.axes["vars"].index(var)
                    hVar.values()[...,0,var_idx] = memoryview(tf.reshape(down[var][ibin:stop], shape))
                    hVar.values()[...,1,var_idx] = memoryview(tf.reshape(up[var][ibin:stop], shape))

                results[f"hist_{fittype}_inclusive_variations"][channel] = narf.ioutils.H5PickleProxy(hVar)

        ibin = stop


# pass meta data into output file
meta = {
    "meta_info" : narf.ioutils.make_meta_info_dict(args=args), 
    "meta_info_input": fitter.indata.metadata,
    "signals": fitter.indata.signals,
    "procs": fitter.indata.procs,
}

outfolder = os.path.dirname(args.output)
if not os.path.exists(outfolder):
    print(f"Creating output folder {outfolder}")
    os.makedirs(outfolder)
    
with h5py.File(args.output, "w") as fout:
    narf.ioutils.pickle_dump_h5py("results", results, fout)
    narf.ioutils.pickle_dump_h5py("meta", meta, fout)
