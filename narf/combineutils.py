import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import scipy
import h5py
import hist
import narf.ioutils

class SimpleSparseTensor:
    def __init__(self, indices, values, dense_shape):
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape

def maketensor(h5dset):
    if 'original_shape' in h5dset.attrs:
        shape = h5dset.attrs['original_shape']
    else:
        shape = h5dset.shape

    if h5dset.size == 0:
        return tf.zeros(shape,h5dset.dtype)

    filename = h5dset.file.filename
    dsetpath = h5dset.name

    atensor = tfio.IOTensor.from_hdf5(filename)(dsetpath).to_tensor()

    atensor = tf.reshape(atensor, shape)
    return atensor

def makesparsetensor(h5group):
    indices = maketensor(h5group['indices'])
    values = maketensor(h5group['values'])
    dense_shape = h5group.attrs['dense_shape']

    return SimpleSparseTensor(indices,values,dense_shape)

class FitInputData:
    def __init__(self, filename, pseudodata=None, normalize=False):
        with h5py.File(filename, mode='r') as f:

            #load text arrays from file
            self.procs = f['hprocs'][...]
            self.signals = f['hsignals'][...]
            self.systs = f['hsysts'][...]
            self.systsnoprofile = f['hsystsnoprofile'][...]
            self.systsnoconstraint = f['hsystsnoconstraint'][...]
            self.systgroups = f['hsystgroups'][...]
            self.systgroupidxs = f['hsystgroupidxs'][...]
            self.chargegroups = f['hchargegroups'][...]
            self.chargegroupidxs = f['hchargegroupidxs'][...]
            self.polgroups = f['hpolgroups'][...]
            self.polgroupidxs = f['hpolgroupidxs'][...]
            self.helgroups = f['hhelgroups'][...]
            self.helgroupidxs = f['hhelgroupidxs'][...]
            self.sumgroups = f['hsumgroups'][...]
            self.sumgroupsegmentids = f['hsumgroupsegmentids'][...]
            self.sumgroupidxs = f['hsumgroupidxs'][...]
            self.chargemetagroups = f['hchargemetagroups'][...]
            self.chargemetagroupidxs = f['hchargemetagroupidxs'][...]
            self.ratiometagroups = f['hratiometagroups'][...]
            self.ratiometagroupidxs = f['hratiometagroupidxs'][...]
            self.helmetagroups = f['hhelmetagroups'][...]
            self.helmetagroupidxs = f['hhelmetagroupidxs'][...]
            self.reggroups = f['hreggroups'][...]
            self.reggroupidxs = f['hreggroupidxs'][...]
            self.poly1dreggroups = f['hpoly1dreggroups'][...]
            self.poly1dreggroupfirstorder = f['hpoly1dreggroupfirstorder'][...]
            self.poly1dreggrouplastorder = f['hpoly1dreggrouplastorder'][...]
            self.poly1dreggroupnames = f['hpoly1dreggroupnames'][...]
            self.poly1dreggroupbincenters = f['hpoly1dreggroupbincenters'][...]
            self.poly2dreggroups = f['hpoly2dreggroups'][...]
            self.poly2dreggroupfirstorder = f['hpoly2dreggroupfirstorder'][...]
            self.poly2dreggrouplastorder = f['hpoly2dreggrouplastorder'][...]
            self.poly2dreggroupfullorder = f['hpoly2dreggroupfullorder'][...]
            self.poly2dreggroupnames = f['hpoly2dreggroupnames'][...]
            self.poly2dreggroupbincenters0 = f['hpoly2dreggroupbincenters0'][...]
            self.poly2dreggroupbincenters1 = f['hpoly2dreggroupbincenters1'][...]
            self.noigroups = f['hnoigroups'][...]
            self.noigroupidxs = f['hnoigroupidxs'][...]
            self.maskedchans = f['hmaskedchans'][...]
            if "hpseudodatanames" in f.keys():
                self.pseudodatanames = f['hpseudodatanames'][...].astype(str)
            else:
                self.pseudodatanames = []

            #load arrays from file
            hconstraintweights = f['hconstraintweights']
            hdata_obs = f['hdata_obs']

            if 'hdata_cov_inv' in f.keys():
                hdata_cov_inv = f['hdata_cov_inv']
                self.data_cov_inv = maketensor(hdata_cov_inv)
            else:
                self.data_cov_inv = None

            self.sparse = not 'hnorm' in f

            if self.sparse:
                hnorm_sparse = f['hnorm_sparse']
                hlogk_sparse = f['hlogk_sparse']
                self.nbinsfull = hnorm_sparse.attrs['dense_shape'][0]
            else:
                hnorm = f['hnorm']
                hlogk = f['hlogk']
                self.nbinsfull = hnorm.attrs['original_shape'][0]

            #infer some metadata from loaded information
            self.dtype = hdata_obs.dtype
            self.nbins = hdata_obs.shape[-1]
            self.nbinsmasked = self.nbinsfull - self.nbins
            self.nproc = len(self.procs)
            self.nsyst = len(self.systs)
            self.nsystnoprofile = len(self.systsnoprofile)
            self.nsystnoconstraint = len(self.systsnoconstraint)
            self.nsignals = len(self.signals)
            self.nsystgroups = len(self.systgroups)
            self.nchargegroups = len(self.chargegroups)
            self.npolgroups = len(self.polgroups)
            self.nhelgroups = len(self.helgroups)
            self.nsumgroups = len(self.sumgroups)
            self.nchargemetagroups = len(self.chargemetagroups)
            self.nratiometagroups = len(self.ratiometagroups)
            self.nhelmetagroups = len(self.helmetagroups)
            self.nreggroups = len(self.reggroups)
            self.npoly1dreggroups = len(self.poly1dreggroups)
            self.npoly2dreggroups = len(self.poly2dreggroups)
            self.nnoigroups = len(self.noigroups)

            # reference meta data if available
            self.metadata = {}
            if "meta" in f.keys():
                from narf.ioutils import pickle_load_h5py
                self.metadata = pickle_load_h5py(f["meta"])
                self.channel_info = self.metadata["channel_info"]
            else:
                self.channel_info = {
                    "ch0":{"axes": [hist.axis.Integer(0, self.nbins, underflow=False, overflow=False, name="obs")]}
                }
                if self.nbinsmasked > 0:
                    self.channel_info["ch1_masked"] = {"axes": [hist.axis.Integer(0, self.nbinsmasked, underflow=False, overflow=False, name="masked")]}

            # compute indices for channels
            ibin = 0
            for channel, info in self.channel_info.items():
                axes = info["axes"]
                shape = tuple([len(a) for a in axes])
                size = np.prod(shape)

                start = ibin
                stop = start + size

                info["start"] = start
                info["stop"] = stop

                ibin = stop

            for channel, info in self.channel_info.items():
                print(channel, info)

            self.axis_procs = hist.axis.StrCategory(self.procs, name="processes")                

            #build tensorflow graph for likelihood calculation

            #start by creating tensors which read in the hdf5 arrays (optimized for memory consumption)
            self.constraintweights = maketensor(hconstraintweights)

            #load data/pseudodata
            if pseudodata is not None:
                if pseudodata in self.pseudodatanames:
                    pseudodata_idx = np.where(self.pseudodatanames == pseudodata)[0][0]
                else:
                    raise Exception("Pseudodata %s not found, available pseudodata sets are %s" % (pseudodata, self.pseudodatanames))
                print("Run pseudodata fit for index %i: " % (pseudodata_idx))
                print(self.pseudodatanames[pseudodata_idx])
                hdata_obs = f['hpseudodata']

                data_obs = maketensor(hdata_obs)
                self.data_obs = data_obs[:, pseudodata_idx]
            else:
                self.data_obs = maketensor(hdata_obs)

            hkstat = f['hkstat']
            self.kstat = maketensor(hkstat)

            if self.sparse:
                self.norm_sparse = makesparsetensor(hnorm_sparse)
                self.logk_sparse = makesparsetensor(hlogk_sparse)
            else:
                self.norm = maketensor(hnorm)
                self.logk = maketensor(hlogk)

            self.normalize = normalize
            if self.normalize:
                # normalize predictoin and each systematic to total event yield in data
                # FIXME this should be done per-channel ideally

                data_sum = tf.reduce_sum(self.data_obs)
                norm_sum = tf.reduce_sum(self.norm)
                lognorm_sum = tf.math.log(norm_sum)[None, None, ...]

                logkavg = self.logk[..., 0, :]
                logkhalfdiff = self.logk[..., 1, :]

                logkdown = logkavg - logkhalfdiff
                logdown_sum = tf.math.log(tf.reduce_sum(tf.exp(-logkdown) * self.norm[..., None], axis=(0,1)))[None, None, ...]
                logkdown = logkdown + logdown_sum - lognorm_sum

                logkup = logkavg + logkhalfdiff
                logup_sum = tf.math.log(tf.reduce_sum(tf.exp(logkup) * self.norm[..., None], axis=(0,1)))[None, None, ...]
                logkup = logkup - logup_sum + lognorm_sum

                # Compute new logkavg and logkhalfdiff
                logkavg = 0.5 * (logkup + logkdown)
                logkhalfdiff = 0.5 * (logkup - logkdown)

                # Stack logkavg and logkhalfdiff to form the new logk_array using tf.stack
                logk_array = tf.stack([logkavg, logkhalfdiff], axis=-2)

                # Finally, set self.logk to the new computed logk_array
                self.logk = logk_array
                self.norm = self.norm * (data_sum / norm_sum)[None, None, ...]


class FitDebugData:
    def __init__(self, indata):

        if indata.sparse:
            raise NotImplementedError("sparse mode is not supported yet")

        self.indata = indata

        self.axis_procs = self.indata.axis_procs
        self.axis_systs = hist.axis.StrCategory(indata.systs, name="systs")
        self.axis_downup = hist.axis.StrCategory(["Down", "Up"], name="DownUp")

        self.data_obs_hists = {}
        self.nominal_hists = {}
        self.syst_hists = {}
        self.syst_active_hists = {}

        ibin = 0
        for channel, info in self.indata.channel_info.items():
            axes = info["axes"]
            shape = [len(a) for a in axes]
            stop = ibin+np.prod(shape)

            shape_norm = [*shape, self.indata.nproc]
            shape_logk = [*shape, self.indata.nproc, 2, self.indata.nsyst]

            data_obs_hist = hist.Hist(*axes, name=f"{channel}_data_obs")
            data_obs_hist.values()[...] = memoryview(tf.reshape(self.indata.data_obs[ibin:stop], shape))

            nominal_hist = hist.Hist(*axes, self.axis_procs, name=f"{channel}_nominal")
            nominal_hist.values()[...] = memoryview(tf.reshape(self.indata.norm[ibin:stop,:], shape_norm))

            # TODO do these operations on logk in tensorflow instead of numpy to use
            # multiple cores
            logk_array = np.asarray(memoryview(tf.reshape(self.indata.logk[ibin:stop,:], shape_logk)))
            logkavg = logk_array[..., 0, :]
            logkhalfdiff = logk_array[..., 1, :]

            syst_hist = hist.Hist(*axes, self.axis_procs, self.axis_systs, self.axis_downup, name=f"{channel}_syst")

            logkdown = logkavg - logkhalfdiff
            syst_hist[{"DownUp" : "Down"}] = np.exp(-logkdown)*nominal_hist.values()[..., None]
            del logkdown

            logkup = logkavg + logkhalfdiff
            syst_hist[{"DownUp" : "Up"}] = np.exp(logkup)*nominal_hist.values()[..., None]
            del logkup

            syst_active_hist = hist.Hist(self.axis_procs, self.axis_systs, name=f"{channel}_syst_active", storage = hist.storage.Int64())
            syst_active_hist.values()[...] = np.sum(np.logical_or(np.abs(logkavg) > 0., np.abs(logkhalfdiff) > 0.), axis=tuple(range(len(axes))))

            self.data_obs_hists[channel] = data_obs_hist
            self.nominal_hists[channel] = nominal_hist
            self.syst_hists[channel] = syst_hist
            self.syst_active_hists[channel] = syst_active_hist

            ibin = stop

    def nonzeroSysts(self, channels = None, procs = None):
        if channels is None:
            channels = self.indata.channel_info.keys()

        if procs is None:
            procs = list(self.axis_procs)

        nonzero_syst_idxs = set()

        for channel in channels:
            syst_active_hist = self.syst_active_hists[channel]
            syst_active_hist = syst_active_hist[{"processes" : procs}]
            idxs = np.nonzero(syst_active_hist)
            syst_axis_idxs = syst_active_hist.axes.name.index("systs")
            syst_idxs = idxs[syst_axis_idxs]
            nonzero_syst_idxs.update(syst_idxs)

        nonzero_systs = []
        for isyst, syst in enumerate(self.indata.systs):
            if isyst in nonzero_syst_idxs:
                nonzero_systs.append(syst)

        return nonzero_systs

    def channelsForNonzeroSysts(self, procs = None, systs = None):
        if procs is None:
            procs = list(self.axis_procs)

        if systs is None:
            systs = list(self.axis_systs)

        channels_out = []

        for channel in self.indata.channel_info:
            syst_active_hist = self.syst_active_hists[channel]
            syst_active_hist = syst_active_hist[{"processes" : procs, "systs" : systs}]
            if np.count_nonzero(syst_active_hist.values()) > 0:
                channels_out.append(channel)

        return channels_out

    def procsForNonzeroSysts(self, channels = None, systs = None):
        if channels is None:
            channels = self.indata.channel_info.keys()

        if systs is None:
            systs = list(self.axis_systs)

        proc_idxs_out = set()

        for channel in channels:
            syst_active_hist = self.syst_active_hists[channel]
            syst_active_hist = syst_active_hist[{"systs" : systs}]
            idxs = np.nonzero(syst_active_hist)
            proc_axis_idxs = syst_active_hist.axes.name.index("processes")
            proc_idxs = idxs[proc_axis_idxs]
            proc_idxs_out.update(proc_idxs)

        nonzero_procs = []
        for iproc, proc in enumerate(self.indata.procs):
            if iproc in proc_idxs_out:
                nonzero_procs.append(proc)

        return nonzero_procs

class Fitter:
    def __init__(self, indata, options):
        self.indata = indata
        self.binByBinStat = options.binByBinStat
        self.normalize = options.normalize
        self.systgroupsfull = self.indata.systgroups.tolist()
        self.systgroupsfull.append("stat")
        if self.binByBinStat:
            self.systgroupsfull.append("binByBinStat")

        if options.externalCovariance and not options.chisqFit:
            raise Exception('option "--externalCovariance" only works with "--chisqFit"')
        if (options.chisqFit or options.externalCovariance) and options.binByBinStat:
            raise Exception('option "--binByBinStat" currently not supported for options "--externalCovariance" and "--chisqFit"')

        self.chisqFit = options.chisqFit
        self.externalCovariance = options.externalCovariance

        self.nsystgroupsfull = len(self.systgroupsfull)

        self.pois = []

        if options.POIMode == "mu":
            self.npoi = self.indata.nsignals
            poidefault = options.POIDefault*tf.ones([self.npoi],dtype=self.indata.dtype)
            for signal in self.indata.signals:
                self.pois.append(signal)
        elif options.POIMode == "none":
            self.npoi = 0
            poidefault = tf.zeros([],dtype=dtype)
        else:
            raise Exception("unsupported POIMode")

        self.parms = np.concatenate([self.pois, self.indata.systs])

        self.allowNegativePOI = options.allowNegativePOI

        if self.allowNegativePOI:
            xpoidefault = poidefault
        else:
            xpoidefault = tf.sqrt(poidefault)

        # tf variable containing all fit parameters
        thetadefault = tf.zeros([self.indata.nsyst],dtype=self.indata.dtype)
        if self.npoi>0:
            xdefault = tf.concat([xpoidefault,thetadefault], axis=0)
        else:
            xdefault = thetadefault

        self.x = tf.Variable(xdefault, trainable=True, name="x")

        # observed number of events per bin
        self.nobs = tf.Variable(self.indata.data_obs, trainable=False, name="nobs")

        if self.chisqFit:
            if self.externalCovariance:
                if self.indata.data_cov_inv is None:
                    raise RuntimeError("No external covariance found in input data.")
                # provided covariance
                self.data_cov_inv = self.indata.data_cov_inv
            else:
                # covariance from data stat
                if any(self.nobs<=0):
                    raise RuntimeError("Bins in 'nobs <= 0' encountered, chi^2 fit can not be performed.")
                self.data_cov_inv = tf.diag(tf.reciprocal(self.nobs))

        # constraint minima for nuisance parameters
        self.theta0 = tf.Variable(tf.zeros([self.indata.nsyst],dtype=self.indata.dtype), trainable=False, name="theta0")

        # global observables for mc stat uncertainty
        self.beta0 = tf.ones_like(self.indata.data_obs)

        nexpfullcentral = self.expected_events(profile=False)
        self.nexpnom = tf.Variable(nexpfullcentral, trainable=False, name="nexpnom")

    def prefit_covariance(self, unconstrained_err=0.):
        # free parameters are taken to have zero uncertainty for the purposes of prefit uncertainties
        var_poi = tf.zeros([self.npoi], dtype=self.indata.dtype)

        # nuisances have their uncertainty taken from the constraint term, but unconstrained nuisances
        # are set to a placeholder uncertainty (zero by default) for the purposes of prefit uncertainties
        var_theta = tf.where(self.indata.constraintweights == 0., unconstrained_err, tf.math.reciprocal(self.indata.constraintweights))

        invhessianprefit = tf.linalg.diag(tf.concat([var_poi, var_theta], axis = 0))
        return invhessianprefit

    @tf.function
    def val_jac(self, fun, *args, **kwargs):
        with tf.GradientTape() as t:
            val = fun(*args, **kwargs)
        jac = t.jacobian(val, self.x)

        return val, jac

    def bayesassign(self):
        if self.npoi > 0:
            raise NotImplementedError("Assignment for Bayesian toys is not currently supported in the presence of explicit POIs")
        self.x.assign(tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype))

    def frequentistassign(self):
        self.theta0.assign(tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype))

    def pulls_and_constraints(self, cov):
        systs = list(self.indata.systs.astype(str))
        axis_systs = hist.axis.StrCategory(systs, name="systs")

        pulls = self.x - self.theta0

        h_pulls = hist.Hist(axis_systs, storage=hist.storage.Double(), name="pulls")
        h_pulls.values()[...] = memoryview(pulls)
        h_pulls = narf.ioutils.H5PickleProxy(h_pulls)

        constraints = tf.sqrt(tf.linalg.diag_part(cov))

        h_constraints = hist.Hist(axis_systs, storage=hist.storage.Double(), name="constraints")
        h_constraints.values()[...] = memoryview(constraints)
        h_constraints = narf.ioutils.H5PickleProxy(h_constraints)

        return h_pulls, h_constraints

    @tf.function(reduce_retracing=True)
    def _compute_impact_group(self, cov, nstat, idxs):
        cov_reduced = tf.gather(cov, idxs, axis=0)
        cov_reduced = tf.gather(cov_reduced, idxs, axis=1)
        v = tf.gather(cov[:nstat,:], idxs, axis=1)
        invC_v = tf.linalg.solve(cov_reduced, tf.transpose(v)) 
        v_invC_v = tf.reduce_sum(v * tf.transpose(invC_v), axis=1)
        return tf.reshape(tf.sqrt(v_invC_v), (1,-1))

    @tf.function
    def _impacts_systs(self, nstat, cov, hess):

        impacts = cov[:nstat,:]

        systgroupidxs = tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32)

        impacts_grouped = tf.concat([
            self._compute_impact_group(cov, nstat, idxs)
            for idxs in self.indata.systgroupidxs
        ], axis=1)

        # impact data stat
        hess_stat = hess[:nstat,:nstat]
        identity = tf.eye(nstat, dtype=hess_stat.dtype)
        inv_hess_stat = tf.linalg.solve(hess_stat, identity)  # Solves H * X = I
        impacts_data_stat = tf.sqrt(tf.linalg.diag_part(inv_hess_stat))
        impacts_data_stat = tf.reshape(impacts_data_stat, (1, -1))

        impacts = tf.concat([impacts, impacts_data_stat], axis=1)
        impacts_grouped = tf.concat([impacts_grouped, impacts_data_stat], axis=1)

        if self.binByBinStat:
            # impact bin-by-bin stat
            val_no_bbb, grad_no_bbb, hess_no_bbb = self.loss_val_grad_hess(profile_grad=False)

            hess_stat_no_bbb = hess_no_bbb[:nstat,:nstat]
            inv_hess_stat_no_bbb = tf.linalg.solve(hess_stat_no_bbb, identity)
            impacts_bbb_sq = tf.linalg.diag_part(inv_hess_stat - inv_hess_stat_no_bbb)
            impacts_bbb = tf.sqrt(tf.maximum(tf.zeros_like(impacts_bbb_sq), impacts_bbb_sq))
            impacts_bbb = tf.reshape(impacts_bbb, (1, -1))

            impacts = tf.concat([impacts, impacts_bbb], axis=1)
            impacts_grouped = tf.concat([impacts_grouped, impacts_bbb], axis=1)

        return impacts, impacts_grouped

    def impacts_systs(self, cov, hess):

        # store impacts for all POIs and unconstrained nuisances
        nstat = self.npoi + self.indata.nsystnoconstraint

        systs = list(self.indata.systs.astype(str))[:nstat]

        impact_names = list(self.indata.systs.astype(str))
        impact_names_grouped = list(self.indata.systgroups.astype(str))

        impacts, impacts_grouped = self._impacts_systs(nstat, cov, hess)

        impact_names.append("stat")
        impact_names_grouped.append("stat")

        if self.binByBinStat:
            impact_names.append("binByBinStat")
            impact_names_grouped.append("binByBinStat")

        # write out histograms
        axis_systs = hist.axis.StrCategory(systs, name="systs")
        axis_impacts = hist.axis.StrCategory(impact_names, name="inpacts")
        axis_impacts_grouped = hist.axis.StrCategory(impact_names_grouped, name="inpacts")

        h = hist.Hist(axis_systs, axis_impacts, storage=hist.storage.Double(), name="impacts")
        h.values()[...] = memoryview(impacts)
        h = narf.ioutils.H5PickleProxy(h)

        h_grouped = hist.Hist(axis_systs, axis_impacts_grouped, storage=hist.storage.Double(), name="impacts_grouped")
        h_grouped.values()[...] = memoryview(impacts_grouped)
        h_grouped = narf.ioutils.H5PickleProxy(h)

        return h, h_grouped

    def _global_impacts(self, cov):
        with tf.GradientTape() as t2:
            t2.watch([self.theta0, self.nobs, self.beta0])
            with tf.GradientTape() as t1:
                t1.watch([self.theta0, self.nobs, self.beta0])
                val = self._compute_loss()
            grad = t1.gradient(val, self.x, unconnected_gradients="zero")
        pd2ldxdtheta0, pd2ldxdnobs, pd2ldxdbeta0 = t2.jacobian(grad, [self.theta0, self.nobs, self.beta0], unconnected_gradients="zero")

        dxdtheta0 = -cov @ pd2ldxdtheta0
        dxdnobs = -cov @ pd2ldxdnobs
        dxdbeta0 = -cov @ pd2ldxdbeta0

        return dxdtheta0, dxdnobs, dxdbeta0

    @tf.function(reduce_retracing=True)
    def _compute_global_impact_group(self, dxdtheta0, idxs):
        gathered = tf.gather(dxdtheta0, idxs, axis=-1)
        squared = tf.square(gathered)
        summed = tf.reduce_sum(squared, axis=-1)
        return tf.reshape(tf.sqrt(summed), (1,-1))

    @tf.function
    def _global_impacts_systs(self, cov):

        dxdtheta0, dxdnobs, dxdbeta0 = self._global_impacts(cov)
        
        # group grobal impacts
        systgroupidxs = tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32)

        impacts_grouped = tf.concat([
            self._compute_global_impact_group(dxdtheta0, idxs)
            for idxs in self.indata.systgroupidxs
        ], axis=1)

        # global impact data stat
        data_stat = tf.sqrt(tf.reduce_sum(tf.square(dxdnobs) * self.nobs, axis=-1))
        impacts_data_stat = tf.reshape(data_stat, (1, -1))
        impacts = tf.concat([dxdtheta0, impacts_data_stat], axis=1)
        impacts_grouped = tf.concat([impacts_grouped, impacts_data_stat], axis=1)

        if self.binByBinStat:
            # global impact bin-by-bin stat
            impacts_bbb = tf.sqrt(tf.reduce_sum(tf.square(dxdbeta0) * tf.math.reciprocal(self.indata.kstat), axis=-1))
            impacts_bbb = tf.reshape(impacts_bbb, (1, -1))
            impacts = tf.concat([impacts, impacts_bbb], axis=1)
            impacts_grouped = tf.concat([impacts_grouped, impacts_bbb], axis=1)

        return impacts, impacts_grouped

    def global_impacts_systs(self, cov):
        # store impacts for all POIs and unconstrained nuisances
        nstat = self.npoi + self.indata.nsystnoconstraint

        systs = list(self.indata.systs.astype(str))[:nstat]

        impacts, impacts_grouped = self._global_impacts_systs(cov[:nstat,:])

        impact_names = list(self.indata.systs.astype(str))
        impact_names_grouped = list(self.indata.systgroups.astype(str))

        # global impact data stat
        impact_names.append("stat")
        impact_names_grouped.append("stat")

        if self.binByBinStat:
            # global impact bin-by-bin stat
            impact_names.append("binByBinStat")
            impact_names_grouped.append("binByBinStat")

        # write out histograms
        axis_systs = hist.axis.StrCategory(systs, name="systs")
        axis_impacts = hist.axis.StrCategory(impact_names, name="inpacts")
        axis_impacts_grouped = hist.axis.StrCategory(impact_names_grouped, name="inpacts")

        h = hist.Hist(axis_systs, axis_impacts, storage=hist.storage.Double(), name="global_impacts")
        h.values()[...] = memoryview(impacts)
        h = narf.ioutils.H5PickleProxy(h)

        h_grouped = hist.Hist(axis_systs, axis_impacts_grouped, storage=hist.storage.Double(), name="global_impacts_grouped")
        h_grouped.values()[...] = memoryview(impacts_grouped)
        h_grouped = narf.ioutils.H5PickleProxy(h)

        return h, h_grouped

    def _expvar_profiled(self, fun_exp, cov, compute_cov=False):
        dxdtheta0, dxdnobs, dxdbeta0 = self._global_impacts(cov)

        with tf.GradientTape() as t:
            t.watch([self.theta0, self.nobs, self.beta0])
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))

        pdexpdx, pdexpdtheta0, pdexpdnobs, pdexpdbeta0 = t.jacobian(expected_flat, [self.x, self.theta0, self.nobs, self.beta0], unconnected_gradients="zero")

        dexpdtheta0 = pdexpdtheta0 + pdexpdx @ dxdtheta0
        dexpdnobs = pdexpdnobs + pdexpdx @ dxdnobs
        dexpdbeta0 = pdexpdbeta0 + pdexpdx @ dxdbeta0

        # FIXME factorize this part better with the global impacts calculation

        var_theta0 = tf.where(self.indata.constraintweights == 0., tf.zeros_like(self.indata.constraintweights), tf.math.reciprocal(self.indata.constraintweights))

        dtheta0 = tf.math.sqrt(var_theta0)
        dnobs = tf.math.sqrt(self.nobs)
        dbeta0 = tf.math.sqrt(tf.math.reciprocal(self.indata.kstat))

        dexpdtheta0 *= dtheta0[None, :]
        dexpdnobs *= dnobs[None, :]
        dexpdbeta0 *= dbeta0[None, :]

        if compute_cov:
            expcov = dexpdtheta0 @ tf.transpose(dexpdtheta0) + dexpdnobs @ tf.transpose(dexpdnobs)
            if self.binByBinStat:
                expcov += dexpdbeta0 @ tf.transpose(dexpdbeta0)
            return expected, expcov
        else:
            expvar = tf.reduce_sum(tf.square(dexpdtheta0), axis=-1) + tf.reduce_sum(tf.square(dexpdnobs), axis=-1)
            if self.binByBinStat:
                expvar += tf.reduce_sum(tf.square(dexpdbeta0), axis=-1)

            expvar = tf.reshape(expvar, expected.shape)

            return expected, expvar

    def _expvar_optimized(self, fun_exp, cov, skipBinByBinStat = False):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix

        #FIXME this doesn't actually work for the positive semi-definite case
        invhesschol = tf.linalg.cholesky(cov)

        # since the full covariance matrix with respect to the bin counts is given by J^T R^T R J, then summing RJ element-wise squared over the parameter axis gives the diagonal elements

        expected = fun_exp()

        # dummy vector for implicit transposition
        u = tf.ones_like(expected)
        with tf.GradientTape(watch_accessed_variables=False) as t1:
            t1.watch(u)
            with tf.GradientTape() as t2:
                expected = fun_exp()
            # this returns dndx_j = sum_i u_i dn_i/dx_j
            Ju = t2.gradient(expected, self.x, output_gradients=u)
            Ju = tf.transpose(Ju)
            Ju = tf.reshape(Ju, [-1, 1])
            RJu = tf.matmul(tf.stop_gradient(invhesschol), Ju, transpose_a=True)
            RJu = tf.reshape(RJu, [-1])
        RJ = t1.jacobian(RJu, u)
        sRJ2 = tf.reduce_sum(RJ**2, axis=0)
        sRJ2 = tf.reshape(sRJ2, expected.shape)
        if self.binByBinStat and not skipBinByBinStat:
            # add MC stat uncertainty on variance
            sumw2 = tf.square(expected)/self.indata.kstat
            sRJ2 = sRJ2 + sumw2
        return expected, sRJ2

    def _chi2(self, fun, cov, profile=True):
        if profile:
            res, rescov = self._expvar_profiled(fun, cov, compute_cov=True)
        else:
            res, rescov = self._expvar(fun, cov, compute_cov=True)

        resv = tf.reshape(res, (-1,1))

        chi_square_value = tf.transpose(resv) @ tf.linalg.solve(rescov, resv)

        return chi_square_value[0,0]

    def _expvar(self, fun_exp, invhess, compute_cov=False):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix
        #FIXME switch back to optimized version at some point?

        with tf.GradientTape() as t:
            t.watch([self.nobs, self.beta0])
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))
        dexpdx, dexpdnobs, dexpdbeta0 = t.jacobian(expected_flat, [self.x, self.nobs, self.beta0])

        cov = dexpdx @ tf.matmul(invhess, dexpdx, transpose_b = True)

        if dexpdnobs is not None:
            varnobs = self.nobs
            cov += dexpdnobs @ (varnobs[:, None] * tf.transpose(dexpdnobs))

        if self.binByBinStat:
            varbeta0 = tf.math.reciprocal(self.indata.kstat)
            cov += dexpdbeta0 @ (varbeta0[:, None] * tf.transpose(dexpdbeta0))

        if compute_cov:
            return expected, cov
        else:
            var = tf.linalg.diag_part(cov)
            var = tf.reshape(var, expected.shape)
            return expected, var

    def _expvariations(self, fun_exp, cov, correlations):
        with tf.GradientTape() as t:
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))
        dexpdx = t.jacobian(expected_flat, self.x)

        if correlations:
            # construct the matrix such that the columns represent
            # the variations associated with profiling a given parameter
            # taking into account its correlations with the other parameters
            dx = cov/tf.math.sqrt(tf.linalg.diag_part(cov))[None, :]

            dexp = dexpdx @ dx
        else:
            dexp = dexpdx*tf.math.sqrt(tf.linalg.diag_part(cov))[None, :]

        dexp = tf.reshape(dexp, (*expected.shape, -1))

        down = expected[..., None] - dexp
        up = expected[..., None] + dexp

        expvars = tf.stack([down, up], axis=-1)

        return expvars

    def _compute_yields_noBBB(self):
        xpoi = self.x[:self.npoi]
        theta = self.x[self.npoi:]

        if self.allowNegativePOI:
            poi = xpoi
            gradr = tf.ones_like(poi)
        else:
            poi = tf.square(xpoi)
            gradr = 2.*xpoi

        rnorm = tf.concat([poi, tf.ones([self.indata.nproc - poi.shape[0]], dtype=self.indata.dtype)], axis=0)
        mrnorm = tf.expand_dims(rnorm,-1)
        ernorm = tf.reshape(rnorm,[1,-1])

        #interpolation for asymmetric log-normal
        twox = 2.*theta
        twox2 = twox*twox
        alpha =  0.125 * twox * (twox2 * (3.*twox2 - 10.) + 15.)
        alpha = tf.clip_by_value(alpha,-1.,1.)

        thetaalpha = theta*alpha

        mthetaalpha = tf.stack([theta,thetaalpha],axis=0) #now has shape [2,nsyst]
        mthetaalpha = tf.reshape(mthetaalpha,[2*self.indata.nsyst,1])

        if self.indata.sparse:
            raise NotImplementedError("sparse mode is not supported yet")
        else:
            #matrix encoding effect of nuisance parameters
            #memory efficient version (do summation together with multiplication in a single tensor contraction step)
            #this is equivalent to
            #alpha = tf.reshape(alpha,[-1,1,1])
            #theta = tf.reshape(theta,[-1,1,1])
            #logk = logkavg + alpha*logkhalfdiff
            #logktheta = theta*logk
            #logsnorm = tf.reduce_sum(logktheta, axis=0)

            mlogk = tf.reshape(self.indata.logk,[self.indata.nbinsfull*self.indata.nproc,2*self.indata.nsyst])
            logsnorm = tf.matmul(mlogk,mthetaalpha)
            logsnorm = tf.reshape(logsnorm,[self.indata.nbinsfull,self.indata.nproc])

            snorm = tf.exp(logsnorm)

            #final expected yields per-bin including effect of signal
            #strengths and nuisance parmeters
            #memory efficient version (do summation together with multiplication in a single tensor contraction step)
            #equivalent to (with some reshaping to explicitly match indices)
            #rnorm = tf.reshape(rnorm,[1,-1])
            #pnormfull = rnorm*snorm*norm
            #nexpfull = tf.reduce_sum(pnormfull,axis=-1)
            snormnorm = snorm*self.indata.norm
            nexpfullcentral = tf.matmul(snormnorm, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral,-1)

            snormnormmasked = snormnorm[self.indata.nbins:]

            normmasked = self.indata.norm[self.indata.nbins:]

            # if options.saveHists:
            normfullcentral = ernorm*snormnorm

        if self.normalize:
            # FIXME this should be done per-channel ideally
            normscale = tf.reduce_sum(self.nobs)/tf.reduce_sum(nexpfullcentral)

            nexpfullcentral *= normscale
            normfullcentral *= normscale

        return nexpfullcentral, normfullcentral

    def _compute_yields_with_beta(self, profile=True, profile_grad=True):
        nexpfullcentral, normfullcentral = self._compute_yields_noBBB()

        nexpfull = nexpfullcentral
        normfull = normfullcentral

        beta = None
        if self.binByBinStat:
            if profile:
                beta = (self.nobs + self.indata.kstat)/(nexpfullcentral+self.indata.kstat)
                if not profile_grad:
                    beta = tf.stop_gradient(beta)
            else:
                beta = self.beta0
            nexpfull = beta*nexpfullcentral
            normfull = beta[..., None]*normfullcentral

            if self.normalize:
                # FIXME this is probably not fully consistent when combined with the binByBinStat
                normscale = tf.reduce_sum(self.nobs)/tf.reduce_sum(nexpfull)

                nexpfull *= normscale
                normfull *= normscale

        return nexpfull, normfull, beta

    def _compute_yields(self, inclusive=True, profile=True, profile_grad=True):
        nexpfullcentral, normfullcentral, beta = self._compute_yields_with_beta(profile=profile, profile_grad=profile_grad)
        if inclusive:
            return nexpfullcentral
        else:
            return normfullcentral

    @tf.function
    def expected_with_variance(self, fun, cov, profile=True):
        if profile:
            return self._expvar_profiled(fun, cov)
        else:
            return self._expvar(fun, cov)

    @tf.function
    def expected_variations(self, fun, cov, correlations=False):
        return self._expvariations(fun, cov, correlations=correlations)

    def expected_hists(self, cov=None, inclusive=True, compute_variance=True, compute_variations=False, correlated_variations=False, profile=True, profile_grad=True, compute_chi2=False, name=None, label=None):

        def fun():
            return self._compute_yields(inclusive=inclusive, profile=profile, profile_grad=profile_grad)

        if compute_variance and compute_variations:
            raise NotImplementedError()

        if compute_variance:
            exp, var = self.expected_with_variance(fun, cov, profile=profile)
        elif compute_variations:
            exp = self.expected_variations(fun, cov, correlations=correlated_variations)
        else:
            exp = tf.function(fun)()

        hists = {}

        for channel, info in self.indata.channel_info.items():
            if "masked" not in channel:

                axes = info["axes"]

                start = info["start"]
                stop = info["stop"]

                hist_axes = axes.copy()

                if not inclusive:
                    hist_axes.append(self.indata.axis_procs)

                if compute_variations:
                    axis_vars = hist.axis.StrCategory(self.parms, name="vars")
                    axis_downUpVar = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")

                    hist_axes.extend([axis_vars, axis_downUpVar])

                shape = tuple([len(a) for a in hist_axes])

                h = hist.Hist(*hist_axes, storage=hist.storage.Weight(), name=f"name_{channel}", label=label)
                h.values()[...] = memoryview(tf.reshape(exp[start:stop], shape))
                if compute_variance:
                    h.variances()[...] = memoryview(tf.reshape(var[start:stop], shape))
                else:
                    h.variances()[...] = 0.
                hists[channel] = narf.ioutils.H5PickleProxy(h)

        if compute_chi2:
            def fun_residual():
                return fun() - self.nobs

            chi2val = self.chi2(fun_residual, cov, profile=profile).numpy()
            ndf = tf.size(exp).numpy() - self.normalize

            return hists, chi2val, ndf
        else:
            return hists

    def expected_projection_hist(self, channel, axes, cov=None, inclusive=True, compute_variance=True, compute_variations=False, correlated_variations=False, profile=True, profile_grad=True, compute_chi2=False, name=None, label=None):

        def fun():
            return self._compute_yields(inclusive=inclusive, profile=profile, profile_grad=profile_grad)

        info = self.indata.channel_info[channel]
        start = info["start"]
        stop = info["stop"]

        channel_axes = info["axes"]

        exp_axes = channel_axes.copy()
        hist_axes = [axis for axis in channel_axes if axis.name in axes]

        if len(hist_axes) != len(axes):
            raise ValueError("axis not found")

        extra_axes = []
        if not inclusive:
            exp_axes.append(self.indata.axis_procs)
            hist_axes.append(self.indata.axis_procs)
            extra_axes.append(self.indata.axis_procs)

        if compute_variations:
            axis_vars = hist.axis.StrCategory(self.parms, name="vars")
            axis_downUpVar = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")

            hist_axes.extend([axis_vars, axis_downUpVar])

        exp_shape = tuple([len(a) for a in exp_axes])

        channel_axes_names = [axis.name for axis in channel_axes]
        exp_axes_names = [axis.name for axis in exp_axes]
        extra_axes_names = [axis.name for axis in extra_axes]

        axis_idxs = [channel_axes_names.index(axis) for axis in axes]

        proj_idxs = [i for i in range(len(channel_axes)) if i not in axis_idxs]

        post_proj_axes_names = [axis for axis in channel_axes_names if axis in axes] + extra_axes_names

        transpose_idxs = [post_proj_axes_names.index(axis) for axis in axes] +  [post_proj_axes_names.index(axis) for axis in extra_axes_names]

        def make_projection_fun(fun_flat):
            def proj_fun():
                exp = fun_flat()[start:stop]
                exp = tf.reshape(exp, exp_shape)
                exp = tf.reduce_sum(exp, axis=proj_idxs)
                exp = tf.transpose(exp, perm=transpose_idxs)

                return exp

            return proj_fun

        projection_fun = make_projection_fun(fun)

        if compute_variance and compute_variations:
            raise NotImplementedError()

        if compute_variance:
            exp, var = self.expected_with_variance(projection_fun, cov, profile=profile)
        elif compute_variations:
            exp = self.expected_variations(projection_fun, cov, correlations=correlated_variations)
        else:
            exp = tf.function(projection_fun)()

        h = hist.Hist(*hist_axes, storage=hist.storage.Weight(), name=name, label=label)
        h.values()[...] = memoryview(exp)
        if compute_variance:
            h.variances()[...] = memoryview(var)
        else:
            h.variances()[...] = 0.
        h = narf.ioutils.H5PickleProxy(h)

        if compute_chi2:
            def fun_residual():
                return fun() - self.nobs

            projection_fun_residual = make_projection_fun(fun_residual)

            chi2val = self.chi2(projection_fun_residual, cov, profile=profile).numpy()
            ndf = tf.size(exp).numpy() - self.normalize

            return h, chi2val, ndf
        else:
            return h

    def observed_hists(self):
        hists_data_obs = {}
        hists_nobs = {}

        for channel, info in self.indata.channel_info.items():
            if "masked" not in channel:

                axes = info["axes"]

                start = info["start"]
                stop = info["stop"]

                shape = tuple([len(a) for a in axes])

                hist_data_obs = hist.Hist(*axes, storage=hist.storage.Weight(), name = "data_obs", label="observed number of events in data")
                hist_data_obs.values()[...] = memoryview(tf.reshape(self.indata.data_obs[start:stop], shape))
                hist_data_obs.variances()[...] = hist_data_obs.values()
                hists_data_obs[channel] = narf.ioutils.H5PickleProxy(hist_data_obs)

                hist_nobs = hist.Hist(*axes, storage=hist.storage.Weight(), name = "nobs", label = "observed number of events for fit")
                hist_nobs.values()[...] = memoryview(tf.reshape(self.nobs.value()[start:stop], shape))
                hist_nobs.variances()[...] = hist_nobs.values()
                hists_nobs[channel] = narf.ioutils.H5PickleProxy(hist_nobs)

        return hists_data_obs, hists_nobs

    @tf.function
    def expected_events(self, profile=True):
        return self._compute_yields(inclusive=True, profile=profile)

    @tf.function
    def chi2(self, fun, cov, profile=True):
        return self._chi2(fun, cov, profile=profile)

    @tf.function
    def saturated_nll(self):
        nobs = self.nobs

        nobsnull = tf.equal(nobs,tf.zeros_like(nobs))

        #saturated model
        nobssafe = tf.where(nobsnull, tf.ones_like(nobs), nobs)
        lognobs = tf.math.log(nobssafe)

        lsaturated = tf.reduce_sum(-nobs*lognobs + nobs, axis=-1)

        ndof = tf.size(nobs) - self.npoi - self.indata.nsystnoconstraint - self.normalize

        return lsaturated, ndof

    @tf.function
    def full_nll(self):
        l, lfull = self._compute_nll()
        return lfull

    def _compute_nll(self, profile=True, profile_grad=True):
        theta = self.x[self.npoi:]

        nexpfullcentral, normfullcentral, beta = self._compute_yields_with_beta(profile=profile, profile_grad=profile_grad)

        nexp = nexpfullcentral

        if self.chisqFit:
            residual = tf.reshape(self.nobs - nexp,[-1,1]) #chi2 residual  
            ln = lnfull = 0.5 * tf.reduce_sum(tf.matmul(residual, tf.matmul(self.data_cov_inv, residual), transpose_a=True))
        else:
            nobsnull = tf.equal(self.nobs,tf.zeros_like(self.nobs))

            nexpsafe = tf.where(nobsnull, tf.ones_like(self.nobs), nexp)
            lognexp = tf.math.log(nexpsafe)

            nexpnomsafe = tf.where(nobsnull, tf.ones_like(self.nobs), self.nexpnom)
            lognexpnom = tf.math.log(nexpnomsafe)

            #final likelihood computation

            #poisson term
            lnfull = tf.reduce_sum(-self.nobs*lognexp + nexp, axis=-1)

            #poisson term with offset to improve numerical precision
            ln = tf.reduce_sum(-self.nobs*(lognexp-lognexpnom) + nexp-self.nexpnom, axis=-1)

        #constraints
        lc = tf.reduce_sum(self.indata.constraintweights*0.5*tf.square(theta - self.theta0))

        l = ln + lc
        lfull = lnfull + lc

        if self.binByBinStat:
            lbetavfull = -self.indata.kstat*tf.math.log(beta/self.beta0) + self.indata.kstat*beta/self.beta0

            lbetav = lbetavfull - self.indata.kstat
            lbeta = tf.reduce_sum(lbetav)

            l = l + lbeta
            lfull = lfull + lbeta

        return l, lfull

    def _compute_loss(self, profile=True, profile_grad=True):
        l, lfull = self._compute_nll(profile=profile, profile_grad=profile_grad)
        return l

    @tf.function
    def loss_val(self):
        val = self._compute_loss()
        return val

    @tf.function
    def loss_val_grad(self):
        with tf.GradientTape() as t:
            val = self._compute_loss()
        grad = t.gradient(val, self.x)

        return val, grad

    @tf.function
    def loss_val_grad_hessp(self, p):
        with tf.autodiff.ForwardAccumulator(self.x, p) as acc:
            with tf.GradientTape() as grad_tape:
                val = self._compute_loss()
            grad = grad_tape.gradient(val, self.x)
        hessp = acc.jvp(grad)

        return val, grad, hessp

    @tf.function
    def loss_val_grad_hess(self, profile=True, profile_grad=True):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_loss(profile=profile, profile_grad=profile_grad)
            grad = t1.gradient(val, self.x)
        hess = t2.jacobian(grad, self.x)

        return val, grad, hess

    def minimize(self):

        def scipy_loss(xval):
            self.x.assign(xval)
            val, grad = self.loss_val_grad()
            print("scipy_loss", val)
            return val.numpy(), grad.numpy()

        def scipy_hessp(xval, pval):
            self.x.assign(xval)
            p = tf.convert_to_tensor(pval)
            val, grad, hessp = self.loss_val_grad_hessp(p)
            print("scipy_hessp", val)
            return hessp.numpy()

        def scipy_hess(xval):
            self.x.assign(xval)
            val, grad, hess = self.loss_val_grad_hess()
            print("scipy_hess", val)
            return hess.numpy()


        xval = self.x.numpy()

        res = scipy.optimize.minimize(scipy_loss, xval, method = "trust-krylov", jac = True, hessp = scipy_hessp)

        xval = res["x"]

        self.x.assign(xval)

        print(res)

        return res


