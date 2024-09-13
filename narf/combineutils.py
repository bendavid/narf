import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import scipy
import h5py
import hist

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

                data_sum = tf.reduce_sum(self.data_obs)
                norm_sum = tf.reduce_sum(self.norm)
                logdata_sum = tf.math.log(data_sum)[None, None, ...]

                logkavg = self.logk[..., 0, :]
                logkhalfdiff = self.logk[..., 1, :]

                logkdown = logkavg - logkhalfdiff
                logkdown_sum = tf.math.log(tf.reduce_sum(tf.exp(-logkdown) * self.norm[..., None], axis=(0,1)))[None, None, ...]
                logkdown = logkdown + logkdown_sum - logdata_sum

                logkup = logkavg + logkhalfdiff
                logkup_sum = tf.math.log(tf.reduce_sum(tf.exp(logkup) * self.norm[..., None], axis=(0,1)))[None, None, ...]
                logkup = logkup - logkup_sum + logdata_sum

                # Compute new logkavg and logkhalfdiff
                logkavg = 0.5 * (logkup + logkdown)
                logkhalfdiff = 0.0 * (logkup - logkdown) # Manually setting it to 0 TODO: FIXME?

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
            stop = ibin+np.product(shape)

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
        self.systgroupsfull = self.indata.systgroups.tolist()
        self.systgroupsfull.append("stat")
        if self.binByBinStat:
            self.systgroupsfull.append("binByBinStat")

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

        # constraint minima for nuisance parameters
        self.theta0 = tf.Variable(tf.zeros([self.indata.nsyst],dtype=self.indata.dtype), trainable=False, name="theta0")

        nexpfullcentral = self.expected_events_noBBB()
        self.nexpnom = tf.Variable(nexpfullcentral, trainable=False, name="nexpnom")

    def prefit_covariance(self):
        # free parameters are taken to have zero uncertainty for the purposes of prefit uncertainties
        var_poi = tf.zeros([self.npoi], dtype=self.indata.dtype)

        # nuisances have their uncertainty taken from the constraint term, but unconstrained nuisances
        # are set to zero uncertainty for the purposes of prefit uncertainties
        var_theta = tf.where(self.indata.constraintweights == 0., 0., tf.math.reciprocal(self.indata.constraintweights))

        invhessianprefit = tf.linalg.diag(tf.concat([var_poi, var_theta], axis = 0))
        return invhessianprefit

    def chi2(self, invhess):
        chi2 = self._chi2_pedantic(self._compute_yields_inclusive, self.indata.data_obs, invhess)
        chi2_val = np.array([1])
        chi2_val[...] = memoryview(chi2)
        return chi2_val[0]

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

    def _experr(self, fun_exp, invhesschol, skipBinByBinStat = False):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix

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

    def _chi2_pedantic(self, fun_exp, observed, invhess):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix
        # (extremely cpu and memory-inefficient version for validation purposes only)

        with tf.GradientTape() as t:
            expected = fun_exp()
            J = t.jacobian(expected, self.x)

        residual = observed - expected
        residual_flat = tf.reshape(residual, (-1, 1))

        # error propagation of covariance between nuisances into covariance between bins
        K = J @invhess @ tf.transpose(J)
        # add data uncertainty on covariance
        K = K + tf.linalg.diag(observed)
        if self.binByBinStat:
            # add MC stat uncertainty on covariance
            sumw2 = tf.square(expected)/self.indata.kstat
            K = K + tf.linalg.diag(sumw2)
        
        Kinv = tf.linalg.inv(K)

        # chi2 = uT*Kinv*u
        chi_square_value = tf.transpose(residual_flat) @ Kinv @ residual_flat

        return chi_square_value

    def _experr_pedantic(self, fun_exp, invhess):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix
        # (extremely cpu and memory-inefficient version for validation purposes only)

        with tf.GradientTape() as t:
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1, 1))
        J = t.jacobian(expected_flat, self.x)

        cov = J @ tf.matmul(invhess, J, transpose_b = True)

        err = tf.linalg.diag_part(cov)
        err = tf.reshape(err, expected.shape)

        if self.binByBinStat:
            # add MC stat uncertainty on variance
            sumw2 = tf.square(expected)/self.indata.kstat
            err = err + sumw2

        return expected, err


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

        return nexpfullcentral, normfullcentral

        # nexpfull = nexpfullcentral
        # normfull = normfullcentral

    def _compute_yields(self):
        nexpfullcentral, normfullcentral = self._compute_yields_noBBB()

        nexpfull = nexpfullcentral
        normfull = normfullcentral

        beta = None
        if self.binByBinStat:
            beta = (self.nobs + self.indata.kstat)/(nexpfullcentral+self.indata.kstat)
            # tf.print("beta", beta)
            # print("beta.shape", beta.shape)
            # print("nexpfullcentral.shape", nexpfullcentral.shape)
            # betadiff = tf.reduce_max(tf.abs(beta-1.))
            # tf.print("betadiff", betadiff)
            nexpfull = beta*nexpfullcentral
            normfull = beta[..., None]*normfullcentral

        return nexpfull, normfull, beta

    def _compute_yields_inclusive(self):
        nexpfullcentral, normfullcentral, beta = self._compute_yields()
        return nexpfullcentral

    def _compute_yields_per_process(self):
        nexpfullcentral, normfullcentral, beta = self._compute_yields()
        return normfullcentral

    def _compute_yields_inclusive_noBBB(self):
        nexpfullcentral, normfullcentral = self._compute_yields_noBBB()
        return nexpfullcentral

    def _compute_yields_per_process_noBBB(self):
        nexpfullcentral, normfullcentral = self._compute_yields_noBBB()
        return normfullcentral

    @tf.function
    def expected_events(self):
        return self._compute_yields_inclusive()

    @tf.function
    def expected_events_noBBB(self):
        return self._compute_yields_inclusive_noBBB()

    @tf.function
    def expected_events_per_process(self):
        return self._compute_yields_per_process()

    @tf.function
    def expected_events_per_process_noBBB(self):
        return self._compute_yields_per_process_noBBB()

    @tf.function
    def expected_events_inclusive_with_variance_noBBB(self, invhesschol):
        return self._experr(self._compute_yields_inclusive_noBBB, invhesschol)

    @tf.function
    def expected_events_per_process_with_variance_noBBB(self, invhesschol):
        return self._experr(self._compute_yields_per_process_noBBB, invhesschol, skipBinByBinStat=True)

    def _compute_nll(self):
        theta = self.x[self.npoi:]

        nexpfullcentral, normfullcentral, beta = self._compute_yields()

        nexp = nexpfullcentral

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
            beta0 = tf.ones_like(beta)
            lbetavfull = -self.indata.kstat*tf.math.log(beta/beta0) + self.indata.kstat*beta/beta0

            lbetav = lbetavfull - self.indata.kstat
            lbeta = tf.reduce_sum(lbetav)

            l = l + lbeta
            lfull = lfull + lbeta

        return l, lfull

    def _compute_loss(self):
        l, lfull = self._compute_nll()
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
    def loss_val_grad_hess(self):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_loss()
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


