import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import h5py

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
    def __init__(self, filename):
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

            #build tensorflow graph for likelihood calculation

            #start by creating tensors which read in the hdf5 arrays (optimized for memory consumption)
            self.constraintweights = maketensor(hconstraintweights)
            self.data_obs = maketensor(hdata_obs)
            hkstat = f['hkstat']
            self.kstat = maketensor(hkstat)

            if self.sparse:
                self.norm_sparse = makesparsetensor(hnorm_sparse)
                self.logk_sparse = makesparsetensor(hlogk_sparse)
            else:
                self.norm = maketensor(hnorm)
                self.logk = maketensor(hlogk)

class Fitter:
    def __init__(self, indata, options):
        self.indata = indata

        self.systgroupsfull = self.indata.systgroups.tolist()
        self.systgroupsfull.append("stat")
        if options.binByBinStat:
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

    def prefit_covariance(self):
        # free parameters are taken to have zero uncertainty for the purposes of prefit uncertainties
        var_poi = tf.zeros([self.npoi], dtype=self.indata.dtype)

        # nuisances have their uncertainty taken from the constraint term, but unconstrained nuisances
        # are set to zero uncertainty for the purposes of prefit uncertainties
        var_theta = tf.where(self.indata.constraintweights == 0., 0., tf.math.reciprocal(self.indata.constraintweights))

        invhessianprefit = tf.linalg.diag(tf.concat([var_poi, var_theta], axis = 0))
        return invhessianprefit


    @tf.function
    def val_jac(self, fun, *args, **kwargs):
        with tf.GradientTape() as t:
            val = fun(*args, **kwargs)
        jac = t.jacobian(val, self.x)

        return val, jac

    def _experr(self, fun_exp, invhesschol):
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
        return expected, sRJ2

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

        return expected, err


    def _compute_yields(self):
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

    def _compute_yields_inclusive(self):
        nexpfullcentral, normfullcentral = self._compute_yields()
        return nexpfullcentral

    def _compute_yields_per_process(self):
        nexpfullcentral, normfullcentral = self._compute_yields()
        return normfullcentral

    @tf.function
    def expected_events_per_process(self):
        return self._compute_yields_per_process()

    @tf.function
    def expected_events_inclusive_with_variance(self, invhesschol):
        return self._experr(self._compute_yields_inclusive, invhesschol)

