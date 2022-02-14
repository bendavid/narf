import hist
import hdf5plugin
import h5py
import narf
import numpy as np
import pickle
import blosc
import os
#print(h._storage_type)

os.environ["BLOSC_NTHREADS"] = str(os.cpu_count())


with h5py.File("test.hdf5", "w") as f:
    a = hist.axis.Regular(1000, 0., 1.)

    h = hist.Hist(a,a,a, storage = hist.storage.Weight(), name = "test")
    #h = np.ones(shape=(1002, 1002, 1002, 2), dtype=np.float64)

    #view = h.view(flow=True)
    #view[...] = np.random.random(view.shape)

    #h.values()[...] = 1.

    #print(h.view(flow=True).dtype.itemsize)

    #assert(0)

    #print(h.__getstate__())
    #print(h.__dict__)
    #print("filling")
    #view = h.view(flow=True)
    #vals = h.values(flow=True)
    #vals[...] = np.random.random(vals.shape)
    #del vals
    #h.variances(flow=True)[...] = np.random.random(view.shape)

    #narf.hist_to_h5py(h, f, compression=None, compression_opts=None)
    #narf.hist_to_h5py(h, f)

    buffers = []
    p = pickle.dumps(h, protocol = pickle.HIGHEST_PROTOCOL, buffer_callback = buffers.append)
    print(p)
    print(buffers)
    arr = np.frombuffer(buffers[0], dtype=np.dtype("b"))
    #arr = np.void(buffers[0])

    dset = f.create_dataset("byte_arr_test", data = arr, **hdf5plugin.Blosc(cname="lz4"), chunks=True)
    dset.attrs["pickle"] = np.frombuffer(p, dtype=np.dtype("b"))

    del h
    del p
    del arr
    del buffers

with h5py.File("test.hdf5") as f:
    dset = f["byte_arr_test"]
    p = dset.attrs["pickle"].tobytes()

    buffers = [bytearray(dset.size)]

    arr = np.frombuffer(buffers[0], dtype=np.dtype("b"))
    dset.read_direct(arr)

    orig = pickle.loads(p, buffers = buffers)

    print(type(orig))
    print(orig)



    #f.create_dataset("test_scalar", data = 5.0)


    #testdict = { "a" : 0, "b" : 1 }

    #testdictarr = np.array(pickle.dumps(testdict))
    #print("testdictarr", type(testdictarr), testdictarr.shape, testdictarr.dtype)

    #testdictdset = f.create_dataset("testdict", data = np.void(blosc.compress(pickle.dumps(testdict), cname = "lz4") ))
    #print("testdict chunks", testdictdset.chunks)

#input("wait")

#with h5py.File("test.hdf5") as f:
    #hist_read = narf.h5py_to_hist(f["test"])

    #test_scalar = f["test_scalar"]
    ##print(test_scalar[...])
    ##print(f["test_scalar"])
    ##print(hist_read.values())
    ##print(hist_read)

    #testdict = pickle.loads(blosc.decompress(f["testdict"][()].tobytes()))
    #print(testdict)
