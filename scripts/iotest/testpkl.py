import hist
import hdf5plugin
import h5py
import narf
import lz4.frame
import pickle
import numpy as np

#print(h._storage_type)

fname = "test.pkl"

#with lz4.frame.open(fname, "wb") as f:
with open(fname, "wb") as f:
    a = hist.axis.Regular(1000, 0., 1.)

    print("making hist")
    #h = hist.Hist(a,a,a, storage = hist.storage.Weight(), name = "test")
    h = np.ones(shape=(1002, 1002, 1002, 2), dtype=np.float64)

    buffers = []



    #h = hist.Hist(a,a,a, storage = hist.storage.Double(), name = "test")

    #print("filling")
    #view = h.view(flow=True)
    #h.values(flow=True)[...] = np.random.random(view.shape)
    #h.variances(flow=True)[...] = np.random.random(view.shape)

    #narf.hist_to_h5py(h, f, compression=None, compression_opts=None)
    #narf.hist_to_h5py(h, f)
    print("writing")

    testdict = { "a" : 0, "b" : 1 }

    #pickle.dump(h, f, protocol = pickle.HIGHEST_PROTOCOL)
    #pickle.dump(h, f)
    p = pickle.dumps(h, protocol = pickle.HIGHEST_PROTOCOL, buffer_callback = buffers.append)
    print(p)
    print(buffers)
    arr = np.frombuffer(buffers[0], dtype=np.dtype("b"))

    #f.create_dataset("byte_arr_test", data = arr, compression = "lz4")

    print(arr.shape)

    print("deleting")
    del h


#with lz4.frame.open(fname, "rb") as f:
#with open(fname, "rb") as f:
    #print("reading")
    #hist_read = pickle.load(f)
    #print('done')
    ###print(hist_read)
