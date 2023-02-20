import pickle
import hdf5plugin
import h5py
import boost_histogram as bh
import copy
import numpy as np
import uuid

# TODO add locks and/or thread local instances for this to make it thread-safe if needed in the future
bufs_for_h5py_direct_read = set()
# current_location = None

current_state = { "current_location" : None }

current_protocol_version = 0

class H5PickleProxy:
    def __init__(self, obj):
        self.obj = obj
        self.proxied_dset = None

    def GetObject(self):
        if self.obj is None:
            if not self.proxied_dset:
                raise ValueError("Trying to read object from H5PickleProxy but the underlying file has been closed.")

            self.obj = pickle_load_h5py(self.proxied_dset)

        return self.obj

    def __getstate__(self):
        self.GetObject()

        current_location = current_state["current_location"]

        if current_location is None:
            # not writing to hdf5 file, so just serialize the object directly
            state = { "obj" : self.obj,
                     "proxied_path" : None }

        else:
            # writing to hdf5 so  write the object separately
            # and serialize only the dataset path inline

            proxied_objects_name = "proxied_objects"

            if proxied_objects_name in current_location:
                proxied_objects = current_location[proxied_objects_name]
            else:
                proxied_objects = current_location.create_group(proxied_objects_name, track_order = True)


            proxied_dset = pickle_dump_h5py(str(uuid.uuid1()), self.obj, proxied_objects)
            proxied_path = proxied_dset.name

            state = { "obj" : None,
                     "proxied_path" : proxied_path }

        return state

    def __setstate__(self, state):
        current_location = current_state["current_location"]

        self.obj = state["obj"]

        proxied_path = state["proxied_path"]
        if proxied_path is not None:
            self.proxied_dset = current_location.file[proxied_path]


def hist_getstate(obj):
    local_dict = copy.copy(obj.__dict__)
    local_dict["axes"] = tuple(obj.axes)
    local_dict["storage_type"] = obj.storage_type

    view = obj.view(flow=True).ravel(order="A").view(dtype=np.ubyte)
    bufview = pickle.PickleBuffer(view)

    bufs_for_h5py_direct_read.add(bufview)

    local_dict["bufview"] = bufview

    return local_dict

def hist_setstate(obj, state):
    axes = state.pop("axes")
    storage_type = state.pop("storage_type")
    bufview = state.pop("bufview")

    obj.__init__(*axes, storage = storage_type())

    for key, value in state.items():
        setattr(obj, key, value)

    view = obj.view(flow=True).ravel(order="A").view(dtype=np.ubyte)
    bufview.read_direct(view)


def pickle_dump_h5py(name, obj, h5out):

    group = h5out.create_group(name, track_order = True)

    group.attrs["narf_h5py_pickle_protocol_version"] = current_protocol_version


    original_location = current_state["current_location"]
    current_state["current_location"] = group

    # dirty hack to work around non-optimized pickling of boost histograms
    getstate_orig = bh.Histogram.__getstate__
    bh.Histogram.__getstate__ = hist_getstate

    try:
        bufs = []
        outbytes = pickle.dumps(obj, protocol = pickle.HIGHEST_PROTOCOL, buffer_callback = bufs.append)
    except:
        raise
    finally:
        current_state["current_location"] = original_location
        bh.Histogram.__getstate__ = getstate_orig

    chunksize = 16*1024*1024

    arr = np.frombuffer(outbytes, dtype = np.ubyte)
    chunks = (min(chunksize, arr.shape[0]),)

    dset = group.create_dataset("pickle_data", shape = arr.shape, dtype = arr.dtype, chunks = chunks, **hdf5plugin.Blosc(cname="lz4"), track_order = True)

    dset.write_direct(arr)


    bufgroup = group.create_group("pickle_buffers", track_order = True)
    for ibuf, buf in enumerate(bufs):
        bufarr = np.frombuffer(buf, dtype = np.ubyte)
        bufchunks = (min(chunksize, bufarr.shape[0]),)

        bufdset = bufgroup.create_dataset(f"buffer_{ibuf}", shape = bufarr.shape, dtype = arr.dtype, chunks=bufchunks, **hdf5plugin.Blosc(cname="lz4"), track_order = True)

        bufdset.write_direct(bufarr)

        if buf in bufs_for_h5py_direct_read:
            bufdset.attrs["do_h5py_direct_read"] = True
            bufs_for_h5py_direct_read.remove(buf)
        else:
            bufdset.attrs["do_h5py_direct_read"] = False

    return group

def pickle_load_h5py(group):
    try:
        narf_protocol_version = group.attrs["narf_h5py_pickle_protocol_version"]
    except KeyError:
        raise ValueError("h5py dataset does not contain a python object pickled by narf.")

    if narf_protocol_version > current_protocol_version:
        raise ValueError(f"Unsuppported narf protocol version {narf_protocol_version}, maximum supported version is {current_protocol_version}")

    dset = group["pickle_data"]

    inbytes = np.empty(dset.size, dtype = dset.dtype)
    dset.read_direct(inbytes)

    bufgroup = group["pickle_buffers"]

    bufs = []
    for bufdset in bufgroup.values():
        do_h5py_direct_read = bufdset.attrs["do_h5py_direct_read"]
        if do_h5py_direct_read:
            bufs.append(bufdset)
        else:
            bufarr = np.empty(bufdset.shape, dtype = bufdset.dtype)
            bufdset.read_direct(bufarr)
            bufs.append(bufarr)

    original_location = current_state["current_location"]
    current_state["current_location"] = group

    # dirty hack to work around non-optimized pickling of boost histograms
    setstate_orig = bh.Histogram.__setstate__
    bh.Histogram.__setstate__ = hist_setstate

    try:
        obj = pickle.loads(inbytes, buffers = bufs)
    except:
        raise
    finally:
        current_state["current_location"] = original_location
        bh.Histogram.__setstate__ = setstate_orig

    return obj
