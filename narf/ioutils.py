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
                proxied_objects = current_location.create_group(proxied_objects_name)


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
    bufview, source_sel = state.pop("bufview")

    obj.__init__(*axes, storage = storage_type())

    for key, value in state.items():
        setattr(obj, key, value)

    view = obj.view(flow=True).ravel(order="A").view(dtype=np.ubyte)
    bufview.read_direct(view, source_sel = source_sel)


def pickle_dump_h5py(name, obj, h5out):

    original_location = current_state["current_location"]
    current_state["current_location"] = h5out

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

    total_size = len(outbytes)
    for buf in bufs:
        total_size += len(buf.raw())

    chunksize = 16*1024*1024
    chunksize = min(chunksize, total_size)
    chunks = (chunksize,)

    dset = h5out.create_dataset(name, (total_size,), dtype = np.ubyte, chunks = chunks, **hdf5plugin.Blosc(cname="lz4"))

    outarr = np.frombuffer(outbytes, dtype = np.ubyte)
    dset.write_direct(outarr, dest_sel = np.s_[:len(outbytes)])

    bufs_do_h5py_direct_read = []
    buf_offsets = []
    offset = len(outbytes)
    for buf in bufs:
        buf_offsets.append(offset)
        size = len(buf.raw())
        arr = np.frombuffer(buf, dtype = np.ubyte)
        dset.write_direct(arr, dest_sel = np.s_[offset:offset+size])

        if buf in bufs_for_h5py_direct_read:
            bufs_do_h5py_direct_read.append(True)
            bufs_for_h5py_direct_read.remove(buf)
        else:
            bufs_do_h5py_direct_read.append(False)

        offset += size

    dset.attrs["bufs_do_h5py_direct_read"] = bufs_do_h5py_direct_read
    dset.attrs["buf_offsets"] = buf_offsets
    dset.attrs["narf_h5py_pickle_protocol_version"] = current_protocol_version

    return dset

def pickle_load_h5py(dset):
    try:
        narf_protocol_version = dset.attrs["narf_h5py_pickle_protocol_version"]
    except KeyError:
        raise ValueError("h5py dataset does not contain a python object pickled by narf.")

    if narf_protocol_version > current_protocol_version:
        raise ValueError(f"Unsuppported narf protocol version {narf_protocol_version}, maximum supported version is {current_protocol_version}")

    bufs_do_h5py_direct_read = dset.attrs["bufs_do_h5py_direct_read"]
    buf_offsets = dset.attrs["buf_offsets"]

    if len(buf_offsets) > 0:
        size = buf_offsets[0]
        end_offsets = list(buf_offsets[1:])
        end_offsets.append(dset.shape[0])
    else:
        size = dset.shape[0]
        end_offsets = []

    inbytes = np.empty((size,), dtype = dset.dtype)
    dset.read_direct(inbytes, source_sel = np.s_[:size])

    bufs = []
    for offset, end_offset, do_h5py_direct_read in zip(buf_offsets, end_offsets, bufs_do_h5py_direct_read):
        if do_h5py_direct_read:
            # pass the h5py dataset directly to __setstate__ so that it can handle the reading itself
            # this is needed because boost histograms can't currently adopt the buffer memory for their
            # internal storage
            bufs.append((dset, np.s_[offset:end_offset]))
        else:
            buf_size = end_offset - offset
            bufarr = np.empty((buf_size,), dtype = dset.dtype)
            dset.read_direct(bufarr, source_sel = np.s_[offset:end_offset])
            bufs.append(bufarr)

    original_location = current_state["current_location"]
    current_state["current_location"] = dset.parent

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

def write_results(results, h5out):
    for dataset, dsetresult in results.items():
        dsetgroup = h5out.create_group(dataset)

        for k, v in dsetresult.items():
            if k == "output":
                # special handling for output: write each object separately so that they can
                # be read separately later to save time and memory
                outputgroup = dsetgroup.create_group(k)
                for kout, vout in v.items():
                    pickle_dump_h5py(kout, vout, outputgroup)

            else:
                pickle_dump_h5py(k, v, dsetgroup)
