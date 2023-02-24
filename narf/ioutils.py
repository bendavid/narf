import pickle
import hdf5plugin
import h5py
import boost_histogram as bh
import hist
import copy
import numpy as np
import uuid
import copyreg
import io

MIN_PROTOCOL_VERSION = 1
CURRENT_PROTOCOL_VERSION = 1

class H5PickleProxy:
    def __init__(self, obj, h5group = None):
        self.obj = obj
        self.h5group = h5group

    def get(self):
        if self.obj is None:
            if self.h5group is None:
                raise ValueError("Trying to read object from H5PickleProxy but neither object nor underlying hdf5 storage is available.")
            elif not self.h5group:
                raise ValueError("Trying to read object from H5PickleProxy but the underlying file has been closed.")

            self.obj = pickle_load_h5py(self.h5group)

        return self.obj

    def release():
        self.obj = None

class H5Buffer():
    def __init__(self, buf, h5dset = None, readonly = None):
        self.buf = buf
        self.h5dset = h5dset
        self.readonly = readonly

        if self.buf is not None:
            self.readonly = memoryview(self.buf).readonly

    def checkdset(self):
        if self.h5dset is None:
            raise ValueError("Trying to read from H5Buffer but neither buffer nor dset are available.")
        elif not self.h5dset:
            raise ValueError("Trying to read from H5Buffer but underlying file has been closed.")

    def getbuffer(self):
        if self.buf is None:
            self.checkdset()
            bufarr = np.empty_like(self.h5dset)
            if self.h5dset.size:
                self.h5dset.read_direct(bufarr)
            bufout = bufarr.data
            if self.readonly:
                bufout = bufout.toreadonly()
            self.buf = bufout

        return self.buf

    def readinto(self, b, /):
        bufarr = np.asarray(b)
        if self.buf is None:
            self.checkdset()
            if self.h5dset.size:
                self.h5dset.read_direct(bufarr)
            else:
                bufarr[...] = self.h5dset[...]
        else:
            bufarr[...] = self.buf

    def __reduce_ex__(self, protocol):
        if protocol >= 5:
            outbuf = pickle.PickleBuffer(outbuf)
        else:
            if memoryview(self.buf).readonly:
                outbuf = bytes(self.buf)
            else:
                outbuf = bytearray(self.buf)

        return (type(self), (outbuf,))

class H5Path():
    def __init__(self, path):
        self.path = path

# workaround for suboptimal pickling of bh.Histogram which doesn't properly use the out-of-band-mechanism
def get_histogram_view(h):
    return pickle.PickleBuffer(h.view(flow=True)).raw()

def make_Histogram(axes, storage, metadata, h5buf):
    hist_read = bh.Histogram(*axes, storage=storage, metadata=metadata)
    hist_read_view = get_histogram_view(hist_read)
    h5buf.readinto(hist_read_view)
    return hist_read

def reduce_Histogram(obj):
    axes = tuple(obj.axes)
    view = get_histogram_view(obj)
    h5buf = H5Buffer(view)

    return (make_Histogram, (axes, obj.storage_type(), obj.metadata, h5buf) )

def make_Hist(axes, storage, metadata, label, name, h5buf):
    hist_read = hist.Hist(*axes, storage=storage, metadata=metadata, label=label, name=name)
    hist_read_view = get_histogram_view(hist_read)
    h5buf.readinto(hist_read_view)
    return hist_read

def reduce_Hist(obj):
    axes = tuple(obj.axes)
    view = get_histogram_view(obj)
    h5buf = H5Buffer(view)

    return (make_Hist, (axes, obj.storage_type(), obj.metadata, obj.label, obj.name, h5buf) )

class H5IO():
    def __init__(self, name, h5out, mode = "r"):
        if mode == "r":
            self.dset = h5out[name]
        elif mode == "w":
            self.dset = h5out.create_dataset(name, shape=(0,), dtype = np.ubyte, maxshape=(None,), chunks = (1024*1024,), **hdf5plugin.Blosc(cname="lz4"))
        else:
            raise ValueError("Unsupported mode.")
        self.pos = 0

    def write(self, b, /):
        if type(b) is bytes:
            flat_view = memoryview(b)
        else:
            flat_view = pickle.PickleBuffer(b).raw()

        oldsize = self.dset.size
        nbytes = flat_view.nbytes
        size = oldsize + nbytes
        self.dset.resize((size,))
        self.dset[oldsize:] = flat_view
        return nbytes

    def read(self, size=-1, /):
        oldpos = self.pos
        self.pos += size
        return self.dset[oldpos:self.pos].data

    def readline(self, size=-1, /):
        raise NotImplementedError("readline is not implemented")

    def readinto(self, b, /):
        arr = np.frombuffer(b, dtype=np.ubyte)
        nbytes = arr.nbytes
        oldpos = self.pos
        self.pos += nbytes
        self.dset.read_direct(arr, source_sel = np.s_[oldpos:self.pos])
        return nbytes

class H5Pickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[bh.Histogram] = reduce_Histogram
    dispatch_table[hist.hist.Hist] = reduce_Hist

    def __init__(self, h5group, protocol = 5):
        self.h5group = h5group
        self.io = H5IO("pickle_data", h5group, mode = "w")
        super().__init__(self.io, protocol=protocol)

    def reduce_H5PickleProxy(self, proxy):
        PROXIED_OBJECTS_NAME = "proxied_objects"
        current_location = self.h5group

        if PROXIED_OBJECTS_NAME in current_location:
            proxied_objects = current_location[PROXIED_OBJECTS_NAME]
        else:
            proxied_objects = current_location.create_group(PROXIED_OBJECTS_NAME, track_order = True)

        iproxy = len(proxied_objects)
        proxied_group = pickle_dump_h5py(f"proxied_object_{iproxy}", proxy.obj, proxied_objects)

        return (type(proxy), (None, proxied_group))

    def reduce_H5Buffer(self, h5buf):
        H5BUFFER_GROUP_NAME = "h5_buffers"
        current_location = self.h5group

        if H5BUFFER_GROUP_NAME in current_location:
            h5_buffers = current_location[H5BUFFER_GROUP_NAME]
        else:
            h5_buffers = current_location.create_group(H5BUFFER_GROUP_NAME, track_order = True)

        ibuf = len(h5_buffers)
        flat_view = pickle.PickleBuffer(h5buf.getbuffer()).raw()
        bufdset = h5_buffers.create_dataset(f"h5_buffer_{ibuf}", data = flat_view, chunks=True, **hdf5plugin.Blosc(cname="lz4"))

        return (type(h5buf), (None, bufdset, flat_view.readonly))

    def reducer_override(self, obj):
        if type(obj) is H5PickleProxy:
            return self.reduce_H5PickleProxy(obj)
        elif type(obj) is H5Buffer:
            return self.reduce_H5Buffer(obj)
        else:
            return NotImplemented

    def persistent_id(self, obj):
        if isinstance(obj, h5py.Group) or isinstance(obj, h5py.Dataset):
            return H5Path(obj.name)
        else:
            return None

class H5Unpickler(pickle.Unpickler):
    def __init__(self, h5group):
        self.h5group = h5group
        self.io = H5IO("pickle_data", h5group, mode = "r")
        super().__init__(self.io)

    def persistent_load(self, pid):
        if type(pid) is H5Path:
            return self.h5group.file[pid.path]
        else:
            raise pickle.UnpicklingError("unsupported persistent object")

def pickle_dump_h5py(name, obj, h5out):
    obj_group = h5out.create_group(name)
    try:
        obj_group.attrs["narf_h5py_pickle_protocol_version"] = CURRENT_PROTOCOL_VERSION
        pickler = H5Pickler(obj_group)
        pickler.dump(obj)
    except:
        del obj_group
        raise
    return obj_group

def pickle_load_h5py(h5group):
    try:
        narf_protocol_version = h5group.attrs["narf_h5py_pickle_protocol_version"]
    except KeyError:
        raise ValueError("h5py dataset does not contain a python object pickled by narf.")

    if narf_protocol_version < MIN_PROTOCOL_VERSION:
        raise ValueError(f"Unsuppported narf protocol version {narf_protocol_version}, minimum supported version is {MIN_PROTOCOL_VERSION}")
    elif narf_protocol_version > CURRENT_PROTOCOL_VERSION:
        raise ValueError(f"Unsuppported narf protocol version {narf_protocol_version}, maximum supported version is {current_protocol_version}")

    unpickler = H5Unpickler(h5group)
    return unpickler.load()
