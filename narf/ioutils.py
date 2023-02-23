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
    def __init__(self, buf, h5dset = None):
        self.buf = buf
        self.h5dset = h5dset

    def checkdset(self):
        if self.h5dset is None:
            raise ValueError("Trying to read from H5Buffer but neither buffer nor dset are available.")
        elif not self.h5dset:
            raise ValueError("Trying to read from H5Buffer but underlying file has been closed.")

    def getbuffer(self):
        if self.buf is None:
            self.checkdset()
            readonly = self.h5dset.attrs["readonly"]
            bufarr = np.empty_like(self.h5dset)
            if self.h5dset.size:
                self.h5dset.read_direct(bufarr)
            bufout = bufarr.data
            if readonly:
                bufout = bufout.toreadonly()
            self.buf = bufout

        return self.buf

    def readinto(self, buf):
        bufarr = np.asarray(buf)
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

class H5Pickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[bh.Histogram] = reduce_Histogram
    dispatch_table[hist.hist.Hist] = reduce_Hist

    def __init__(self, protocol = 5):
        self.current_group = None
        self.io = io.BytesIO()
        self.pickle_buffers = []
        super().__init__(self.io, protocol=protocol, buffer_callback=self.pickle_buffers.append)

    def dump(self, name, obj, h5group):
        obj_group = h5group.create_group(name, track_order = True)
        self.current_group = obj_group

        try:
            obj_group.attrs["narf_h5py_pickle_protocol_version"] = CURRENT_PROTOCOL_VERSION

            self.io.truncate(0)
            self.io.seek(0)
            self.pickle_buffers.clear()

            super().dump(obj)

            dset = obj_group.create_dataset("pickle_data", data = self.io.getbuffer(), chunks = True, **hdf5plugin.Blosc(cname="lz4"), track_order = True)

            bufgroup = obj_group.create_group("pickle_buffers", track_order = True)
            for ibuf, buf in enumerate(self.pickle_buffers):
                bufdset = bufgroup.create_dataset(f"pickle_buffer_{ibuf}", data = buf.raw(), chunks=True, **hdf5plugin.Blosc(cname="lz4"), track_order = True)
                bufdset.attrs["readonly"] = buf.raw().readonly
        except:
            del obj_group
            raise
        finally:
            self.current_group = None
            self.io.truncate(0)
            self.io.seek(0)
            self.pickle_buffers.clear()

        return obj_group

    def reduce_H5PickleProxy(self, proxy):
        PROXIED_OBJECTS_NAME = "proxied_objects"
        current_location = self.current_group

        if PROXIED_OBJECTS_NAME in current_location:
            proxied_objects = current_location[PROXIED_OBJECTS_NAME]
        else:
            proxied_objects = current_location.create_group(PROXIED_OBJECTS_NAME, track_order = True)

        iproxy = len(proxied_objects)
        proxied_group = pickle_dump_h5py(f"proxied_object_{iproxy}", proxy.obj, proxied_objects)

        return (type(proxy), (None, proxied_group,))

    def reduce_H5Buffer(self, h5buf):
        H5BUFFER_GROUP_NAME = "h5_buffers"
        current_location = self.current_group

        if H5BUFFER_GROUP_NAME in current_location:
            h5_buffers = current_location[H5BUFFER_GROUP_NAME]
        else:
            h5_buffers = current_location.create_group(H5BUFFER_GROUP_NAME, track_order = True)

        ibuf = len(h5_buffers)
        buf = pickle.PickleBuffer(h5buf.getbuffer())
        bufdset = h5_buffers.create_dataset(f"h5_buffer_{ibuf}", data = buf.raw(), chunks=True, **hdf5plugin.Blosc(cname="lz4"), track_order = True)
        bufdset.attrs["readonly"] = buf.raw().readonly

        return (type(h5buf), (None, bufdset,))

    def persistent_id(self, obj):
        if isinstance(obj, h5py.Group) or isinstance(obj, h5py.Dataset):
            return H5Path(obj.name)
        else:
            return None

    def reducer_override(self, obj):
        if type(obj) is H5PickleProxy:
            return self.reduce_H5PickleProxy(obj)
        elif type(obj) is H5Buffer:
            return self.reduce_H5Buffer(obj)
        else:
            return NotImplemented

class H5Unpickler(pickle.Unpickler):
    def __init__(self):
        self.current_group = None
        self.io = io.BytesIO()
        self.pickle_buffers = []
        super().__init__(self.io, buffers = self.pickle_buffers)
    def load(self, h5group):
        try:
            narf_protocol_version = h5group.attrs["narf_h5py_pickle_protocol_version"]
        except KeyError:
            raise ValueError("h5py dataset does not contain a python object pickled by narf.")

        if narf_protocol_version < MIN_PROTOCOL_VERSION:
            raise ValueError(f"Unsuppported narf protocol version {narf_protocol_version}, minimum supported version is {MIN_PROTOCOL_VERSION}")
        elif narf_protocol_version > CURRENT_PROTOCOL_VERSION:
            raise ValueError(f"Unsuppported narf protocol version {narf_protocol_version}, maximum supported version is {current_protocol_version}")

        self.current_group = h5group

        try:
            dset = h5group["pickle_data"]

            self.io.truncate(0)
            self.io.seek(0)
            self.pickle_buffers.clear()

            arr = np.empty_like(dset)
            if dset.size:
                dset.read_direct(arr)

            self.io.write(arr)
            self.io.seek(0)

            bufgroup = h5group["pickle_buffers"]
            for bufdset in bufgroup.values():
                readonly = bufdset.attrs["readonly"]
                bufarr = np.empty_like(bufdset)
                if bufdset.size:
                    bufdset.read_direct(bufarr)
                bufout = bufarr.data
                if readonly:
                    bufout = bufout.toreadonly()
                self.pickle_buffers.append(bufout)

            obj = super().load()

        finally:
            self.current_group = None
            self.io.truncate(0)
            self.io.seek(0)
            self.pickle_buffers.clear()

        return obj

    def persistent_load(self, pid):
        if isinstance(pid, H5Path):
            return self.current_group.file[pid.path]
        else:
            raise pickle.UnpicklingError("unsupported persistent object")

def pickle_dump_h5py(name, obj, h5out):
    pickler = H5Pickler()
    return pickler.dump(name, obj, h5out)

def pickle_load_h5py(h5group):
    unpickler = H5Unpickler()
    return unpickler.load(h5group)
