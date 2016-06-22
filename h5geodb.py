"""
Hdf5 store for spatiotemporal data
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
# Consider using https://github.com/meteotest/h5pySWMR instead of h5py to
# enable concurrent read/write access
# import h5pyswmr as h5py
import h5py
from sklearn.neighbors import KDTree
# note that we don't use scipy.spatial.cKDTree() because it seemed to leak
# memory and it does not support pickling. It also has less options.


class GeoDB():
    """
    Hdf5-based data store.

    Imposes the following structure on hdf5 files:

    filename.h5
    │
    ├── coords_wsg84
    │   ├── lat
    │   ├── lon
    │  ...
    ├── data
    │    ├── temperature
    │    │   ├── 20140225
    │    │   ├── 20140226
    │    │   ├── 20140227
    │    │  ...
    │    ├── windspeed
    │    │   ├── 20140225
    │    │   ├── 20140226
    │    │   ├── 20140227
    │    │  ...

    Data for a given day is stored in a distinguished 3D dataset. For example,
    if data has a frequency of 30 minutes, the shape of a daily dataset is
    48 x M x N.

    Parts of a (daily) dataset can be uninitialized.
    To identify uninitialized sub-arrays, every dataset has an attribute
    ``bitmask`` that consists of a a bit-array.
    For example, for a data frequency of 60 minutes, a dataset's ``bitmask``
    attribute may look as follows:
    111111111111100000000000
    This means that the dataset contains arrays for 0:00, 1:00, ..., 12:00,
    but lacks data from 13:00 to 23:00.
    """

    # names of hdf5 groups where coordinate system and data are stored
    GRP_GRIDS = '/coords_wsg84'
    GRP_DATA = '/data'
    # name of the bitmask attribute (see above for details)
    ATTR_BITMASK = "bitmask"
    # daily datasets naming schema (YYYYMMDD)
    DATE_FMT = "%Y%m%d"

    def __init__(self, h5file):
        """
        Initializes an existing hdf5 file

        Args:
            h5file: name of the hdf5 file

        Raises:
            IOError if file does not exist
        """
        self.h5file = h5file

        # read data shape and frequency from file attributes
        with h5py.File(self.h5file, 'r') as f:
            # convert numpy datatypes to Python datatypes
            self.shape = tuple(f.attrs['shape'])
            self.frequency = int(f.attrs['frequency'])

        # reference to k-dimensional search tree for nearest neighbor queries
        self.__kdtree = None

    @classmethod
    def create(cls, h5file, frequency, lat, lon):
        """
        Create a GeoDB.

        Args:
            h5file: name of the hdf5 file. Existing files are overwritten.
            frequency: data frequency in minutes
            lat: 2D array containing latitude grid. Grid must be oriented
                "top-down / left-right".
            lon: 2D array containing longitude grid. Grid must be oriented
                "top-down / left-right".

        Returns:
            GeoDB object
        """
        with h5py.File(h5file, 'w') as f:
            f.attrs['shape'] = lat.shape
            f.attrs['frequency'] = frequency
            # create groups for grids and data
            f.require_group(cls.GRP_DATA)
            grp_grids = f.require_group(cls.GRP_GRIDS)

            # create lat/lon datasets
            grp_grids.create_dataset(name='lat', data=lat, dtype=np.float64)
            grp_grids.create_dataset(name='lon', data=lon, dtype=np.float64)

        db = GeoDB(h5file)

        return db

    def store(self, param, data, time):
        """
        Store an array for given parameter name and time.

        Args:
            param: parameter identifier (string)
            data: numpy array
            time: datetime object

        Raises:
            ValueError if data.shape is incompatible with lat/lon grid
            or if time is incompatible with file's frequency.
            For example, if file has a frequency of 10 minutes, then a
            time of 10:11 raises an error.
        """
        # check if data has same shape as lat/lon grid
        if data.shape[0:2] != self.shape[0:2]:
            raise ValueError("data shape incompatible with file. Expected "
                             "first two dimensions: {0}. Got: {1}."
                             .format(self.shape[0:2], data.shape[0:2]))

        date_str = time.strftime(self.DATE_FMT)
        dst_path = os.path.join(self.GRP_DATA, param, date_str)
        steps_per_day = 1440 // self.frequency
        shape = (steps_per_day, ) + data.shape
        with h5py.File(self.h5file, 'r+') as f:
            # create dataset if it does not yet exist
            dst = f.require_dataset(name=dst_path, shape=shape,
                                    exact=True, dtype=data.dtype)
            # write data to disk
            idx = self._datetime_to_index(time)
            dst[idx] = data
            # set idx-th bit of bitmask to 1
            # TODO in a concurrent environment, this must be wrapped in a lock
            # to avoid race conditions. To keep things simple, locking has been
            # removed for this tutorial.
            if self.ATTR_BITMASK not in dst.attrs:
                # initialize bitmask to 000... (all time steps uninitialized)
                dst.attrs[self.ATTR_BITMASK] = np.zeros(shape[0],
                                                        dtype=np.uint8)
            # read and update bitmask
            bitmask = dst.attrs[self.ATTR_BITMASK]
            bitmask[idx] = 1
            # write back to file
            dst.attrs[self.ATTR_BITMASK] = bitmask

    def data(self, param, time,
             minlat=None, maxlat=None,
             minlon=None, maxlon=None):
        """
        Query array data.

        Args:
            param: parameter identifier (string)
            time: datetime object
            minlat: southern bound
            maxlat: northern bound
            minlon: western bound
            maxlon: eastern bound

        Returns:
            Full data array if no bounds are given.
            Otherwise returns a tuple (lat, lon, data), where
            ``lat``/``lon`` and ``data`` are the coordinates and data of
            the requested region, respectively.

        Raises:
            Exception if data is missing for given time and/or param.
        """
        with h5py.File(self.h5file, 'r') as f:
            try:
                idx = self._datetime_to_index(time)
            except ValueError:
                raise Exception('no data exists for {}'.format(time))
            path_param = os.path.join(self.GRP_DATA, param)
            if path_param not in f:
                raise Exception("unknown param_id")
            dst_path = os.path.join(path_param, time.strftime(self.DATE_FMT))
            try:
                dst = f[dst_path]
            except KeyError:
                raise Exception('no data exists for {}'.format(time))

            # check bitmask to determine if dataset is uninitialized
            if dst.attrs[self.ATTR_BITMASK][idx] == 0:
                raise Exception('no data exists for {}'.format(time))
            if None in (minlat, maxlat, minlon, maxlon):
                # return whole 2d array
                return dst[idx]
            else:
                lat, lon = self.latlon
                slice_ax1, slice_ax2 = extract_subgrid(x=lon, y=lat,
                                                       low_x=minlon,
                                                       high_x=maxlon,
                                                       low_y=minlat,
                                                       high_y=maxlat)
                lon = lon[slice_ax1, slice_ax2]
                lat = lat[slice_ax1, slice_ax2]
                data = dst[idx, slice_ax1, slice_ax2]

                return (lat, lon, data)

    @property
    def latlon(self):
        """
        Returns a tuple (lat, lon) with 2d arrays
        """
        with h5py.File(self.h5file, 'r') as f:
            path_lat = os.path.join(self.GRP_GRIDS, 'lat')
            path_lon = os.path.join(self.GRP_GRIDS, 'lon')
            lat, lon = f[path_lat][:], f[path_lon][:]

        return lat, lon

    def _datetime_to_index(self, t):
        """
        Returns the first dimension (time) index corresponding to
        given datetime t.

        Raises:
            ValueError if ``t`` is out of range.
        """
        minutes = t.hour * 60 + t.minute
        if minutes % self.frequency != 0:
            raise ValueError("datetime incompatible with file's frequency "
                             "({0} Min.)".format(self.frequency))
        idx = minutes // self.frequency

        return idx

    def point_data(self, param_id, lat0, lon0, start, end):
        """
        Return a timeseries for given parameter and coordinates.

        Args:
            param_id: parameter identifier (string)
            lat0: latitude of point
            lon0: longitude of point
            start: datetime object
            end: datetime object

        Returns:
            a tuple (lat, lon, ts), where lat/lon are the coordinates of the
            nearest grid point and ts is a Pandas Series:

            2014-07-16 14:00    20.8
            2014-07-16 15:00    24.2
            ...

        Raises:
            Exception if invalid parameter is passed
        """
        f = h5py.File(self.h5file, 'r')

        if os.path.join(self.GRP_DATA, param_id) not in f:
            raise Exception('unknown param_id')

        idx_row, idx_col, lat, lon = self.kdtree.nearest(lat0, lon0)

        # get data for whole days and crop in the end
        # this could obviously be improved...
        date_start = start.date()
        no_days = abs(date_start - end.date()).days + 1
        dates = [date_start + timedelta(days=d) for d in range(no_days)]
        steps = (24 * 60) // self.frequency  # no. of steps per day
        # we create an empty masked 1D array and mask positions for which
        # data is uninitialized
        data = np.ma.empty(no_days * steps)
        for i, date_ in enumerate(dates):
            dst_path = os.path.join(self.GRP_DATA, param_id,
                                    date_.strftime(self.DATE_FMT))
            try:
                dst = f[dst_path]
            except KeyError:
                # no data exists for this day => mask whole day
                data[i*steps:(i+1)*steps] = np.ma.array(np.empty(steps),
                                                        mask=np.ones(steps))
                continue
            data_daily = dst[:, idx_row, idx_col]
            # mask array positions for which data is uninitialized
            # (i.e., attribute bitmask is 0 at this position)
            bitmask = np.array([bit ^ True for bit in
                                dst.attrs[self.ATTR_BITMASK]], dtype=np.bool)
            data[i*steps:(i+1)*steps] = np.ma.array(data_daily, mask=bitmask)

        f.close()

        # crop data to interval [start_time end_time] and drop masked values
        timerange = pd.date_range(dates[0], periods=data.shape[0],
                                  freq='{0}Min'.format(self.frequency))
        ts = pd.Series(index=timerange, data=data)
        ts.dropna(inplace=True)
        ts = ts[start:end]

        return (lat, lon, ts)

    @property
    def kdtree(self):
        """
        Returns a k-d tree for nearest neighbor search.
        """
        if self.__kdtree is None:
            lat, lon = self.latlon
            self.__kdtree = KDTreeTunnelDist(lat, lon)

        return self.__kdtree

        # once you're sure that the shape of a geoDB does not change, you may
        # want to pickle your k-d-tree. Like so:
        # datadir, h5filename = os.path.split(self.h5file)
        # pkl_file = os.path.join(datadir, "{}.pkl".format(h5filename))
        # try:
        #     with open(pkl_file, "rb") as f:
        #         self.__kdtree = pickle.load(f)
        # except FileNotFoundError:
        #     # create kd-tree and pickle it
        #     lon, lat = self.latlon
        #     self.__kdtree = KDTreeTunnelDist(lat, lon)
        #     # rename to final name only after writing is done! Note that
        #     # temp. file must be on same filesystem otherwise os.rename()
        #     # might not work
        #     with NamedTemporaryFile(mode="w+b", dir=datadir, delete=False) as f:
        #         pickle.dump(self.__kdtree, f)
        #         temp_filename = f.name
        #     os.rename(temp_filename, pkl_file)  # os.rename() is atomic


class KDTreeTunnelDist:
    """
    Geospatial k-d-tree that uses "tunnel distance" between two points on
    earth as a metric.
    Cf. http://en.wikipedia.org/wiki/K-d_tree.
    """

    def __init__(self, lat, lon):
        """
        Construct tree
        """
        x, y, z = latlon2unitsphere(lat, lon)
        points = np.dstack((x.ravel(), y.ravel(), z.ravel()))[0]
        self.kdtree = KDTree(points, metric="euclidean")
        self.lat, self.lon = lat, lon

    def nearest(self, lat0, lon0):
        """
        Query tree
        """
        x0, y0, z0 = latlon2unitsphere(lat0, lon0)
        distance_squared, idx_1d = self.kdtree.query([[x0, y0, z0]], k=1)
        distance_squared = distance_squared[0][0]
        idx_1d = idx_1d[0][0]

        # since our kd-tree is based on a 1d list of points, we may need to
        # convert idx_1d to 2D coordinates
        if len(self.lat.shape) == 1:
            idx = idx_1d
            lat = self.lat[idx]
            lon = self.lon[idx]
            return idx, lat, lon
        elif len(self.lat.shape) == 2:
            idx_row, idx_col = np.unravel_index(idx_1d, self.lat.shape)
            lat = self.lat[idx_row, idx_col]
            lon = self.lon[idx_row, idx_col]
            return idx_row, idx_col, lat, lon


def latlon2unitsphere(lat, lon):
    """
    Convert (lat, lon) to (x, y, z) coordinates w.r.t. unit sphere.

    Args:
        lat: latitude(s) in °, scalar or numpy array
        lon: longitude(s) in °, scalar or numpy array
    """
    # convert to radians
    rad_factor = np.pi / 180
    lat_rad = lat * rad_factor
    lon_rad = lon * rad_factor
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return (x, y, z)


def extract_subgrid(x, y, low_x, high_x, low_y, high_y):
    """
    Helper method.
    Extracts a subgrid from an x/Y grid.

    x and y must define a coordinate system with the origin in the lower
    left corner of the arrays!
    The origin is not necessarily 0/0, but the smallest x and y values must
    be in the lower left corner.
    E.g., latitudes must be descending (from top to down) and longitudes
    must be ascending (from left to right)

    Note that x and y can also be longitudes and latitudes, respectively.

    Returns a tuple (slice1, slice2) containing two slice objects
    (ax1_start, ax1_end, None) and (ax2_end, ax2_end, None), respectively.

    Args:
        x: 2d numpy array containing x values
        y: 2d numpy array containing y values
        low_x: lower bound x value
        low_y: lower bound y value
        high_x: upper bound x value
        high_y: upper bound y value

    Returns:
        TODO
    """
    no_cols = x.shape[1]

    # recall that slicing does not include end-values.
    # E.g., x[0:3] return elements 0, 1, and 2.

    # find out start and end rows
    #############################

    # note that rows are counted top/down (0 is 1st row, 1 is 2nd row, etc.)
    # and that the 1st row contains the largest y-values (recall that
    # grid-origin is in lower left corner)
    #
    # example of a y grid:
    #
    #      0      1       2       3
    # 0  48.1 | 48.12 | 48.08 | 48.2
    #    ---------------------------
    # 1  47.0 | 47.11 | 47.05 | 47.1
    #    ---------------------------
    # 2  46.0 | 46.11 | 46.05 | 46.1
    #    ---------------------------
    # 3  45.0 | 45.11 | 45.05 | 45.1
    #    ---------------------------
    # 4  44.0 | 44.11 | 44.05 | 44.1

    # we first look for the start row...
    # if y is nowhere <= high_y, we return an empty result
    tmp = y <= high_y
    if not tmp.max():
        return (slice(0, 0, None), slice(0, 0, None))
    # otherwise, start_row is the first row where at least one y-value
    # is <= high_y
    else:
        # index (in flattened tmp-array) of first True value
        idx_start = np.argmax(tmp)
        # row of first True value
        start_row = idx_start // no_cols

    # now we look for the end row...
    # if y is nowhere >= low_y, we return an empty result
    tmp = y >= low_y
    if not tmp.max():
        return (slice(0, 0, None), slice(0, 0, None))
    # if y is everywhere >= low_y, then end row is the last row
    elif tmp.min():
        end_row = y.shape[0]
    # otherwise, end_row is the first row where all the y-values are < low_y
    else:
        # Well, this is a bit hard to understand. A better explanation would
        # help. TODO
        end_row = np.argmin(tmp, 0).max() + 1

    # find out start and end columns
    ################################

    # example of an x grid:
    #
    #      0      1      2     3
    # 0  5.21 | 6.13 | 7.08 | 8.20
    #    -------------------------
    # 1  5.21 | 6.13 | 7.08 | 8.20
    #    -------------------------
    # 2  5.21 | 6.13 | 7.09 | 8.20
    #    -------------------------
    # 3  5.21 | 6.13 | 7.18 | 8.20
    #    -------------------------
    # 4  5.21 | 6.13 | 7.08 | 8.20

    # we first look for the start column...
    # if x is nowhere >= low_x, we return an empty subgrid
    tmp = x >= low_x
    if not tmp.max():
        return (slice(0, 0, None), slice(0, 0, None))
    # otherwise, start_col is the first column with a value >= low_x:
    else:
        start_col = np.argmax(tmp, 1).min()

    # now we look for the end column...
    # if x is nowhere <= high_x, we return an empty subgrid
    tmp = x <= high_x
    if not tmp.max():
        return (slice(0, 0, None), slice(0, 0, None))
    # if x is everywhere <= high_x, then end_col is the last column
    if tmp.min():
        end_col = x.shape[1]  # last col
    # otherwise, end_col is the first column with a value > high_x
    else:
        end_col = np.argmin(tmp, 1).max()

    return (slice(start_row, end_row, None), slice(start_col, end_col, None))
