"""Contour class"""
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

def long_diff(a, b, deg):
    """Difference in longitudes."""
    diff = np.asarray(a) - np.asarray(b)
    pi = 180 if deg else np.pi
    diff[diff > pi] -= 2*pi
    diff[diff < -pi] += 2*pi
    return diff

class BasePolygon:
    """Base polygon class."""
    def __init__(self, vertices):
        if not np.isclose(vertices[0], vertices[-1]).all():
            vertices = np.vstack((vertices, vertices[[0]]))
        self._vertices = np.asarray(vertices)
        self._area_units = ''

    @property
    def area_units(self):
        """Area units."""
        return self._area_units

    @property
    def vertices(self):
        """Polygon vertices (note that the last vertex equals to the first vertex)."""
        return self._vertices

    @property
    def size(self):
        """Number of vertices in the polygon."""
        return len(self.vertices) - 1

    @property
    def area(self):
        """Polygon area."""
        raise NotImplementedError

    @property
    def perimeter(self):
        """Perimeter of the polygon."""
        raise NotImplementedError

    @property
    def bbox(self):
        """Bounding box."""
        raise NotImplementedError

    @property
    def summary(self):
        """Summary properties."""
        raise NotImplementedError


class PlanePolygon(BasePolygon):
    """Plane polygon class.

    Parameters
    ----------
    vertices : array of shape (n, 2)
        Coordinates (x, y) of the vertices.
    """
    def __init__(self, vertices):
        super().__init__(vertices)
        self._area_units = 'dxdy'

    @property
    def x(self):
        """x-coordinates of the vertices."""
        return self.vertices[:, 0]

    @property
    def y(self):
        """y-coordinates of the vertices."""
        return self.vertices[:, 1]

    @property
    def area(self):
        """Polygon area."""
        a = np.hstack((self.x[-2], self.x[:-2]))
        b = self.x[1:]
        return abs(sum((a - b) * self.y[:-1])) / 2

    @property
    def perimeter(self):
        """Perimeter of the polygon."""
        return np.sqrt(np.diff(self.vertices, axis=1)**2).sum()

    @property
    def bbox(self):
        """Returns a tuple (x_min, y_min, x_max, y_max)."""
        return (self.x.min(), self.y.min(), self.x.max(), self.y.max())

    @property
    def bbox_center(self):
        """Returns a tuple (x_cen, y_cen)."""
        return ((self.x.min() + self.x.max()) / 2,
                (self.y.min() + self.y.max()) / 2)

    @property
    def dx(self):
        """Entent in x-direction."""
        return np.ptp(self.x)

    @property
    def dy(self):
        """Entent in y-direction."""
        return np.ptp(self.y)

    @property
    def summary(self):
        """Summary properties."""
        return dict(vertices=[list(x) for x in self.vertices],
                    area=self.area,
                    area_units=self.area_units,
                    bbox=list(self.bbox),
                    bbox_center=list(self.bbox_center),
                    dx=self.dx,
                    dy=self.dy,
                    perimeter=self.perimeter,
                    type=self.__class__.__name__)


class SphericalPolygon(BasePolygon):
    """Spherical polygon class.

    Parameters
    ----------
    vertices : array of shape (n, 2)
        Coordinates (lat, long) of the vertices.
    deg : bool
        True if coordinates are in degrees. False for radians.
    """
    def __init__(self, vertices, deg=True):
        super().__init__(vertices)
        self._deg = deg
        self._area_units = 'millionth of the solar hemisphere'

    @property
    def lats(self):
        """Latitude of the vertices."""
        return self.vertices[:, 0]

    @property
    def lons(self):
        """Longitude of the vertices."""
        return self.vertices[:, 1]

    @property
    def deg(self):
        """True if coordinates are in dergees."""
        return self._deg

    @property
    def area(self):
        """Polygon area in millionth of the solar hemisphere."""
        lats = np.deg2rad(self.lats) if self.deg else self.lats
        lons = np.deg2rad(self.lons) if self.deg else self.lons
        a = np.hstack((lons[-2], lons[:-2]))
        b = lons[1:]
        area = abs(sum(long_diff(a, b, deg=False) * np.sin(lats[:-1]))) / 2
        return area * 1e6 / 2 / np.pi

    @property
    def perimeter(self):
        """Perimeter of the polygon on a unit sphere."""
        res = 0
        a = np.deg2rad(self.lats) if self.deg else self.lats
        b = np.deg2rad(self.lons) if self.deg else self.lons
        for i in range(self.size):
            res += haversine_distances([[a[i], b[i]]], [[a[i+1], b[i+1]]])[0, 0]
        return res

    @property
    def bbox(self):
        """Returns a tuple (min lat, min lon, max lat, max lon)."""
        diffs = long_diff(self.lons, self.lons[0], deg=self.deg)
        l_min = (self.lons[0] + diffs.min()) % (360 if self.deg else 2*np.pi)
        l_max = (self.lons[0] + diffs.max()) % (360 if self.deg else 2*np.pi)
        return (self.lats.min(), l_min, self.lats.max(), l_max)

    @property
    def bbox_center(self):
        """Returns a tuple (lat_cen, lon_cen)."""
        lat_min, lon_min, lat_max, lon_max = self.bbox
        lon_cen = (lon_min + long_diff([lon_max], [lon_min], deg=self.deg)[0] / 2)
        lon_cen = lon_cen % (360 if self.deg else 2*np.pi)
        return ((lat_min + lat_max) / 2, lon_cen)

    @property
    def dlon(self):
        """Longitudinal extent."""
        return np.ptp(long_diff(self.lons, self.lons[0], deg=self.deg))

    @property
    def dlat(self):
        """Latitudinal extent."""
        return np.ptp(self.lats)

    @property
    def summary(self):
        """Summary properties."""
        return dict(vertices=[list(x) for x in self.vertices],
                    area=self.area,
                    area_units=self.area_units,
                    bbox=list(self.bbox),
                    bbox_center=list(self.bbox_center),
                    perimeter=self.perimeter,
                    dlat=self.dlat,
                    dlon=self.dlon,
                    deg=self.deg,
                    type=self.__class__.__name__)
