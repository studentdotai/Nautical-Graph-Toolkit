"""
port_utils.py

Utility classes for handling Port data. This includes loading standard
port shapefiles, managing custom user-defined ports, handling acronyms,
and providing helper functions for data retrieval and formatting.
"""
import logging
from typing import Union, List, Dict
from shapely.geometry import Point
import pandas as pd
from pathlib import Path
from shapely.geometry import box, MultiPolygon

from ..utils.misc_utils import CoordinateConverter
from ..utils.s57_utils import S57Utils
from ..core.s57_data import ENCDataFactory

try:
    import geopandas as gpd
except ImportError:
    print("Warning: geopandas is not installed. Some functionality may be limited.")

logger = logging.getLogger(__name__)


class PortData:
    """
    A utility class to manage and provide access to world port data.
    It handles loading the port shapefile and acronyms, and provides
    helper methods to access and format port information.
    """

    def __init__(self):
        """
        Initializes the PortData utility by loading the necessary data files.
        """
        # The data directory is one level up from the 'utils' directory.
        self._data_dir = Path(__file__).resolve().parent.parent / 'data'
        self.ports_msi_shp_path = self._data_dir / 'World Port Index_2019Shapefile' / 'WPI.shp'
        self.custom_ports_path = self._data_dir / 'custom_ports.csv'
        self.ports_msi_acronyms_path = self._data_dir / 'WorldPortIndex_2019.csv'

        # Load standard ports and then merge custom ports
        self.standard_ports_df = self._load_port_shapefile()
        self.port_df = self._load_and_merge_custom_ports()
        self.port_acronym_df = self._load_port_acronyms()
        self._initialize_custom_ports_file()

    def _load_port_shapefile(self) -> 'gpd.GeoDataFrame':
        """Loads the port shapefile into a GeoDataFrame."""
        if not self.ports_msi_shp_path.exists():
            raise FileNotFoundError(f"Port shapefile not found at: {self.ports_msi_shp_path}")
        return gpd.read_file(self.ports_msi_shp_path).to_crs(epsg=4326)

    def _load_port_acronyms(self) -> 'pd.DataFrame':
        """Loads the port acronyms CSV into a DataFrame."""
        if not self.ports_msi_acronyms_path.exists():
            raise FileNotFoundError(f"Port acronyms file not found at: {self.ports_msi_acronyms_path}")
        return pd.read_csv(self.ports_msi_acronyms_path)

    def _load_and_merge_custom_ports(self) -> 'gpd.GeoDataFrame':
        """
        Loads custom ports from a CSV file and merges them with the standard ports.
        """
        standard_ports = self.standard_ports_df.copy()

        if not self.custom_ports_path.exists():
            logger.info("No custom ports file found. Using standard ports only.")
            return standard_ports

        try:
            custom_ports_df = pd.read_csv(self.custom_ports_path)
            if custom_ports_df.empty:
                return standard_ports

            # Create geometry from longitude and latitude
            geometry = gpd.points_from_xy(custom_ports_df.LONGITUDE, custom_ports_df.LATITUDE)
            custom_ports_gdf = gpd.GeoDataFrame(custom_ports_df, geometry=geometry, crs="EPSG:4326")

            # Ensure column names match for concatenation
            custom_ports_gdf.columns = [col.upper() for col in custom_ports_gdf.columns]

            logger.info(f"Loaded {len(custom_ports_gdf)} custom ports. Merging with standard ports.")
            # Concatenate and return the merged GeoDataFrame
            return pd.concat([standard_ports, custom_ports_gdf], ignore_index=True)

        except Exception as e:
            logger.error(f"Error loading or processing custom ports file: {e}")
            return standard_ports

    def _initialize_custom_ports_file(self):
        """
        Initializes the custom_ports.csv file with headers if it doesn't exist.
        The headers are derived from the standard port shapefile schema.
        """
        if not self.custom_ports_path.exists():
            logger.info(f"Custom ports file not found. Creating template at: {self.custom_ports_path}")
            # Use the schema from the standard ports dataframe
            schema = self.standard_ports_df.columns.tolist()
            # Create an empty dataframe with this schema and save it
            pd.DataFrame(columns=schema).to_csv(self.custom_ports_path, index=False)

    def create_custom_port(self, port_name: str, lon: float, lat: float, country: str = 'CUSTOM',
                           if_exists: str = 'skip', **kwargs):
        """
        Adds a new custom port to the custom_ports.csv file.

        Args:
            port_name (str): The name of the new custom port.
            lon (float): The longitude of the port.
            lat (float): The latitude of the port.
            country (str): The country of the port.
            if_exists (str): Action to take if a port with the same name already exists.
                             Options: 'skip' (default), 'overwrite', 'update'.
            **kwargs: Additional attributes to set for the port (e.g., HARBORSIZE='S').
                      Keys must match the column headers in the CSV.
        """
        self._initialize_custom_ports_file()  # Ensure file and headers exist

        # Check if the port already exists
        custom_ports_df = pd.read_csv(self.custom_ports_path)
        port_exists = not custom_ports_df[custom_ports_df['PORT_NAME'].str.upper() == port_name.upper()].empty

        if port_exists:
            if if_exists == 'skip':
                logger.info(f"Custom port '{port_name}' already exists. Skipping creation.")
                return
            elif if_exists == 'overwrite':
                logger.info(f"Port '{port_name}' exists. Overwriting.")
                self.delete_custom_port(port_name)
            elif if_exists == 'update':
                logger.info(f"Port '{port_name}' exists. Updating.")
                # Combine lat/lon with other kwargs for the update
                update_args = {'lon': lon, 'lat': lat, 'country': country, **kwargs}
                self.update_custom_port(port_name, **update_args)
                return
            else:
                raise ValueError(f"Invalid value for if_exists: '{if_exists}'. Must be 'skip', 'overwrite', or 'update'.")

        # Convert decimal coordinates to Degrees, Minutes, Hemisphere (DMH) format
        lat_deg, lat_min, lat_hemi = CoordinateConverter.decimal_to_dmh(lat, is_latitude=True)
        lon_deg, lon_min, lon_hemi = CoordinateConverter.decimal_to_dmh(lon, is_latitude=False)

        # Create a Shapely Point and get its WKT representation for the geometry column
        point_geom = Point(lon, lat)

        new_port_data = {
            'PORT_NAME': port_name.upper(),
            'COUNTRY': country.upper(),
            'LATITUDE': lat,
            'LONGITUDE': lon,
            'LAT_DEG': lat_deg,
            'LAT_MIN': lat_min,
            'LAT_HEMI': lat_hemi,
            'LONG_DEG': lon_deg,
            'LONG_MIN': lon_min,
            'LONG_HEMI': lon_hemi,
            'geometry': point_geom.wkt,
        }

        # Add any additional user-provided attributes
        new_port_data.update({k.upper(): v for k, v in kwargs.items()})

        # Get the full list of expected columns from the standard port dataframe
        # This ensures that the new row has all columns, filling missing ones with None/NaN
        full_schema_columns = self.standard_ports_df.columns.tolist()

        # Create a dictionary with all schema columns, filling with None for missing values
        # and then updating with the provided new_port_data
        full_row_data = {col: new_port_data.get(col, None) for col in full_schema_columns}
        new_port_df = pd.DataFrame([full_row_data])

        new_port_df.to_csv(self.custom_ports_path, mode='a', header=False, index=False)
        logger.info(f"Appended custom port '{port_name}' to {self.custom_ports_path}")

        # Reload the main port dataframe to include the new port immediately
        self.port_df = self._load_and_merge_custom_ports()

    def delete_custom_port(self, port_name: str):
        """
        Deletes a custom port from the custom_ports.csv file by name.

        Args:
            port_name (str): The name of the custom port to delete.
        """
        if not self.custom_ports_path.exists():
            logger.warning("Custom ports file does not exist. Cannot delete port.")
            return

        custom_ports_df = pd.read_csv(self.custom_ports_path)
        original_count = len(custom_ports_df)

        # Filter out the port to be deleted (case-insensitive)
        updated_df = custom_ports_df[custom_ports_df['PORT_NAME'].str.upper() != port_name.upper()]

        if len(updated_df) == original_count:
            logger.warning(f"Custom port '{port_name}' not found for deletion.")
            return

        # Save the updated DataFrame back to the CSV
        updated_df.to_csv(self.custom_ports_path, index=False)
        logger.info(f"Deleted custom port '{port_name}'.")

        # Reload the main port dataframe to reflect the change
        self.port_df = self._load_and_merge_custom_ports()

    def update_custom_port(self, port_name: str, **kwargs):
        """
        Updates attributes of an existing custom port.

        Args:
            port_name (str): The name of the custom port to update.
            **kwargs: Key-value pairs of attributes to update (e.g., lat=38.0, HARBORSIZE='M').
        """
        if not self.custom_ports_path.exists():
            logger.warning("Custom ports file does not exist. Cannot update port.")
            return

        custom_ports_df = pd.read_csv(self.custom_ports_path)
        port_index = custom_ports_df[custom_ports_df['PORT_NAME'].str.upper() == port_name.upper()].index

        if port_index.empty:
            logger.warning(f"Custom port '{port_name}' not found. Cannot update.")
            return

        idx = port_index[0]

        # Check if coordinates are being updated
        new_lat = kwargs.get('lat', custom_ports_df.loc[idx, 'LATITUDE'])
        new_lon = kwargs.get('lon', custom_ports_df.loc[idx, 'LONGITUDE'])

        # If lat or lon is provided, recalculate all coordinate fields
        if 'lat' in kwargs or 'lon' in kwargs:
            lat_deg, lat_min, lat_hemi = CoordinateConverter.decimal_to_dmh(new_lat, is_latitude=True)
            lon_deg, lon_min, lon_hemi = CoordinateConverter.decimal_to_dmh(new_lon, is_latitude=False)
            point_geom = Point(new_lon, new_lat)

            custom_ports_df.loc[idx, 'LATITUDE'] = new_lat
            custom_ports_df.loc[idx, 'LONGITUDE'] = new_lon
            custom_ports_df.loc[idx, 'LAT_DEG'] = lat_deg
            custom_ports_df.loc[idx, 'LAT_MIN'] = lat_min
            custom_ports_df.loc[idx, 'LAT_HEMI'] = lat_hemi
            custom_ports_df.loc[idx, 'LONG_DEG'] = lon_deg
            custom_ports_df.loc[idx, 'LONG_MIN'] = lon_min
            custom_ports_df.loc[idx, 'LONG_HEMI'] = lon_hemi
            custom_ports_df.loc[idx, 'geometry'] = point_geom.wkt

        # Update other provided attributes
        for key, value in kwargs.items():
            if key.upper() in custom_ports_df.columns:
                custom_ports_df.loc[idx, key.upper()] = value

        # Save the updated DataFrame back to the CSV
        custom_ports_df.to_csv(self.custom_ports_path, index=False)
        logger.info(f"Updated attributes for custom port '{port_name}'.")

        # Reload the main port dataframe to reflect the changes
        self.port_df = self._load_and_merge_custom_ports()

    def update_custom_port_coords(self, port_name: str, new_lon: float, new_lat: float):
        """
        Updates the coordinates of an existing custom port in the custom_ports.csv file.

        This method will find the port by name, update all coordinate-related fields
        (decimal, DMH, and WKT geometry), and save the changes.

        Args:
            port_name (str): The name of the custom port to update.
            new_lon (float): The new longitude for the port.
            new_lat (float): The new latitude for the port.
        """
        self.update_custom_port(port_name, lon=new_lon, lat=new_lat)

    def get_port_names(self) -> list:
        """Returns a sorted list of all port names."""
        return sorted(self.port_df['PORT_NAME'].tolist())

    def get_port_by_name(self, port_name: str) -> 'pd.Series':
        """
        Retrieves the full data record for a single port by its name.

        Args:
            port_name (str): The name of the port to retrieve.

        Returns:
            pd.Series: A pandas Series containing the port's information.
        """
        port_info = self.port_df[self.port_df['PORT_NAME'] == port_name.upper()]
        if not port_info.empty:
            return port_info.iloc[0]
        return None

    def get_port_index(self, port_name: str) -> int:
        """
        Gets the numerical index of a port from the sorted list of port names.
        Useful for setting default indices in Streamlit selectboxes.

        Args:
            port_name (str): The name of the port.

        Returns:
            int: The index of the port. Returns 0 if not found.
        """
        try:
            return self.get_port_names().index(port_name.upper())
        except ValueError:
            return 0

    def format_port_string(self, port_series: 'pd.Series') -> str:
        """Formats a port's information into a human-readable string."""
        if port_series is None or not isinstance(port_series, pd.Series):
            return "Invalid Port Data"
        return (
            f"{port_series.get('PORT_NAME', 'N/A')}, {port_series.get('COUNTRY', 'N/A')} "
            f"(LAT: {int(port_series.get('LAT_DEG', 0))}° {port_series.get('LAT_MIN', 0)}' {port_series.get('LAT_HEMI', '')}  "
            f"LONG: {int(port_series.get('LONG_DEG', 0))}° {port_series.get('LONG_MIN', 0)}' {port_series.get('LONG_HEMI', '')})"
        )

    def get_port_details_df(self, port_series: 'pd.Series') -> 'pd.DataFrame':
        """Converts a port's data into a cleaned DataFrame for display."""
        df = port_series.copy().to_frame(name='Value')
        indices_to_drop = ['INDEX_NO', 'REGION_NO', 'LAT_DEG', 'LAT_MIN', 'LAT_HEMI', 'LONG_DEG', 'LONG_MIN', 'LONG_HEMI', 'geometry']
        df = df.drop(index=indices_to_drop, errors='ignore')
        df = df.replace(['N', 0, '', None], pd.NA).dropna()
        acronym_dict = dict(zip(self.port_acronym_df['Acronym'], self.port_acronym_df['Meaning']))
        df.index = df.index.map(lambda x: acronym_dict.get(x, x))
        return df


class Boundaries:
    """A utility class for creating and manipulating geospatial boundaries."""

    def create_geo_boundary(self, geometries: Union[gpd.GeoDataFrame, gpd.GeoSeries, List['BaseGeometry']],
                            expansion: Union[float, Dict[str, float]] = None,
                            crs: int = 4326,
                            precision: int = 3,
                            date_line: bool = False) -> 'gpd.GeoDataFrame':
        """
         Creates a boundary box (or a MultiPolygon of 2 boxes) from input geometries with optional expansion.

        When `date_line` is True and the original bounds indicate a dateline crossing
        (i.e. the longitudinal span > 180°), this function applies expansion separately for the west and east.

        Args:
            geometries: Input geometries (GeoDataFrame, GeoSeries or list of geometries).
            expansion: Expansion distance in nautical miles. Either a uniform float or a dict with directional keys.
                       For directional expansion, use keys 'W', 'E', 'N', 'S'.
            crs: Coordinate reference system (default: EPSG:4326)
            precision: Decimal precision for rounding boundary coordinates.
            date_line: If True, treat the case where the geometry spans the dateline.

        Returns:
            A GeoDataFrame containing the boundary geometry. In the dateline case a MultiPolygon is returned.
        """

        def _rnd(_value):
            return round(_value, precision)

        # Convert input to GeoSeries if needed
        if isinstance(geometries, (list, tuple)):
            geom_series = gpd.GeoSeries(geometries)
        elif isinstance(geometries, gpd.GeoDataFrame):
            geom_series = geometries.geometry
        else:
            geom_series = geometries

        # Ensure CRS is set
        geom_series = geom_series.set_crs(crs)

        # Capture the original bounds.
        orig_minx, orig_miny, orig_maxx, orig_maxy = geom_series.total_bounds

        # Define a conversion factor: 1 nautical mile is roughly 1/60 of a degree.
        nm_to_deg = 1 / 60.0

        # Prepare expansion values for directional adjustments.
        if expansion is None:
            exp_w = exp_e = exp_n = exp_s = 0.0
        elif isinstance(expansion, (int, float)):
            exp_w = exp_e = exp_n = exp_s = expansion * nm_to_deg
        elif isinstance(expansion, dict):
            exp_w = expansion.get('W', 0) * nm_to_deg
            exp_e = expansion.get('E', 0) * nm_to_deg
            exp_n = expansion.get('N', 0) * nm_to_deg
            exp_s = expansion.get('S', 0) * nm_to_deg
        else:
            exp_w = exp_e = exp_n = exp_s = 0.0

        # If date_line is requested and the original span suggests dateline crossing.
        if date_line and ((orig_maxx - orig_minx) > 180):
            # Apply expansion separately to each side.
            left_minx = orig_maxx - exp_w
            left_miny = orig_miny - exp_s
            left_maxy = orig_maxy + exp_n
            left_box = box(_rnd(left_minx), _rnd(left_miny), 180, _rnd(left_maxy))

            right_maxx = orig_minx + exp_e
            right_miny = orig_miny - exp_s
            right_maxy = orig_maxy + exp_n
            right_box = box(-180, _rnd(right_miny), _rnd(right_maxx), _rnd(right_maxy))

            multi = MultiPolygon([left_box, right_box])
            return gpd.GeoDataFrame({'geometry': [multi]}, crs=crs)
        else:
            # Non‑dateline: Apply expansion uniformly/directionally.
            new_minx = _rnd(orig_minx - exp_w)
            new_miny = _rnd(orig_miny - exp_s)
            new_maxx = _rnd(orig_maxx + exp_e)
            new_maxy = _rnd(orig_maxy + exp_n)
            single_bbox = box(new_minx, new_miny, new_maxx, new_maxy)
            return gpd.GeoDataFrame({'geometry': [single_bbox]}, crs=crs)
