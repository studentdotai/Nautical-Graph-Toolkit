import timeit
import logging
from typing import Union, List, Dict

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, MultiPolygon, mapping
from shapely.geometry.base import BaseGeometry


logger = logging.getLogger(__name__)


class Miscellaneous:
    """A collection of miscellaneous utility functions."""

    def __init__(self):
        pass

    @staticmethod
    def perf_test(function, name: str = "", iterations: int = 1):
        """
        A simple performance test utility.

        Args:
            function: The function to test.
            name (str): A descriptive name for the test.
            iterations (int): The number of times to run the function.
        """
        time_taken = timeit.timeit(lambda: function(), number=iterations)
        print(f"'{name}' executed {iterations} times in {time_taken:.4f} seconds")

    @staticmethod
    def shp_to_gdf(input_file_path: str, crs: int = 4326) -> gpd.GeoDataFrame:
        """Loads a shapefile into a GeoDataFrame and re-projects it."""
        return gpd.read_file(input_file_path).to_crs(epsg=crs)

    @staticmethod
    def shapely_polygon_to_geojson(polygon: Union[BaseGeometry]) -> dict:
        """Converts a Shapely Polygon or MultiPolygon into a GeoJSON dictionary."""
        return mapping(polygon)

    @staticmethod
    def _standardize_enc_name_column(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes ENC name column from 'dsid_dsnm' to 'ENC_NAME'."""
        df = df.copy()
        if 'dsid_dsnm' in df.columns:
            df = df.rename(columns={'dsid_dsnm': 'ENC_NAME'})
        return df

    @staticmethod
    def name_list_to_bands(name_list: List[str]) -> Dict[str, List[str]]:
        """
        Segregates a list of ENC names into appropriate Usage Bands.

        Usage Bands:
        1: Overview, 2: General, 3: Coastal, 4: Approach, 5: Harbour, 6: Berthing
        """
        if not name_list:
            return {}

        usage_bands = {
            'Overview': [], 'General': [], 'Coastal': [],
            'Approach': [], 'Harbour': [], 'Berthing': []
        }

        for enc in name_list:
            if len(enc) > 2:
                usage_band_char = enc[2]
                if usage_band_char == '1':
                    usage_bands['Overview'].append(enc)
                elif usage_band_char == '2':
                    usage_bands['General'].append(enc)
                elif usage_band_char == '3':
                    usage_bands['Coastal'].append(enc)
                elif usage_band_char == '4':
                    usage_bands['Approach'].append(enc)
                elif usage_band_char == '5':
                    usage_bands['Harbour'].append(enc)
                elif usage_band_char == '6':
                    usage_bands['Berthing'].append(enc)

        return {k: v for k, v in usage_bands.items() if v}

    @staticmethod
    def gdf_to_df(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Converts a GeoDataFrame to a regular Pandas DataFrame by extracting coordinates
        from a Point geometry column into separate 'longitude' and 'latitude' columns.
        """
        df = gdf.copy()
        if 'geometry' in df.columns and not df.geometry.empty:
            # Ensure all geometries are points before accessing .x and .y
            if all(isinstance(geom, gpd.points.Point) for geom in df.geometry):
                df['longitude'] = df.geometry.x
                df['latitude'] = df.geometry.y
        df = df.drop(columns=['geometry'], errors='ignore')
        return df

    @staticmethod
    def miles_to_decimal(nautical_miles: float) -> float:
        """
        Converts nautical miles to decimal degrees.
        1 nautical mile ≈ 1/60 of a degree.
        """
        return nautical_miles / 60.0


class CoordinateConverter:
    """A utility class for converting between different coordinate formats."""

    @staticmethod
    def dmh_to_decimal(degrees: float, minutes: float, hemisphere: str) -> float:
        """
        Converts Degrees, Minutes, Hemisphere (DMH) to decimal degrees.

        Args:
            degrees (float): The degrees part of the coordinate.
            minutes (float): The minutes part of the coordinate.
            hemisphere (str): The hemisphere ('N', 'S', 'E', 'W').

        Returns:
            float: The coordinate in decimal degrees.
        """
        if not isinstance(hemisphere, str) or hemisphere.upper() not in ['N', 'S', 'E', 'W']:
            raise ValueError("Hemisphere must be one of 'N', 'S', 'E', 'W'.")

        decimal = float(degrees) + (float(minutes) / 60.0)

        if hemisphere.upper() in ['S', 'W']:
            decimal *= -1

        return decimal

    @staticmethod
    def decimal_to_dmh(decimal_degrees: float, is_latitude: bool) -> tuple:
        """
        Converts decimal degrees to Degrees, Minutes, Hemisphere (DMH).

        Args:
            decimal_degrees (float): The coordinate in decimal degrees.
            is_latitude (bool): True if the coordinate is a latitude, False for longitude.

        Returns:
            tuple: A tuple containing (degrees, minutes, hemisphere).
        """
        decimal_degrees = float(decimal_degrees)
        abs_decimal = abs(decimal_degrees)

        degrees = int(abs_decimal)
        minutes = (abs_decimal - degrees) * 60

        if is_latitude:
            hemisphere = 'N' if decimal_degrees >= 0 else 'S'
        else:
            hemisphere = 'E' if decimal_degrees >= 0 else 'W'

        return degrees, minutes, hemisphere

    @staticmethod
    def string_to_float(coord_string: str) -> float:
        """Converts a coordinate string to a float, handling potential errors."""
        try:
            return float(coord_string)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def dms_string_to_decimal(coord_string: str) -> float:
        """
        Converts a coordinate string in Degrees-Minutes-Hemisphere format to decimal degrees.

        This function is designed to parse strings like:
        - "49' 25,4 N"
        - "016 55,6 E"
        - "118 20.7 W"

        Args:
            coord_string (str): The coordinate string to convert.

        Returns:
            float: The coordinate in decimal degrees.
        """
        if not isinstance(coord_string, str) or not coord_string.strip():
            return 0.0

        try:
            # Clean and split the string
            cleaned_string = coord_string.strip().replace(',', '.').replace("'", " ")
            parts = cleaned_string.split()

            if len(parts) < 2:
                raise ValueError("Coordinate string format is invalid.")

            hemisphere = parts[-1]
            minutes = float(parts[-2])
            degrees = float(parts[-3]) if len(parts) > 2 else 0.0

            # If there are more parts, they all belong to degrees (e.g. "016 55.6 E")
            if len(parts) == 3:
                degrees = float(parts[0])

            return CoordinateConverter.dmh_to_decimal(degrees, minutes, hemisphere)
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse coordinate string '{coord_string}': {e}")
            return 0.0