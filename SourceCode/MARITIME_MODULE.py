import logging

import geopandas as gpd
import pandas as pd
import numpy as np
import fiona
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from shapely import wkt
from shapely.geometry import box, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
import plotly.graph_objects as go
import plotly.colors as pc
import os
import timeit

from shapely.geometry.polygon import Polygon
from sqlalchemy import create_engine
from typing import Union, List, Dict, Any

from Data import Data, NOAA_DB, GPKG, PostGIS


class Miscellaneous:
	def __init__(self):
		pass

	def perf_test(self, function, name = "", iterations = 1):
		# Time create_grid_graph
		time_cgg = timeit.timeit(lambda: function(), number=iterations)
		print(f"{name} executed {iterations} times in {time_cgg:.4f} seconds")

	# Load the input file
	def shp_to_gdf(self, input_file_path, crs = 4326):
		output_file = gpd.read_file(input_file_path).to_crs(epsg=crs)
		return output_file

	@staticmethod
	def shapely_polygon_to_geojson(polygon):
		"""
		Converts a Shapely Polygon or MultiPolygon into a GeoJSON dictionary.
		Parameters:
			polygon (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The Shapely geometry.
		Returns:
			dict: A GeoJSON representation of the input polygon.
		"""
		from shapely.geometry import mapping
		return mapping(polygon)

	def _standardize_enc_name_column(self, df) -> pd.DataFrame:
		"""
		Standardizes ENC name column from 'dsid_dsnm' to 'ENC_NAME'
		Args:
			df: Input DataFrame containing ENC names
		Returns:
			pd.DataFrame: DataFrame with standardized column name
		Examples:
			df = standardize_enc_name_column(df)
		"""
		df = df.copy()
		if 'dsid_dsnm' in df.columns:
			df = df.rename(columns={'dsid_dsnm': 'ENC_NAME'})
		return df

	def test_enc_paths(self, paths: list[str]) -> dict:
		"""
		Tests a list of file paths, identifies valid GeoPackage (ENC) files,
		extracts their ENC names, and compiles a list of unique layers.
		Args:
			paths (list[str]): A list of file paths to test.
		Returns:
			dict: A dictionary containing:
				- "enc_names": A list of ENC names extracted from valid GeoPackage files.
				- "unique_layers": A list of unique layer names across all valid ENCs.
		"""
		valid_enc_names = []
		all_layers = set()

		for path in paths:
			try:
				# Check if the file exists and is a GeoPackage
				if not os.path.exists(path):
					print(f"Warning: Path does not exist: {path}")
					continue
				if not path.lower().endswith(".gpkg"):
					print(f"Warning: Not a .gpkg file: {path}")
					continue

				# Extract ENC name using Fiona
				try:
					with fiona.open(path, layer="DSID") as dsid_layer:
						dsid_info = next(iter(dsid_layer))['properties']
						enc_name = dsid_info.get('DSID_DSNM', 'Unknown')
						enc_name = enc_name.rsplit('.', 1)[0]  # Remove .gpkg or .000
						valid_enc_names.append(enc_name)
				except Exception as e:
					print(f"Warning: Could not read ENC name from {path}: {e}")
					continue

				# Extract layer names using Fiona
				try:
					layers = fiona.listlayers(path)
					all_layers.update(layers)
				except Exception as e:
					print(f"Warning: Could not list layers from {path}: {e}")
					continue

			except Exception as e:
				print(f"Error processing path {path}: {e}")

		return {
			"enc_names": valid_enc_names,
			"unique_layers": sorted(list(all_layers)),
		}

	def name_list_to_bands(self, name_list: list) -> dict[str, list]:
		"""
		Sefregate ENC_NAME list to appropriate Usage bands

		Usage Bands:
		1: Overview
		2: General
		3: Coastal
		4: Approach
		5: Harbour
		6: Berthing
		"""

		if name_list:
			usage_bands = {
				'Overview': [], 'General': [], 'Coastal': [],
				'Approach': [], 'Harbour': [], 'Berthing': []
			}

			for enc in name_list:
				usage_band = enc[2]  # Get usage band from ENC name
				if usage_band == '1':
					usage_bands['Overview'].append(enc)
				elif usage_band == '2':
					usage_bands['General'].append(enc)
				elif usage_band == '3':
					usage_bands['Coastal'].append(enc)
				elif usage_band == '4':
					usage_bands['Approach'].append(enc)
				elif usage_band == '5':
					usage_bands['Harbour'].append(enc)
				elif usage_band == '6':
					usage_bands['Berthing'].append(enc)

			return {k: v for k, v in usage_bands.items() if v}  # Return only non-empty bands



	def create_geo_boundary(self, geometries: Union[gpd.GeoDataFrame, gpd.GeoSeries, List[BaseGeometry]],
							expansion: Union[float, Dict[str, float]] = None,
							crs: int = 4326,
							precision: int = 3,
							date_line:bool = False):
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
			# Left box extends from left_minx to 180.
			left_box = box(_rnd(left_minx), _rnd(left_miny), 180, _rnd(left_maxy))

			right_maxx = orig_minx + exp_e
			right_miny = orig_miny - exp_s
			right_maxy = orig_maxy + exp_n
			# Right box extends from -180 to right_maxx.
			right_box = box(-180, _rnd(right_miny), _rnd(right_maxx), _rnd(right_maxy))

			multi = MultiPolygon([left_box, right_box])
			bbox_gdf = gpd.GeoDataFrame({
				'geometry': [multi],
				'expansion_type': ['dateline_directional'],
				'expansion_value': [expansion]
			}, crs=crs)
		else:
			# Non‑dateline: Apply expansion uniformly/directionally.
			new_minx = orig_minx - exp_w
			new_miny = orig_miny - exp_s
			new_maxx = orig_maxx + exp_e
			new_maxy = orig_maxy + exp_n
			new_minx = _rnd(new_minx)
			new_miny = _rnd(new_miny)
			new_maxx = _rnd(new_maxx)
			new_maxy = _rnd(new_maxy)
			single_bbox = box(new_minx, new_miny, new_maxx, new_maxy)
			bbox_gdf = gpd.GeoDataFrame({
				'geometry': [single_bbox],
				'expansion_type': ['uniform' if isinstance(expansion, (int, float))
								   else 'directional' if expansion else 'none'],
				'expansion_value': [expansion]
			}, crs=crs)

		return bbox_gdf

	def gdf_to_df(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
		"""
		Converts a GeoDataFrame to a regular Pandas DataFrame by extracting coordinates
		from geometry column into separate latitude and longitude columns.
		Args:
			gdf (gpd.GeoDataFrame): Input GeoDataFrame with geometry column
		Returns:
			pd.DataFrame: DataFrame with geometry converted to lat/lon columns
		"""
		# Create a copy to avoid modifying the original
		df = gdf.copy()

		# Extract coordinates from geometry column
		df['longitude'] = df.geometry.x
		df['latitude'] = df.geometry.y

		# Drop the geometry column
		df = df.drop(columns=['geometry'])
		return df

	def miles_to_decimal(self, nautical_miles: float) -> float:
		"""
		Converts nautical miles to decimal degrees.
		1 nautical mile = 1/60 of a degree (1 minute of arc)
		Args:
			nautical_miles (float): Distance in nautical miles
		Returns:
			float: Distance in decimal degrees

		"""
		return nautical_miles / 60.0

class ENC:
	def __init__(self, driver: str = "GPKG" ):
		"""

		:param driver: Optionas: "GPKG" or "POSTGIS"
		"""
		if driver == "GPKG":
			try:
				self.folder_path = ""
				self.gpkg = GPKG(self.folder_path)
			except:
				raise ValueError("Invalid folder path or driver.")

		self.enc_files = []
		self.sorted_encs = defaultdict(list)
		self.usage_bands = {
			'1': 'Overview',
			'2': 'General',
			'3': 'Coastal',
			'4': 'Approach',
			'5': 'Harbor',
			'6': 'Berthing'
		}
		self.enc_db = NOAA_DB().get_dataframe()
		self.data = Data()
		self.postgis = PostGIS()


	def bbox_gpkg(self, enc_files):
		boundary_data = []

		for enc_file in enc_files:
			enc_path = os.path.join(self.folder_path, enc_file)
			stamp = self.gpkg._fiona_stamp(enc_path)

			with fiona.open(enc_path, layer='M_COVR') as layer:
				for feature in layer:
					if feature['properties'].get('CATCOV') == 1:  # 1 indicates coverage area
						boundary = {
							'ENC_NAME': stamp['ENC_NAME'],
							'geometry': feature['geometry']
						}
						boundary_data.append(boundary)

		return gpd.GeoDataFrame(boundary_data, crs="EPSG:4326")

	def inspect_layer(self, dataframe):
		df = dataframe.copy()

		# Create copy and drop geometry column if exists
		if 'wkb_geometry' in df.columns:
			df = df.drop(columns=['wkb_geometry'])

		# Rename columns with attribute names
		for column in df.columns.tolist():
			attr_name = self.data.s57_attributes_convert(column.upper())
			if attr_name:
				df = df.rename(columns={column: f'{attr_name} ({column})'})

		# Return transposed DataFrame
		return df

	def layer_property_inspector(self, dataframe, acronym_convert=True, property_convert=True, prop_mixed=True,
								 unique=False, debug=False):
		"""
		Inspect and convert S-57 layer properties in a DataFrame.

		Args:
			dataframe: Input DataFrame containing S-57 layer data
			acronym_convert: If True, converts S-57 acronyms to full names
			property_convert: If True, converts property values to human-readable form
			prop_mixed: If True, includes original code with converted value
			unique: If True, returns a reference table with unique property values per column
			debug: If True, enables detailed logging

		Returns:
			pd.DataFrame: DataFrame with converted column names and values
		"""
		# Initialize logger for debugging if needed
		logger = logging.getLogger(__name__)
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		logger.setLevel(logging.DEBUG if debug else logging.INFO)

		# Create a deep copy to avoid modifying the original
		df = dataframe.copy()

		# Handle geometry column if present
		if 'wkb_geometry' in df.columns:
			df = df.drop(columns=['wkb_geometry'])
		elif 'geometry' in df.columns:
			df = df.drop(columns=['geometry'])

		# Define columns to exclude from conversion
		exclude_columns = ['geometry', 'wkb_geometry', 'ENC_NAME', 'ENC_EDITION',
						   'ENC_UPDATE', 'dsid_dsnm', 'dsid_edtn', 'dsid_updn']


		# If unique mode, return reference table of unique values per attribute
		if unique:
			unique_values_dict = {}
			# Get valid S-57 acronyms from data.s57_properties_df
			property_df = self.data.s57_properties_df()
			# Find the maximum number of values for any attribute
			max_values = 0
			valid_acronyms = set(property_df['Acronym'].str.upper())
			# Keep only valid columns and excluded columns
			valid_columns = [col for col in df.columns if col.upper() in valid_acronyms]

			# Find maximum number of values for any attribute
			for col in valid_columns:
				col_upper = col.upper()
				acronym_group = property_df[property_df['Acronym'].str.upper() == col_upper]
				max_values = max(max_values, len(acronym_group))

			# Create a dataframe with enough rows for all values
			result_df = pd.DataFrame(index=range(max_values))

			# Process only the acronyms that match columns in the dataframe
			for col in valid_columns:
				col_upper = col.upper()
				# Get the property group for this acronym
				acronym_group = property_df[property_df['Acronym'].str.upper() == col_upper]

				if not acronym_group.empty:
					# Create formatted values series
					formatted_values = acronym_group.apply(
						lambda row: f"{row['ID']} - {row['Meaning']}", axis=1
					).reset_index(drop=True)

					# Pad with empty strings if needed
					if len(formatted_values) < max_values:
						padding = pd.Series([''] * (max_values - len(formatted_values)))
						formatted_values = pd.concat([formatted_values, padding], ignore_index=True)

					# Create column name
					column_name = self.data.s57_attributes_convert(col_upper)
					if column_name:
						column_name = f"{column_name} ({col})"
					else:
						column_name = col

					# Add the series to the result dataframe
					result_df[column_name] = formatted_values

			# Fill NaN values with empty string
			result_df = result_df.fillna('')

			return result_df

		if acronym_convert:
			# First process property values BEFORE renaming columns
			if property_convert:
				for col in df.columns:
					if col not in exclude_columns:
						try:
							# Process each value individually like in the original function
							# This is less efficient but preserves the exact behavior
							converted_properties = {}
							for idx, value in df[col].items():
								# Use uppercase column name for consistency with S-57 acronyms

								converted_value = self.data.s57_properties_convert(
									col.upper(), value, prop_mixed=prop_mixed, debug=debug
								)
								converted_properties[idx] = converted_value
								# Create a new Series from converted properties
								converted_series = pd.Series(converted_properties)

								# Assign converted values back to DataFrame
								df[col] = converted_series

						except Exception as e:
							if debug:
								original_shape = df[col].shape
								value_sample = str(list(df[col].head(5).values))
								print(f"Error converting values in column '{col}': {e}")
								print(f"  - Column shape: {original_shape}")
								print(f"  - Column dtype: {df[col].dtype}")
								print(f"  - Sample values: {value_sample}")
								if hasattr(e, '__dict__'):
									print(f"  - Error details: {e.__dict__}")

			# Now rename columns after processing values
			converted_columns = {}
			for col in df.columns:
				if col not in exclude_columns:
					attr_name = self.data.s57_attributes_convert(col.upper())
					if attr_name:
						converted_columns[col] = f'{attr_name} ({col})'

			df = df.rename(columns=converted_columns)

		return df




#
#     def bbox_plot(self, enc_files, land_df, port_df, port_names=None, usage_bands=None, fig_w=15, fig_h=10):
#         # Get bounding boxes from ENC files
#         bbox = self.bbox(enc_files)
#
#         # Categorize ENC boundaries into usage bands
#         try:
#             if usage_bands:
#                 bbox_filtered = bbox[bbox['ENC_NAME'].str[2].astype(int).isin(usage_bands)]
#             else:
#                 bbox_filtered = bbox
#         except KeyError:
#             raise KeyError("The 'ENC_NAME' column is missing from the bounding box DataFrame.")
#
#         usage_band_dict = {
#             '1': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '1'], 'label': 'Overview ENCs',
#                   'color': 'blue'},
#             '2': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '2'], 'label': 'General ENCs',
#                   'color': 'green'},
#             '3': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '3'], 'label': 'Coastal ENCs',
#                   'color': 'orange'},
#             '4': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '4'], 'label': 'Approach ENCs',
#                   'color': 'red'},
#             '5': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '5'], 'label': 'Harbor ENCs',
#                   'color': 'purple'},
#             '6': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '6'], 'label': 'Berthing ENCs',
#                   'color': 'brown'}
#         }
#
#         # Convert port names to uppercase for case-insensitive matching
#         if port_names:
#             port_names = [name.upper() for name in port_names]
#             # Ensure 'PORT_NAME' column is uppercase for matching
#             port_df['PORT_NAME'] = port_df['PORT_NAME'].str.upper()
#             # Filter ports based on provided names
#             ports = port_df[port_df['PORT_NAME'].isin(port_names)]
#         else:
#             ports = port_df.copy()  # Use all ports if none specified
#
#         fig, ax = plt.subplots(figsize=(fig_w, fig_h))
#
#         # Plot land with label for legend
#         land_df.plot(ax=ax, color='lightgrey', label='Land')
#
#         # Plot ports if available
#         if not ports.empty:
#             ports.plot(ax=ax, color='navy', markersize=10, label='Ports', marker='o')
#             for idx, row in ports.iterrows():
#                 ax.annotate(row['PORT_NAME'], (row['geometry'].x, row['geometry'].y),
#                             xytext=(3, 0), textcoords='offset points', fontsize=5, ha='left')
#
#         def plot_enc_boundaries(df, ax, edgecolor='blue', linewidth=1, label=None, annotation_color='blue',
#                                 annotation_fontsize=5, annotation_text='ENC_NAME', annotation_position='lower_left',
#                                 annotation_offset=(3, 5), annotation_textcoords='offset points', annotation_ha='left',
#                                 annotation_va='bottom'):
#             """
#             Plots ENC boundary boxes and annotates each with its ENC name.
#
#             Parameters:
#             -----------
#             df : GeoDataFrame
#                 GeoDataFrame containing ENC boundaries with 'geometry' and 'ENC_NAME' columns.
#             ax : matplotlib.axes.Axes
#                 The axes to plot on.
#             edgecolor : color, optional
#                 Edge color of the boundary plots (default is 'blue').
#             linewidth : float, optional
#                 Line width of the boundary plots (default is 1).
#             label : str, optional
#                 Label for the boundaries (used in legends).
#             annotation_text : str, optional
#                 The column name in df to use for annotations (default is 'ENC_NAME').
#             annotation_position : {'lower_left', 'centroid'}, optional
#                 Position to place the annotation on each polygon (default is 'lower_left').
#             annotation_offset : tuple, optional
#                 Offset for the annotation text (default is (3, 5)).
#             annotation_textcoords : str, optional
#                 Coordinate system for the annotation text offset (default is 'offset points').
#             annotation_ha : str, optional
#                 Horizontal alignment of the annotation text (default is 'left').
#             annotation_va : str, optional
#                 Vertical alignment of the annotation text (default is 'bottom').
#             annotation_fontsize : int or float, optional
#                 Font size for the annotation text (default is 5).
#             annotation_color : color, optional
#                 Color for the annotation text (default is 'blue').
#             """
#             if not df.empty:
#                 # Plot ENC boundary boxes
#                 df.boundary.plot(ax=ax, edgecolor=edgecolor, linewidth=linewidth, label=label)
#
#                 # Annotate each ENC boundary box with its ENC name
#                 for idx, row in df.iterrows():
#                     polygon = row['geometry']
#                     # Determine annotation position
#                     if annotation_position == 'centroid':
#                         x, y = polygon.centroid.coords[0]
#                     elif annotation_position == 'lower_left':
#                         x, y = polygon.bounds[0], polygon.bounds[1]
#                     else:
#                         raise ValueError("Annotation_position must be 'centroid' or 'lower_left'")
#                     # Annotate
#                     ax.annotate(
#                         text=row[annotation_text],
#                         xy=(x, y),
#                         xytext=annotation_offset,
#                         textcoords=annotation_textcoords,
#                         ha=annotation_ha,
#                         va=annotation_va,
#                         fontsize=annotation_fontsize,
#                         color=annotation_color
#                     )
#
#         # Keep track of labels to prevent duplicate legend entries
#         plotted_labels = set()
#
#         if usage_bands:
#             for band in usage_bands:
#                 band_str = str(band)
#                 data = usage_band_dict.get(band_str)
#                 if data and not data['df'].empty:
#                     label = data['label'] if data['label'] not in plotted_labels else None
#                     plot_enc_boundaries(
#                         data['df'], ax,
#                         edgecolor=data['color'],
#                         linewidth=1,
#                         label=label,
#                         annotation_color=data['color'],
#                         annotation_fontsize=5
#                     )
#                     if label:
#                         plotted_labels.add(label)
#         else:
#             # Plot all ENC boundaries
#             plot_enc_boundaries(
#                 bbox, ax,
#                 edgecolor='blue',
#                 linewidth=1,
#                 label='ENC Boundaries',
#                 annotation_color='blue',
#                 annotation_fontsize=5
#             )
#
#         # Set plot limits to include all data
#         data_frames = [bbox_filtered.geometry]
#         if not ports.empty:
#             data_frames.append(ports.geometry)
#         combined = gpd.GeoSeries(pd.concat(data_frames, ignore_index=True))
#         total_bounds = combined.total_bounds
#         ax.set_xlim(total_bounds[0], total_bounds[2])
#         ax.set_ylim(total_bounds[1], total_bounds[3])
#
#         # Add title and labels
#         ax.set_title('ENC Boundary Boxes with Land and Ports')
#         ax.set_xlabel('Longitude')
#         ax.set_ylabel('Latitude')
#
#         # Add legend
#         handles, labels = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         ax.legend(by_label.values(), by_label.keys())
#
#         plt.tight_layout()
#         plt.show()
#
#     def bbox_plot_plotly(self, enc_files, land_df, port_df, port_names=None, usage_bands=None):
#
#         # Ensure all GeoDataFrames are in EPSG:4326 (latitude and longitude)
#         land_df = land_df.to_crs(epsg=4326)
#         bbox = self.bbox(enc_files).to_crs(epsg=4326)
#
#         # Categorize ENC boundaries into usage bands
#         usage_band_dict = {
#             1: {'df': bbox[bbox['ENC_NAME'].str[2] == '1'].copy(), 'label': 'Overview ENCs', 'color': 'blue'},
#             2: {'df': bbox[bbox['ENC_NAME'].str[2] == '2'].copy(), 'label': 'General ENCs', 'color': 'green'},
#             3: {'df': bbox[bbox['ENC_NAME'].str[2] == '3'].copy(), 'label': 'Coastal ENCs', 'color': 'orange'},
#             4: {'df': bbox[bbox['ENC_NAME'].str[2] == '4'].copy(), 'label': 'Approach ENCs', 'color': 'red'},
#             5: {'df': bbox[bbox['ENC_NAME'].str[2] == '5'].copy(), 'label': 'Harbor ENCs', 'color': 'purple'},
#             6: {'df': bbox[bbox['ENC_NAME'].str[2] == '6'].copy(), 'label': 'Berthing ENCs', 'color': 'brown'}
#         }
#
#         # Filter ports based on provided names
#         if port_names:
#             port_names = [name.upper() for name in port_names]
#             port_df['PORT_NAME'] = port_df['PORT_NAME'].str.upper()
#             ports = port_df[port_df['PORT_NAME'].isin(port_names)]
#         else:
#             ports = port_df  # Use all ports if none specified
#
#         if not ports.empty:
#             ports = ports.to_crs(epsg=4326)
#             ports['lon'] = ports.geometry.x
#             ports['lat'] = ports.geometry.y
#
#         # Initialize an empty figure
#         fig = go.Figure()
#
#         # Plot land polygons as a single trace
#         if not land_df.empty:
#             land_geojson = json.loads(land_df.to_json())
#             fig.add_trace(go.Choroplethmapbox(
#                 geojson=land_geojson,
#                 locations=land_df.index,
#                 z=[1] * len(land_df),
#                 colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
#                 showscale=False,
#                 marker_opacity=1,
#                 marker_line_width=0,
#                 name='Land',
#                 showlegend=True
#             ))
#
#         # Plot ports as a single trace
#         if not ports.empty:
#             fig.add_trace(go.Scattermapbox(
#                 lat=ports['lat'],
#                 lon=ports['lon'],
#                 mode='markers+text',
#                 marker=dict(
#                     size=10,
#                     color='red',
#                     symbol='circle'
#                 ),
#                 text=ports['PORT_NAME'],
#                 name='Ports',
#                 legendgroup='Ports',
#                 textposition='top right',
#                 hoverinfo='text',
#                 showlegend=True
#             ))
#
#         # Plot usage bands
#         if usage_bands:
#             for band in usage_bands:
#                 data = usage_band_dict.get(band)
#                 if data and not data['df'].empty:
#                     df = data['df']
#                     label = data['label']
#                     color = data['color']
#
#                     # Convert usage band geometries to GeoJSON
#                     usage_geojson = json.loads(df.to_json())
#
#                     fig.add_trace(go.Choroplethmapbox(
#                         geojson=usage_geojson,
#                         locations=df.index,
#                         z=[band] * len(df),
#                         colorscale=[[0, color], [1, color]],
#                         showscale=False,
#                         marker_opacity=0.5,
#                         marker_line_width=2,
#                         marker_line_color=color,
#                         name=label,
#                         showlegend=True,
#                         hovertemplate='<b>%{text}</b><extra></extra>',
#                         text=df['ENC_NAME']
#                     ))
#
#         # Update the layout
#         fig.update_layout(
#             mapbox_style="open-street-map",
#             mapbox=dict(
#                 center=dict(
#                     lat=ports['lat'].mean() if not ports.empty else 0,
#                     lon=ports['lon'].mean() if not ports.empty else 0
#                 ),
#                 zoom=2
#             ),
#             margin=dict(l=5, r=5, t=20, b=5, pad=10),
#             width=1280,
#             height=720,
#             showlegend=True,
#             legend=dict(
#                 title=dict(text='Legend', font=dict(size=14)),
#                 yanchor="top",
#                 y=0.99,
#                 xanchor="left",
#                 x=0.01
#             )
#         )
#
#         fig.show()
#
#     def load_enc(self, show_otdated=False):
#         """Find all ENC files with .gpkg extension in the specified folder."""
#         """Return a list of all the ENC files."""
#         self.enc_files = [file for file in os.listdir(self.folder_path) if file.endswith('.gpkg')]
#         if show_otdated:
#             outdated_encs = []
#             for file in self.enc_files:
#                 stamp = self.fiona_stamp(os.path.join(self.folder_path, file))
#
#                 # Check if the ENC is outdated by comparing with enc_db
#                 db_entry = self.enc_db[self.enc_db['ENC_Name'] == stamp['ENC_NAME']]
#                 if not db_entry.empty:
#                     db_edition = db_entry['Edition'].iloc[0]
#                     db_update = db_entry['Update'].iloc[0]
#                     if (stamp['ENC_EDITION'] < db_edition) or \
#                             (stamp['ENC_EDITION'] == db_edition and stamp['ENC_UPDATE'] < db_update):
#                         outdated_encs.append((file, stamp))
#
#             if outdated_encs:
#                 print("Outdated ENCs:")
#                 for file, stamp in outdated_encs:
#                     print(f"  - {file}: Edition {stamp['ENC_EDITION']}, Update {stamp['ENC_UPDATE']}")
#         else:
#             return self.enc_files
#
#     def usage_band(self):
#         """Sort ENC files by usage band."""
#         """Return: A dictionary with usage bands as keys and a list of filenames as values."""
#         for filename in self.enc_files:
#             usage_band = filename[2]
#             if usage_band in self.usage_bands:
#                 self.sorted_encs[self.usage_bands[usage_band]].append(filename)
#
#         # Add count to each usage band
#         enc_by_udage_band = {band: (files, len(files)) for band, files in self.sorted_encs.items()}
#         return enc_by_udage_band
#
#     def select_by_usage_band(self, usage_band):
#         """
#         Select ENC files by usage band and return them as a new list.
#
#         :param usage_band: String representing the usage band (e.g.,
#             1: 'Overview',
#             2: 'General',
#             3: 'Coastal',
#             4: 'Approach',
#             5: 'Harbor',
#             6: 'Berthing')
#
#         :return: List of ENC files for the specified usage band
#         """
#         if not self.sorted_encs:
#             self.usage_band()
#
#         return self.sorted_encs.get(usage_band, [])
#
#     def select_by_name(self, name_list):
#         """
#         Select ENC files based on a provided list of names.
#
#         :param name_list: List of ENC file names to select
#         :return: List of selected ENC files that exist in the folder
#         """
#         selected_encs = [file for file in self.enc_files if file in name_list]
#         return selected_encs
#
#     def select_by_port(self, port_names, port_df):
#         # Convert port names to uppercase for case-insensitive matching
#         port_names = [name.upper() for name in port_names]
#
#         # Ensure 'PORT_NAME' column is uppercase for matching
#         port_df['PORT_NAME'] = port_df['PORT_NAME'].str.upper()
#
#         # Filter ports based on provided names
#         selected_ports = port_df[port_df['PORT_NAME'].isin(port_names)]
#
#         # Get bounding boxes for all ENCs
#         enc_bbox = self.bbox(self.enc_files)
#
#         # Create rectangle boundary boxes for selected ports
#         port_bbox = selected_ports.copy()
#         port_bbox['geometry'] = port_bbox.apply(lambda row: row['geometry'].envelope, axis=1)
#
#         # Find intersections between port boundary boxes and ENC boundary boxes
#         intersecting_encs = []
#         for _, port in port_bbox.iterrows():
#             intersecting = enc_bbox[enc_bbox.intersects(port.geometry)]
#             intersecting_encs.extend(intersecting['ENC_NAME'].tolist())
#
#         # Remove duplicates
#         intersecting_encs = list(set(intersecting_encs))
#
#         # Filter ENC files based on intersecting ENCs
#         selected_encs = [enc for enc in self.enc_files if
#                          any(intersecting_enc in enc for intersecting_enc in intersecting_encs)]
#
#         return selected_encs
#
#     def select_by_port2(self, port_names, port_df):
#
#         # Convert port names to uppercase for case-insensitive matching
#         port_names = [name.upper() for name in port_names]
#
#         # Ensure 'PORT_NAME' column is uppercase for matching
#         port_df['PORT_NAME'] = port_df['PORT_NAME'].str.upper()
#
#         # Filter ports based on provided names
#         selected_ports = port_df[port_df['PORT_NAME'].isin(port_names)]
#
#         # Get bounding boxes for all ENCs
#         enc_bbox = self.bbox(self.enc_files)
#
#         # Ensure geometries are in the same CRS
#         if selected_ports.crs != enc_bbox.crs:
#             enc_bbox = enc_bbox.to_crs(selected_ports.crs)
#
#         # Initialize an empty list to store ENC names
#         intersecting_encs = []
#
#         # Spatial index for faster queries
#         enc_bbox_sindex = enc_bbox.sindex
#
#         # Iterate over each selected port
#         for _, port in selected_ports.iterrows():
#             port_geom = port.geometry
#
#             # Possible matches using spatial index
#             possible_matches_index = list(enc_bbox_sindex.intersection(port_geom.bounds))
#             possible_matches = enc_bbox.iloc[possible_matches_index]
#
#             # Precise matches using spatial predicates
#             intersects = possible_matches[possible_matches.intersects(port_geom)]
#             within = possible_matches[possible_matches.within(port_geom)]
#             contains = possible_matches[possible_matches.contains(port_geom)]
#
#             # Combine all matching ENCs
#             encs = pd.concat([intersects, within, contains]).drop_duplicates()
#
#             # Add ENC names to the list
#             intersecting_encs.extend(encs['ENC_NAME'].tolist())
#
#         # Remove duplicates
#         intersecting_encs = list(set(intersecting_encs))
#
#         # Filter ENC files based on intersecting ENCs
#         selected_encs = [enc for enc in self.enc_files if any(enc_name in enc for enc_name in intersecting_encs)]
#
#         return selected_encs
#
#     def read_dsid_layer(self, enc_files):
#         """
#         Read DSID Layer of single or multiple ENC files and return values with column names.
#
#         :param enc_files: Single ENC file name or list of ENC file names
#         :return: DataFrame containing DSID Layer information
#         """
#         if isinstance(enc_files, str):
#             enc_files = [enc_files]
#
#         dsid_data = []
#         for file in enc_files:
#             file_path = os.path.join(self.folder_path, file)
#             gdf = gpd.read_file(file_path, layer='DSID')
#             dsid_data.append(gdf)
#
#         combined_dsid = pd.concat(dsid_data, ignore_index=True)
#         return combined_dsid
#
#     def layers(self, enc_files, unique=False, acronym_convert=False):
#
#         if isinstance(enc_files, str):
#             enc_files = [enc_files]  # Convert single path to list
#         print(f"Reading {len(enc_files)} ENC files...")
#
#         all_layers = set()  # Use a set to store unique layers
#
#         for enc in enc_files:
#             # List all layers in the GeoPackage
#             full_path = os.path.join(self.folder_path, enc)
#             layers = fiona.listlayers(full_path)
#
#             if not unique:
#                 print(f"\nReading {enc}...")
#                 print(f"ENC file {enc} contains the following layers:")
#             sorted_layers = []
#             for layer in layers:
#                 sorted_layers.append(layer)
#                 all_layers.add(layer)
#
#             if not unique:
#                 # Sort the list alphabetically by obj_name
#                 sorted_layers.sort(key=lambda x: (x[0] or "").lower())
#
#                 # Print the sorted layers
#                 for layer in sorted_layers:
#                     if acronym_convert:
#                         obj_name = self.data.s57_objects_convert(layer)
#                         print(f"  - {obj_name} ({layer})")
#                     else:
#                         print(f"  - {layer}")
#
#         if unique:
#             print("\nUnique layers across all ENC files:")
#             sorted_unique_layers = sorted(all_layers, key=lambda x: (x[0] or "").lower())
#             for layer in sorted_unique_layers:
#                 if acronym_convert:
#                     obj_name = self.data.s57_objects_convert(layer)
#                     print(f"  - {obj_name} ({layer})")
#                 else:
#                     print(f"  - {layer}")
#
#     def layers_properties(self, enc_name, layer_names=None):
#         """
#         Opens an ENC and returns specified layers with all features.
#
#         :param enc_name: Name of the ENC file to open
#         :param layer_names: List of layer names to return. If None, returns all layers.
#         :return: Dictionary with layer names as keys and features as values
#         """
#         enc_path = os.path.join(self.folder_path, enc_name)
#
#         if not os.path.exists(enc_path):
#             raise FileNotFoundError(f"ENC file {enc_name} not found in the specified folder.")
#
#         if layer_names:
#             # Load specific layer
#             gdf = gpd.read_file(enc_path, layer=layer_names)
#
#             return gdf
#         else:
#             # Load all layers
#             layers = fiona.listlayers(enc_path)
#             layer_data = {}
#             for layer in layers:
#                 gdf = gpd.read_file(enc_path, layer=layer)
#                 layer_data[layer] = gdf
#
#             return layer_data
#
#     def get_properties(self, enc_name, layer_name):
#         result = gpd.GeoDataFrame()
#
#         # Ensure enc_name and layer_name are lists
#         enc_names = [enc_name] if isinstance(enc_name, str) else enc_name
#         layer_names = [layer_name] if isinstance(layer_name, str) else layer_name
#
#         for enc in enc_names:
#             enc_path = os.path.join(self.folder_path, enc)
#             if not os.path.exists(enc_path):
#                 print(f"ENC file {enc} not found.")
#                 continue
#
#             stamp = self.fiona_stamp(enc_path)
#
#             for layer in layer_names:
#                 try:
#                     gdf = gpd.read_file(enc_path, layer=layer)
#                     gdf['ENC_NAME'] = stamp['ENC_NAME']
#                     gdf['ENC_EDITION'] = stamp['ENC_EDITION']
#                     gdf['ENC_UPDATE'] = stamp['ENC_UPDATE']
#                     gdf['LAYER_NAME'] = layer
#                     result = pd.concat([result, gdf], ignore_index=True)
#                 except Exception as e:
#                     print(f"Error reading layer {layer} from {enc}: {str(e)}")
#
#         return result

# class ENC:
#
# 	def inspect_layer(self, dataframe):
# 		data = Data()
# 		df = dataframe.copy()
#
# 		# Create copy and drop geometry column if exists
# 		if 'wkb_geometry' in df.columns:
# 			df = df.drop(columns=['wkb_geometry'])
#
# 		# Rename columns with attribute names
# 		for column in df.columns.tolist():
# 			attr_name = data.s57_attributes_convert(column.upper())
# 			if attr_name:
# 				df = df.rename(columns={column: f'{attr_name} ({column})'})
#
# 		# Return transposed DataFrame
# 		return df
#
#
# 	def bbox_plot(self, enc_files, land_df, port_df, port_names=None, usage_bands=None, fig_w=15, fig_h=10):
#
# 	# Get bounding boxes from ENC files
# 		bbox = self.bbox(enc_files)
#
# 		# Categorize ENC boundaries into usage bands
# 		try:
# 			if usage_bands:
# 				bbox_filtered = bbox[bbox['ENC_NAME'].str[2].astype(int).isin(usage_bands)]
# 			else:
# 				bbox_filtered = bbox
# 		except KeyError:
# 			raise KeyError("The 'ENC_NAME' column is missing from the bounding box DataFrame.")
#
# 		usage_band_dict = {
# 			'1': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '1'], 'label': 'Overview ENCs',
# 				  'color': 'blue'},
# 			'2': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '2'], 'label': 'General ENCs',
# 				  'color': 'green'},
# 			'3': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '3'], 'label': 'Coastal ENCs',
# 				  'color': 'orange'},
# 			'4': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '4'], 'label': 'Approach ENCs',
# 				  'color': 'red'},
# 			'5': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '5'], 'label': 'Harbor ENCs',
# 				  'color': 'purple'},
# 			'6': {'df': bbox_filtered[bbox_filtered['ENC_NAME'].str[2] == '6'], 'label': 'Berthing ENCs',
# 				  'color': 'brown'}
# 		}
#
# 		# Convert port names to uppercase for case-insensitive matching
# 		if port_names:
# 			port_names = [name.upper() for name in port_names]
# 			# Ensure 'PORT_NAME' column is uppercase for matching
# 			port_df['PORT_NAME'] = port_df['PORT_NAME'].str.upper()
# 			# Filter ports based on provided names
# 			ports = port_df[port_df['PORT_NAME'].isin(port_names)]
# 		else:
# 			ports = port_df.copy()  # Use all ports if none specified
#
# 		fig, ax = plt.subplots(figsize=(fig_w, fig_h))
#
# 		# Plot land with label for legend
# 		land_df.plot(ax=ax, color='lightgrey', label='Land')
#
# 		# Plot ports if available
# 		if not ports.empty:
# 			ports.plot(ax=ax, color='navy', markersize=10, label='Ports', marker='o')
# 			for idx, row in ports.iterrows():
# 				ax.annotate(row['PORT_NAME'], (row['geometry'].x, row['geometry'].y),
# 							xytext=(3, 0), textcoords='offset points', fontsize=5, ha='left')
#
# 		def plot_enc_boundaries(df, ax, edgecolor='blue', linewidth=1, label=None, annotation_color='blue',
# 								annotation_fontsize=5, annotation_text='ENC_NAME', annotation_position='lower_left',
# 								annotation_offset=(3, 5), annotation_textcoords='offset points',
# 								annotation_ha='left',
# 								annotation_va='bottom'):
# 			"""
# 			Plots ENC boundary boxes and annotates each with its ENC name.
#
# 			Parameters:
# 			-----------
# 			df : GeoDataFrame
# 				GeoDataFrame containing ENC boundaries with 'geometry' and 'ENC_NAME' columns.
# 			ax : matplotlib.axes.Axes
# 				The axes to plot on.
# 			edgecolor : color, optional
# 				Edge color of the boundary plots (default is 'blue').
# 			linewidth : float, optional
# 				Line width of the boundary plots (default is 1).
# 			label : str, optional
# 				Label for the boundaries (used in legends).
# 			annotation_text : str, optional
# 				The column name in df to use for annotations (default is 'ENC_NAME').
# 			annotation_position : {'lower_left', 'centroid'}, optional
# 				Position to place the annotation on each polygon (default is 'lower_left').
# 			annotation_offset : tuple, optional
# 				Offset for the annotation text (default is (3, 5)).
# 			annotation_textcoords : str, optional
# 				Coordinate system for the annotation text offset (default is 'offset points').
# 			annotation_ha : str, optional
# 				Horizontal alignment of the annotation text (default is 'left').
# 			annotation_va : str, optional
# 				Vertical alignment of the annotation text (default is 'bottom').
# 			annotation_fontsize : int or float, optional
# 				Font size for the annotation text (default is 5).
# 			annotation_color : color, optional
# 				Color for the annotation text (default is 'blue').
# 			"""
# 			if not df.empty:
# 				# Plot ENC boundary boxes
# 				df.boundary.plot(ax=ax, edgecolor=edgecolor, linewidth=linewidth, label=label)
#
# 				# Annotate each ENC boundary box with its ENC name
# 				for idx, row in df.iterrows():
# 					polygon = row['geometry']
# 					# Determine annotation position
# 					if annotation_position == 'centroid':
# 						x, y = polygon.centroid.coords[0]
# 					elif annotation_position == 'lower_left':
# 						x, y = polygon.bounds[0], polygon.bounds[1]
# 					else:
# 						raise ValueError("Annotation_position must be 'centroid' or 'lower_left'")
# 					# Annotate
# 					ax.annotate(
# 						text=row[annotation_text],
# 						xy=(x, y),
# 						xytext=annotation_offset,
# 						textcoords=annotation_textcoords,
# 						ha=annotation_ha,
# 						va=annotation_va,
# 						fontsize=annotation_fontsize,
# 						color=annotation_color
# 					)
#
# 		# Keep track of labels to prevent duplicate legend entries
# 		plotted_labels = set()
#
# 		if usage_bands:
# 			for band in usage_bands:
# 				band_str = str(band)
# 				data = usage_band_dict.get(band_str)
# 				if data and not data['df'].empty:
# 					label = data['label'] if data['label'] not in plotted_labels else None
# 					plot_enc_boundaries(
# 						data['df'], ax,
# 						edgecolor=data['color'],
# 						linewidth=1,
# 						label=label,
# 						annotation_color=data['color'],
# 						annotation_fontsize=5
# 					)
# 					if label:
# 						plotted_labels.add(label)
# 		else:
# 			# Plot all ENC boundaries
# 			plot_enc_boundaries(
# 				bbox, ax,
# 				edgecolor='blue',
# 				linewidth=1,
# 				label='ENC Boundaries',
# 				annotation_color='blue',
# 				annotation_fontsize=5
# 			)
#
# 		# Set plot limits to include all data
# 		data_frames = [bbox_filtered.geometry]
# 		if not ports.empty:
# 			data_frames.append(ports.geometry)
# 		combined = gpd.GeoSeries(pd.concat(data_frames, ignore_index=True))
# 		total_bounds = combined.total_bounds
# 		ax.set_xlim(total_bounds[0], total_bounds[2])
# 		ax.set_ylim(total_bounds[1], total_bounds[3])
#
# 		# Add title and labels
# 		ax.set_title('ENC Boundary Boxes with Land and Ports')
# 		ax.set_xlabel('Longitude')
# 		ax.set_ylabel('Latitude')
#
# 		# Add legend
# 		handles, labels = ax.get_legend_handles_labels()
# 		by_label = dict(zip(labels, handles))
# 		ax.legend(by_label.values(), by_label.keys())
#
# 		plt.tight_layout()
# 		plt.show()
#
# 	def bbox_plot_plotly(self, enc_files, land_df, port_df, port_names=None, usage_bands=None):
#
# 		# Ensure all GeoDataFrames are in EPSG:4326 (latitude and longitude)
# 		land_df = land_df.to_crs(epsg=4326)
# 		bbox = self.bbox(enc_files).to_crs(epsg=4326)
#
# 		# Categorize ENC boundaries into usage bands
# 		usage_band_dict = {
# 			1: {'df': bbox[bbox['ENC_NAME'].str[2] == '1'].copy(), 'label': 'Overview ENCs', 'color': 'blue'},
# 			2: {'df': bbox[bbox['ENC_NAME'].str[2] == '2'].copy(), 'label': 'General ENCs', 'color': 'green'},
# 			3: {'df': bbox[bbox['ENC_NAME'].str[2] == '3'].copy(), 'label': 'Coastal ENCs', 'color': 'orange'},
# 			4: {'df': bbox[bbox['ENC_NAME'].str[2] == '4'].copy(), 'label': 'Approach ENCs', 'color': 'red'},
# 			5: {'df': bbox[bbox['ENC_NAME'].str[2] == '5'].copy(), 'label': 'Harbor ENCs', 'color': 'purple'},
# 			6: {'df': bbox[bbox['ENC_NAME'].str[2] == '6'].copy(), 'label': 'Berthing ENCs', 'color': 'brown'}
# 		}
#
# 		# Filter ports based on provided names
# 		if port_names:
# 			port_names = [name.upper() for name in port_names]
# 			port_df['PORT_NAME'] = port_df['PORT_NAME'].str.upper()
# 			ports = port_df[port_df['PORT_NAME'].isin(port_names)]
# 		else:
# 			ports = port_df  # Use all ports if none specified
#
# 		if not ports.empty:
# 			ports = ports.to_crs(epsg=4326)
# 			ports['lon'] = ports.geometry.x
# 			ports['lat'] = ports.geometry.y
#
# 		# Initialize an empty figure
# 		fig = go.Figure()
#
# 		# Plot land polygons as a single trace
# 		if not land_df.empty:
# 			land_geojson = json.loads(land_df.to_json())
# 			fig.add_trace(go.Choroplethmapbox(
# 				geojson=land_geojson,
# 				locations=land_df.index,
# 				z=[1] * len(land_df),
# 				colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
# 				showscale=False,
# 				marker_opacity=1,
# 				marker_line_width=0,
# 				name='Land',
# 				showlegend=True
# 			))
#
# 		# Plot ports as a single trace
# 		if not ports.empty:
# 			fig.add_trace(go.Scattermapbox(
# 				lat=ports['lat'],
# 				lon=ports['lon'],
# 				mode='markers+text',
# 				marker=dict(
# 					size=10,
# 					color='red',
# 					symbol='circle'
# 				),
# 				text=ports['PORT_NAME'],
# 				name='Ports',
# 				legendgroup='Ports',
# 				textposition='top right',
# 				hoverinfo='text',
# 				showlegend=True
# 			))
#
# 		# Plot usage bands
# 		if usage_bands:
# 			for band in usage_bands:
# 				data = usage_band_dict.get(band)
# 				if data and not data['df'].empty:
# 					df = data['df']
# 					label = data['label']
# 					color = data['color']
#
# 					# Convert usage band geometries to GeoJSON
# 					usage_geojson = json.loads(df.to_json())
#
# 					fig.add_trace(go.Choroplethmapbox(
# 						geojson=usage_geojson,
# 						locations=df.index,
# 						z=[band] * len(df),
# 						colorscale=[[0, color], [1, color]],
# 						showscale=False,
# 						marker_opacity=0.5,
# 						marker_line_width=2,
# 						marker_line_color=color,
# 						name=label,
# 						showlegend=True,
# 						hovertemplate='<b>%{text}</b><extra></extra>',
# 						text=df['ENC_NAME']
# 					))
#
# 		# Update the layout
# 		fig.update_layout(
# 			mapbox_style="open-street-map",
# 			mapbox=dict(
# 				center=dict(
# 					lat=ports['lat'].mean() if not ports.empty else 0,
# 					lon=ports['lon'].mean() if not ports.empty else 0
# 				),
# 				zoom=2
# 			),
# 			margin=dict(l=5, r=5, t=20, b=5, pad=10),
# 			width=1280,
# 			height=720,
# 			showlegend=True,
# 			legend=dict(
# 				title=dict(text='Legend', font=dict(size=14)),
# 				yanchor="top",
# 				y=0.99,
# 				xanchor="left",
# 				x=0.01
# 			)
# 		)
#
# 		fig.show()

class VISUALIZATION:
	def __init__(self):
		self.misc = Miscellaneous()
		self.enc_labels = [
			'Overview ENCs',
			'General ENCs',
			'Coastal ENCs',
			'Approach ENCs',
			'Harbor ENCs',
			'Berthing ENCs'
		]

	# Function to find trace index by name
	@staticmethod
	def get_trace_list(fig, group=False):
		"""
			Returns list of tuples containing (index, name) for each trace in figure
			Args:
				fig: Plotly figure object
			Returns:
				List of tuples [(index, name), ...]
			"""
		trace_list = []
		excluded_names = [None, "Equator", "Meridian"]

		for idx, trace in enumerate(fig.data):
			if trace.name not in excluded_names:
				if group:
					trace_list.append((idx, trace.name, trace.legendgroup))
				else:
					trace_list.append((idx, trace.name))
		return trace_list

	def get_trace_by_name(self, fig, trace_name:str):
		trace_list = self.get_trace_list(fig)
		for idx, name in trace_list:
			if name == trace_name:
				print(f"Found trace with name '{trace_name}' at index {idx}")
				return idx
		print(f"Error: Trace with name '{trace_name}' not found.")
		return None

	def get_trace_item_by_name(self, fig, trace_name:str, param_name: str):
		trace_list = self.get_trace_list(fig)
		for idx, name in trace_list:
			if name == trace_name:
				print(f"Found trace with name '{trace_name}' at index {idx}")
				try:
					return fig.data[idx].__getattribute__(param_name)
				except AttributeError:
					print(f"Error: Trace '{trace_name}' does not have attribute '{param_name}'.")
					return None

		print(f"Error: Trace with name '{trace_name}' not found.")
		return None


	def remove_trace(self, fig, trace_name):
		"""
		Removes trace from figure based on trace name
		Args:
		   fig: Plotly figure object
		   trace_name: Name of the trace to be removed
		"""
		trace_index = self.get_trace_list(fig)
		indices_to_remove = []

		for idx, name in trace_index:
			if name == trace_name:
				indices_to_remove.append(idx)

		# Remove traces from highest index to lowest to maintain correct indexing
		for idx in sorted(indices_to_remove, reverse=True):
			temp_data = list(fig.data)  # Convert to a list
			temp_data.pop(idx)  # Remove the second trace
			fig.data = tuple(temp_data)

	def trace_enc_bbox_changes(self, base_gdf: gpd.GeoDataFrame, new_enc_list: list, old_enc_list: list) -> dict:
		"""
		Traces changes between old and new ENC lists, categorizing by usage bands.

		Args:
			base_gdf: GeoDataFrame containing all ENC information
			new_enc_list: List of new ENC names
			old_enc_list: List of old ENC names

		Returns:
			dict: Changes categorized by usage band containing:
				- added: {usage_band: [(index, enc_name), ...]}
				- removed: {usage_band: [(index, enc_name), ...]}
		"""
		# standardize ENC_NAME column
		base_gdf = self.misc._standardize_enc_name_column(base_gdf)

		changes = {}

		# Map usage band numbers to names
		band_names = {
			'1': 'Overview',
			'2': 'General',
			'3': 'Coastal',
			'4': 'Approach',
			'5': 'Harbour',
			'6': 'Berthing'
		}

		# Find added ENCs
		added = set(new_enc_list) - set(old_enc_list)
		added_changes = {}
		for enc in added:
			if enc in base_gdf['ENC_NAME'].values:
				idx = base_gdf[base_gdf['ENC_NAME'] == enc].index[0]
				usage_band = band_names[enc[2]]
				if usage_band not in added_changes:
					added_changes[usage_band] = []
				added_changes[usage_band].append((idx, enc))

		if added_changes:
			changes['added'] = added_changes

		# Find removed ENCs
		removed = set(old_enc_list) - set(new_enc_list)
		removed_changes = {}
		for enc in removed:
			if enc in base_gdf['ENC_NAME'].values:
				idx = base_gdf[base_gdf['ENC_NAME'] == enc].index[0]
				usage_band = band_names[enc[2]]
				if usage_band not in removed_changes:
					removed_changes[usage_band] = []
				removed_changes[usage_band].append((idx, enc))

		if removed_changes:
			changes['removed'] = removed_changes

		return changes

	def plotly_base_config(self, figure) -> dict:
		plotly_config={
		  'displayModeBar': 'hover',  # Show mode bar on hover
		  'responsive': True,  # Make the chart responsive
		  'scrollZoom': True,  # Enable scroll to zoom
		  'displaylogo': False,
		  'modeBarButtonsToRemove': ['zoomIn', 'zoomOut', 'pan', 'select', 'lassoSelect'],
		  'modeBarButtonsToAdd': ['autoScale', 'hoverCompareCartesian']
		}
		return plotly_config

	def plotly_dinamic_zoom(self, distance):
		"""
		Convert distance in kilometers to a Plotly zoom level.
		This is a heuristic and may need adjustment based on specific requirements.
		"""
		if distance < 50:
			return 10  # City level
		elif distance < 500:
			return 6  # Regional level
		elif distance < 2000:
			return 4  # Country level
		else:
			return 2  # World level

	def set_zoom_to(self, figure, geometry, zoom_level:int = 7):
		"""
		Sets the zoom level of the map based on the provided geometry’s centroid.
		This function accepts Shapely geometries (e.g., Polygon, MultiPolygon, LineString)
		as well as a WKT string.

		Parameters:
			figure: Plotly figure object.
			geometry: A Shapely geometry or a WKT string.
			zoom_level (int): The desired zoom level.
		"""
		# If a string is provided, convert it to a Shapely geometry.
		if isinstance(geometry, str):
			geometry = wkt.loads(geometry)

		# Use the centroid of the geometry if available; works for polygons and linestrings.
		if hasattr(geometry, "centroid"):
			center = geometry.centroid
		else:
			center = geometry

		figure.update_layout(
			mapbox=dict(
				center=dict(
					lat=geometry.y,
					lon=geometry.x
				),
				zoom=zoom_level,
				bearing=0
			)
		)

	def update_legend(self, figure, traceorder='normal', legend_text = "Legend", font_size = 16, font_color = "black"):

		figure.update_layout(
			legend = dict(
				traceorder=traceorder,
				tracegroupgap=1,
				title=dict(
					text=legend_text,
					font=dict(
						family='Arial, sans-serif',
						size=font_size,
						color=font_color
					)
				),
				font=dict(
					family="Calibri",
					size=font_size,
					color=font_color
				),
				indentation=5,
				yanchor="top",
				xanchor="left",
				y=0.99,
				x=0.005,
				bgcolor="#889ec8",
				bordercolor="#ededed",
				borderwidth=1
			)
		)

	def set_trace_visibility(self, figure, trace_name: str) -> None:
		"""
		Toggles the visibility of a trace in a Plotly figure based on its trace name.
		If the trace is currently visible, it will be hidden (set to "legendonly"), and if
		it is hidden it will be shown (set to True).

		Args:
			figure: Plotly figure object.
			trace_name: Name of the trace to toggle visibility.
		"""
		for trace in figure.data:
			if trace.name == trace_name:
				if trace.visible not in [False, "legendonly"]:
					trace.visible = "legendonly"
				else:
					trace.visible = True

	def create_base_map(self, title: str ="Global map", mapbox_token: str  = "MAP_BOX_PUBLIC_TOKEN",
	                    layout_flex: bool = True,
	                    height_px: int = 1080,
	                    width_px: int = 1920,
	                    plot_type: str = "mapbox") -> go.Figure:
		"""
        Creates a base map that supports both Scattergl (for performance with large datasets)
        and Mapbox (for street maps) visualization.

        Args:
            title: Title of the map
            mapbox_token: Mapbox token for street map access
            layout_flex: Whether to use flexible layout or fixed dimensions
            height_px: Height in pixels (for fixed layout)
            width_px: Width in pixels (for fixed layout)
            plot_type: Type of plot to use ("mapbox" or "scattergl")

        Returns:
            go.Figure: A Plotly figure object configured for the chosen plot type
        """

		# Add top margin if title is Provided and larger then 2 characters
		if len(title) >= 2:
			layout_margin = dict(l=0, r=0, t=30, b=0)
		else:
			layout_margin = dict(l=0, r=0, t=0, b=0)

		# make plot flexible or fixed size
		if layout_flex:
			layout_size = dict(autosize = True)
		else:
			layout_size =  dict(height=height_px, width=width_px)


		figure = go.Figure()
		# Configure based on plot type
		if plot_type.lower() == "mapbox":
			figure.update_layout(
				layout_size,
				height=height_px,
				mapbox=dict(
					style='mapbox://styles/vikont/cm4yf9ahl005d01sfanlib2f6',  # Vector tile style
					accesstoken = mapbox_token,
					center=dict(
						lat=0,
						lon=0
					),
					zoom=2.0,
					bearing = 0,
				),
				title_text = title,
				title_x=0.5,
				template='plotly_white',
				legend=dict(
					orientation="v",
					indentation=5,
					yanchor="top",
					xanchor="left",
					y=0.99,
					x=0.005,
					bgcolor="#889ec8",
					bordercolor="#ededed",
					borderwidth=1
				),

				#autosize=True,
				#height=1080,
				#width=1920,
				margin=layout_margin,
			)
		elif plot_type.lower() == "scattergl":
			# Scattergl-based configuration for large datasets
			figure.update_layout(
				**layout_size,
				height=height_px,
				geo=dict(
					showland=True,
					landcolor="rgb(212, 212, 212)",
					showocean=True,
					oceancolor="rgb(220, 240, 255)",
					showcountries=True,
					countrycolor="rgb(180, 180, 180)",
					showcoastlines=True,
					coastlinecolor="rgb(180, 180, 180)",
					projection_type="equirectangular"
				),
				title_text=title,
				title_x=0.5,
				template='plotly_white',
				legend=dict(
					orientation="v",
					indentation=5,
					yanchor="top",
					xanchor="left",
					y=0.99,
					x=0.005,
					bgcolor="#889ec8",
					bordercolor="#ededed",
					borderwidth=1
				),
				margin=layout_margin,

			)
		else:
			raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'mapbox' or 'scattergl'.")

		return figure

	def add_geo_grids(self, figure ) :
		# Define bounds (modify as needed)
		lon_min, lon_max = -180, 180
		lat_min, lat_max = -90, 90

		# Define grid spacing
		lon_spacing = 10  # degrees
		lat_spacing = 10  # degrees

		# Generate arrays of longitudes and latitudes for the grid lines.
		lons = np.arange(lon_min, lon_max + lon_spacing, lon_spacing)
		lats = np.arange(lat_min, lat_max + lat_spacing, lat_spacing)

		grid_traces = []

		# Create horizontal grid lines (lines of constant latitude)
		for lat in lats:
			grid_traces.append(
				go.Scattermapbox(
					lon=[lon_min, lon_max],
					lat=[lat, lat],
					mode='lines',
					line=dict(color='gray', width=1),
					hoverinfo='none',
					showlegend=False
				)
			)

		# Create vertical grid lines (lines of constant longitude)
		for lon in lons:
			grid_traces.append(
				go.Scattermapbox(
					lon=[lon, lon],
					lat=[lat_min, lat_max],
					mode='lines',
					line=dict(color='gray', width=1),
					hoverinfo='none',
					showlegend=False
				)
			)

		# Add the grid traces to the figure
		for trace in grid_traces:
			figure.add_trace(trace)

		# Create a black Equator trace with thicker line and hover text
		equator_trace = go.Scattermapbox(
			lon=[lon_min, lon_max],
			lat=[0, 0],
			mode='lines',
			line=dict(color='black', width=2),
			name='Equator',
			text=['Equator'],
			hoverinfo='text',
			showlegend = False
		)
		figure.add_trace(equator_trace)

		# Create a black Prime Meridian trace with thicker line and hover text
		prime_meridian_trace = go.Scattermapbox(
			lon=[0, 0],
			lat=[lat_min, lat_max],
			mode='lines',
			line=dict(color='black', width=2),
			name='Meridian',
			text=['Meridian'],
			hoverinfo='text',
			showlegend = False
		)
		figure.add_trace(prime_meridian_trace)


	def add_ports_trace(self, figure, port_df, name = "Ports",color:str = 'black', show_legend=True):
		"""Adds port locations as scatter points to the base map"""
		port_sizes = {
			'L': 12,
			'M': 8,
			'S': 4,
			'nan': 2
		}
		port_size_values = port_df['HARBORSIZE'].map(port_sizes).fillna(2)

		figure.add_trace(
			go.Scattermapbox(
				lon=port_df['LONGITUDE'],
				lat=port_df['LATITUDE'],
				text=port_df['PORT_NAME'],
				mode='markers',
				marker=dict(
					size=port_size_values,
					color=color,
					symbol='circle'
				),
				hoverinfo='text',
				name=name,
				showlegend=show_legend
			)
		)
		return figure

	def add_single_port_trace(self, figure, port_series, name="Ports", leg_group = "Ports", leg_title = "", color: str = 'black', show_legend=True):
		"""
		Adds a single port location as scatter point to the base map

		Args:
			figure: Base plotly figure to add trace to
			port_series: Series containing single port data with HARBORSIZE, LONGITUDE, LATITUDE, PORT_NAME
			name: Name for the trace legend
			color: Marker color
			show_legend: Whether to show in legend
		"""
		port_sizes = {
			'L': 12,
			'M': 8,
			'S': 4,
			'nan': 2
		}
		size = port_sizes.get(port_series['HARBORSIZE'], 2)

		figure.add_trace(
			go.Scattermapbox(
				lon=[port_series['LONGITUDE']],
				lat=[port_series['LATITUDE']],
				text=[port_series['PORT_NAME']],
				mode='markers',
				marker=dict(
					size=14,
					color=color,
					symbol='circle'
				),
				hoverinfo='text',
				legendgroup=leg_group,  # this can be any string, not just "group"
				legendgrouptitle_text=leg_title,
				name=name,
				showlegend=show_legend
			)
		)
		return figure

	def add_node_trace(self, figure, nodes_data):
		"""
		Creates a Plotly Scattermapbox trace for graph nodes.

		Parameters:
		  nodes_data: List of rows containing node and geojson geometry.

		Returns:
		  go.Scattermapbox trace for the nodes.
		"""
		node_lon, node_lat, node_text = [], [], []
		for row in nodes_data:
			try:
				geom_json = json.loads(row[2])
				point = shape(geom_json)
				node_lon.append(point.x)
				node_lat.append(point.y)
				node_text.append(f"Node ID: {row[0]}<br>Lat: {point.y:.6f}<br>Lon: {point.x:.6f}")
			except Exception as e:
				print("Error processing node:", e)

		figure.add_trace(go.Scattermapbox(
			lon=node_lon,
			lat=node_lat,
			mode='markers',
			marker=dict(size=8, color='red'),
			text=node_text,
			hoverinfo='text',
			name='Nodes'
		))
		return figure

	def add_edge_trace(self, figure, edges_data):
		"""
		Creates a Plotly Scattermapbox trace for graph edges.

		Parameters:
		  edges_data: List of rows containing edge information and geojson geometry.

		Returns:
		  go.Scattermapbox trace for the edges.
		"""
		edge_lon, edge_lat = [], []
		for row in edges_data:
			try:
				geom_json = json.loads(row[3])
				line = shape(geom_json)
				coords = list(line.coords)
				for coord in coords:
					edge_lon.append(coord[0])
					edge_lat.append(coord[1])
				# Separate edges by inserting a None value.
				edge_lon.append(None)
				edge_lat.append(None)
			except Exception as e:
				print("Error processing edge:", e)


		figure.add_trace(go.Scattermapbox(
			lon=edge_lon,
			lat=edge_lat,
			mode='lines',
			line=dict(color='blue', width=2),
			hoverinfo='none',
			name='Edges'
		))
		return figure

	def add_edge_trace_batch(self, figure, edges_data, batch_size=400000, line_color='blue', line_width=2,
	                   name='Edges', hover_info='none', show_legend_first_only=True):
		"""
		Creates Plotly Scattermapbox traces for graph edges, processing in batches to reduce rendering load.

		Parameters:
		  figure: Plotly figure object to add the traces to
		  edges_data: List of rows containing edge information and geojson geometry
		  batch_size: Number of edges to include in each batch/trace (default: 1000)
		  line_color: Color of the edge lines
		  line_width: Width of the edge lines
		  name: Base name for the edge traces
		  hover_info: Hover information display mode ('none', 'text', etc.)
		  show_legend_first_only: If True, only the first batch shows in legend

		Returns:
		  Updated figure with added edge traces
		"""
		if not edges_data:
			return figure

		# Initialize variables
		batch_count = 0
		total_edges = len(edges_data)

		# Process edges in batches
		for batch_start in range(0, total_edges, batch_size):
			batch_end = min(batch_start + batch_size, total_edges)
			current_batch = edges_data[batch_start:batch_end]
			batch_count += 1

			# Initialize coordinate lists for this batch
			edge_lon, edge_lat = [], []

			# Process edges in current batch
			for row in current_batch:
				try:
					geom_json = json.loads(row[3])
					line = shape(geom_json)
					coords = list(line.coords)
					for coord in coords:
						edge_lon.append(coord[0])
						edge_lat.append(coord[1])
					# Separate edges by inserting a None value
					edge_lon.append(None)
					edge_lat.append(None)
				except Exception as e:
					print(f"Error processing edge: {e}")
					continue

			# Skip empty batches
			if not edge_lon:
				continue

			# Determine if this batch should appear in legend
			show_in_legend = True if batch_count == 1 or not show_legend_first_only else False
			trace_name = name if batch_count == 1 else f"{name} (Batch {batch_count})"

			# Create trace for this batch
			figure.add_trace(go.Scattermapbox(
				lon=edge_lon,
				lat=edge_lat,
				mode='lines',
				line=dict(color=line_color, width=line_width),
				hoverinfo=hover_info,
				name=trace_name,
				showlegend=show_in_legend,
				legendgroup=name  # Group all batches under same legend entry
			))

		return figure

	def add_weighted_edge_trace(self, figure, edges_data, weight_column_index=None, weight_attribute=None,
	                            edge_width=1, show_weights=True, weight_thresholds=None, colors=None):
		"""
		Adds colored edge traces to a Plotly figure with distinct colors for different
		weight categories (preferred, safe, caution, unsafe).

		Parameters:
			figure: Plotly figure object to add trace to
			edges_data: List of edge data rows (either PostGIS format or NetworkX G.edges format)
			weight_column_index: Index of the weight column in PostgreSQL edge data rows
			weight_attribute: Name of the weight attribute when using NetworkX edge data
			edge_width: Width of edge lines
			show_weights: Whether to show weight values in hover info
			weight_thresholds: Dict with threshold values for categories (optional)
						   e.g. {'preferred': 0.8, 'safe': 1.2, 'caution': 2.5}
			colors: Dict with colors for each category (optional)
				   e.g. {'preferred': 'green', 'safe': 'blue', 'caution': 'orange', 'unsafe': 'red'}

		Returns:
			The figure object with added categorical edge traces
		"""
		# Define default thresholds and colors if not provided
		base_weight = 0.006
		pr_val = 0.9 * base_weight
		saf_val = 1.5 * base_weight
		cautn_val = 5.0 * base_weight

		if weight_thresholds is None:
			weight_thresholds = {
				'preferred': pr_val,  # Weights below this are "preferred"
				'safe': saf_val,  # Weights below this are "safe"
				'caution': cautn_val  # Weights below this are "caution", above are "unsafe"
			}

		if colors is None:
			colors = {
				'preferred': 'darkgreen',
				'safe': 'royalblue',
				'caution': 'darkorange',
				'unsafe': 'crimson'
			}

		# Initialize dictionaries to store edges for each category
		edge_categories = {
			'preferred': {'lon': [], 'lat': [], 'weights': [], 'texts': []},
			'safe': {'lon': [], 'lat': [], 'weights': [], 'texts': []},
			'caution': {'lon': [], 'lat': [], 'weights': [], 'texts': []},
			'unsafe': {'lon': [], 'lat': [], 'weights': [], 'texts': []}
		}

		# Determine if we're dealing with NetworkX edges or PostgreSQL edge rows
		is_networkx = False
		if edges_data:
			# Check the first edge to determine the format
			first_edge = next(iter(edges_data))
			if isinstance(first_edge, tuple) and len(first_edge) == 3 and isinstance(first_edge[2], dict):
				is_networkx = True


		# Process edge data and separate into categories
		for edge in edges_data:
			try:
				# Extract weight and geometry based on data format
				if is_networkx:
					# NetworkX edge format: (source, target, attr_dict)
					source, target, attr_dict = edge

					# Get weight from attribute dictionary
					if weight_attribute and weight_attribute in attr_dict:
						weight = float(attr_dict[weight_attribute])
					elif 'adjusted_weight' in attr_dict:
						weight = float(attr_dict['adjusted_weight'])
					elif 'weight' in attr_dict:
						weight = float(attr_dict['weight'])
					else:
						weight = 1.0  # Default weight if not found

					# Get geometry from attributes
					if 'geom' in attr_dict:
						geom_json = attr_dict['geom']
						coords = geom_json['coordinates']
					else:
						# Use node coordinates if no geometry provided
						coords = [source, target]
				else:
					# PostgreSQL edge row format
					if weight_column_index is not None:
						weight = float(edge[weight_column_index])
					else:
						# Default to index 2 for weight if not specified
						weight = float(edge[2])

					# Parse geometry from the edge data
					# Typically at index 3 for PostGIS data, but let's check a few positions
					geom_json = None
					for i in range(3, min(6, len(edge))):
						try:
							if isinstance(edge[i], str) and edge[i].startswith('{"type":'):
								geom_json = json.loads(edge[i])
								break
							elif isinstance(edge[i], dict) and 'type' in edge[i]:
								geom_json = edge[i]
								break
						except (ValueError, json.JSONDecodeError):
							continue

					if geom_json:
						coords = geom_json['coordinates']
					else:
						# If we can't find the geometry, try to use the first two elements as nodes
						try:
							source = json.loads(edge[0].replace("'", '"'))
							target = json.loads(edge[1].replace("'", '"'))
							coords = [source, target]
						except (ValueError, json.JSONDecodeError, AttributeError):
							print(f"Could not parse geometry from edge: {edge}")
							continue

				# Determine category based on thresholds
				if weight <= weight_thresholds['preferred']:
					category = 'preferred'
				elif weight <= weight_thresholds['safe']:
					category = 'safe'
				elif weight <= weight_thresholds['caution']:
					category = 'caution'
				else:
					category = 'unsafe'

				# Add coordinates to the appropriate category
				if len(coords) >= 2:
					# Extract coordinates
					for coord in coords:
						edge_categories[category]['lon'].append(coord[0])
						edge_categories[category]['lat'].append(coord[1])
						edge_categories[category]['weights'].append(weight)
						edge_categories[category]['texts'].append(f"Weight: {weight:.4f}")

					# Add None values to create line breaks
					edge_categories[category]['lon'].append(None)
					edge_categories[category]['lat'].append(None)
					edge_categories[category]['weights'].append(None)
					edge_categories[category]['texts'].append(None)

			except Exception as e:
				print(f"Error processing edge: {e}")
				import traceback
				traceback.print_exc()
				continue

		# Create traces for each category
		for category, data in edge_categories.items():
			# Skip empty categories
			if not data['lon'] or len(data['lon']) <= 1:
				continue

			# Create trace
			trace = go.Scattermapbox(
				lon=data['lon'],
				lat=data['lat'],
				mode='lines',
				line=dict(
					color=colors[category],
					width=edge_width
				),
				text=data['texts'] if show_weights else None,
				hoverinfo='text' if show_weights else 'none',
				name=f"{category.capitalize()} Routes",
				showlegend=True
			)

			# Add trace to figure
			figure.add_trace(trace)

		return figure

	def add_weighted_edge_trace_gl(self, figure, edges_data, edge_width=1, colorscale='Viridis',
	                               show_weights=False, weight_column_index=4):
		"""
		Creates a Plotly Scattergeo trace for weighted graph edges, optimized for large datasets.
		Edge color represents weight value.

		Parameters:
		  figure: Plotly figure object to add the trace to
		  edges_data: List of rows containing edge information, geometry, and weight values
		  edge_width: Width of the edge lines
		  colorscale: Plotly colorscale name for weight visualization
		  show_weights: Whether to show weight values in hover info
		  weight_column_index: Index of the weight column in edge data rows

		Returns:
		  Updated figure with added weighted edge trace
		"""
		# Prepare arrays for coordinates and weights
		lon_coords = []
		lat_coords = []
		weights = []
		hover_texts = []

		# Store valid weights for min/max calculation
		valid_weights = []

		for row in edges_data:
			try:
				# Get weight value
				weight = float(row[weight_column_index])
				valid_weights.append(weight)

				# Parse GeoJSON geometry
				geom_json = json.loads(row[3])

				if geom_json['type'] == 'LineString':
					coords = geom_json['coordinates']

					if len(coords) >= 2:
						# Extract all coordinates from the LineString
						for coord in coords:
							lon_coords.append(coord[0])
							lat_coords.append(coord[1])
							weights.append(weight)
							hover_texts.append(f"Weight: {weight:.4f}")

						# Add None values to break the line between edges
						lon_coords.append(None)
						lat_coords.append(None)
						weights.append(None)
						hover_texts.append(None)
			except Exception as e:
				print(f"Error processing edge: {e}")
				continue

		# If no valid data, return without adding trace
		if not valid_weights:
			print("No valid edges to display")
			return figure

		min_weight = min(valid_weights)
		max_weight = max(valid_weights)

		# Create a trace with color representing weight
		figure.add_trace(go.Scattergeo(
			lon=lon_coords,
			lat=lat_coords,
			mode='lines',
			line=dict(
				width=edge_width,
				color=weights,
				colorscale=colorscale,
				cmin=min_weight,
				cmax=max_weight
			),
			text=hover_texts if show_weights else None,
			hoverinfo='text' if show_weights else 'none',
			name="Weighted Edges",
			showlegend=True,
			colorbar=dict(
				title="Edge Weight",
				thickness=15,
				len=0.5,
			)
		))

		return figure

	def add_route_trace(self, figure, line, name="Route", color="blue", width=3, showlegend=True):
		"""
		Adds a route trace to a Plotly Mapbox figure using a LineString geometry.
		The input 'line' can be provided either as a GeoJSON (dict or str) created from Models.py
		or as a Shapely LineString. In the case of GeoJSON, it will be converted to a Shapely LineString.

		Args:
			figure (go.Figure): A Plotly Mapbox figure.
			line: Either a GeoJSON-format object (dict or str) representing a LineString or a Shapely LineString.
			name (str): The trace name.
			color (str): The color of the line.
			width (int): The thickness of the line.
			showlegend (bool): Whether to show the trace in the legend.

		Returns:
			go.Figure: The figure with the added route trace.
		"""
		# If the input is a string, assume it's a GeoJSON string; if it's a dict, it's a parsed GeoJSON.
		# Otherwise if it doesn't have a .coords attribute, convert it.
		if not hasattr(line, 'coords'):
			# If line is a string, convert it to a dict first
			if isinstance(line, str):
				import json
				line = json.loads(line)
			# Convert GeoJSON to a Shapely geometry
			line = shape(line)

		# Now extract coordinates from the Shapely LineString
		coords = list(line.coords)
		if not coords:
			print("Empty LineString provided; no trace added.")
			return figure

		# Unpack coordinates into separate lists for longitude and latitude
		lons, lats = zip(*coords)

		# Add the LineString as a Scattermapbox trace to the figure
		figure.add_trace(
			go.Scattermapbox(
				lon=lons,
				lat=lats,
				mode="lines",
				line=dict(color=color, width=width),
				name=name,
				showlegend=showlegend
			)
		)
		return figure

	def add_enc_bbox_trace(self, figure, bbox_df, usage_bands:list[int], line_width:int = 1, show_legend=True):
		"""
		Adds a trace for the boundary box to the given figure.
		"""
		bbox_df = self.misc._standardize_enc_name_column(bbox_df)
		# Ensure all GeoDataFrames are in EPSG:4326 (latitude and longitude)
		bbox = bbox_df.to_crs(epsg=4326)
		if usage_bands:
			usage_band_list = [usage_bands] if isinstance(usage_bands, int) else usage_bands
		else:
			usage_band_list = [1, 2, 3, 4, 5, 6]
		# Categorize ENC boundaries into usage bands
		usage_band_dict = {
			1: {'df': bbox[bbox['ENC_NAME'].str[2] == '1'].copy(), 'label': 'Overview ENCs', 'color': 'blue'},
			2: {'df': bbox[bbox['ENC_NAME'].str[2] == '2'].copy(), 'label': 'General ENCs', 'color': 'green'},
			3: {'df': bbox[bbox['ENC_NAME'].str[2] == '3'].copy(), 'label': 'Coastal ENCs', 'color': 'orange'},
			4: {'df': bbox[bbox['ENC_NAME'].str[2] == '4'].copy(), 'label': 'Approach ENCs', 'color': 'red'},
			5: {'df': bbox[bbox['ENC_NAME'].str[2] == '5'].copy(), 'label': 'Harbour ENCs', 'color': 'purple'},
			6: {'df': bbox[bbox['ENC_NAME'].str[2] == '6'].copy(), 'label': 'Berthing ENCs', 'color': 'brown'}
		}

		for band in usage_band_list:
			data = usage_band_dict.get(band)
			if data and not data['df'].empty:
				df = data['df']
				label = data['label']
				color = data['color']

				# Convert usage band geometries to GeoJSON
				usage_geojson = json.loads(df.to_json())

				figure.add_trace(go.Choroplethmapbox(
					geojson=usage_geojson,
					locations=df.index,
					z=[band] * len(df),
					colorscale=[[0, color], [1, color]],
					showscale=False,
					marker_opacity=0.5,
					marker_line_width=2,
					marker_line_color=color,
					name=label,
					showlegend=True,
					hovertemplate='<b>%{text}</b><extra></extra>',
					text=df['ENC_NAME']
				))

	def add_grid_trace(self, figure, grid_geojson, name="Grid", color="blue", fill_opacity=0.5, line_width=2,
							   line_color="black", showlegend=True):
		"""
		Adds a grid polygon trace to a Plotly Mapbox figure using a GeoJSON input.

		Parameters:
		  figure (go.Figure): The Plotly Mapbox figure to add the trace to.
		  grid_geojson: A GeoJSON object representing the grid polygon(s). This can be a dict or a JSON string.
		  name (str): Name of the trace.
		  color (str): Fill color for the grid polygons.
		  fill_opacity (float): Fill opacity (0 to 1).
		  line_width (int): Width of the boundary/outline.
		  line_color (str): Color of the polygon outline.
		  showlegend (bool): Whether to show the trace in the legend.

		Returns:
		  go.Figure: The updated figure.
		"""
		# Handle list of grids
		if isinstance(grid_geojson, list):
			for grid in grid_geojson:
				grid_name = grid['name']
				grid_geom = grid['geom']

				# Convert to FeatureCollection if needed
				if isinstance(grid_geom, str):
					grid_geom = json.loads(grid_geom)

				if grid_geom.get("type", "").lower() != "featurecollection":
					grid_geom = {
						"type": "FeatureCollection",
						"features": [
							{
								"type": "Feature",
								"id": 0,
								"properties": {},
								"geometry": grid_geom
							}
						]
					}

				# Add trace for this grid
				figure.add_trace(
					go.Choroplethmapbox(
						geojson=grid_geom,
						featureidkey="id",
						locations=[0],
						z=[1],
						colorscale=[[0, color], [1, color]],
						marker_opacity=fill_opacity,
						marker_line_width=1,
						marker_line_color=color,
						name=f"{name} - {grid_name}",
						showlegend=True,
						hoverinfo='text',
						text=grid_name,
						showscale=False
					)
				)
		else:

			# If grid_geojson is a JSON string, convert it to a dict.
			if isinstance(grid_geojson, str):
				grid_geojson = json.loads(grid_geojson)

			# If the provided geojson is a single Polygon (or MultiPolygon) rather than a FeatureCollection,
			# wrap it in a FeatureCollection.
			if grid_geojson.get("type", "").lower() != "featurecollection":
				grid_geojson = {
					"type": "FeatureCollection",
					"features": [
						{
							"type": "Feature",
							"id": 0,
							"properties": {},
							"geometry": grid_geojson
						}
					]
				}

			# For Choroplethmapbox, locations must point to feature ids.
			# In our dummy case, we have one feature with id 0.
			figure.add_trace(
				go.Choroplethmapbox(
					geojson=grid_geojson,
					featureidkey="id",
					locations=[0],
					z=[1],
					colorscale=[[0, color], [1, color]],
					marker_line_color=line_color,
					marker_line_width=line_width,
					marker_opacity=fill_opacity,
					name=name,
					showlegend=showlegend,
					hoverinfo="skip",
					showscale=False
				)
			)
		return figure

	def add_grid_trace_gl(self, figure, grid_geojson, name="Grid", color="blue", fill_opacity=0.5,
	                      line_width=2, line_color="black", showlegend=True):
		"""
		Adds a grid polygon trace to a Plotly figure using Choropleth, compatible with ScatterGl.

		Parameters:
			figure (go.Figure): The Plotly figure to add the trace to.
			grid_geojson: A GeoJSON object representing the grid polygon(s). This can be a dict,
						 a JSON string, or a list of grid objects with 'name' and 'geom' keys.
			name (str): Name of the trace.
			color (str): Fill color for the grid polygons.
			fill_opacity (float): Fill opacity (0 to 1).
			line_width (int): Width of the boundary/outline.
			line_color (str): Color of the polygon outline.
			showlegend (bool): Whether to show the trace in the legend.

		Returns:
			go.Figure: The updated figure.
		"""
		# Handle list of grids
		if isinstance(grid_geojson, list):
			for grid in grid_geojson:
				grid_name = grid['name']
				grid_geom = grid['geom']

				# Convert to FeatureCollection if needed
				if isinstance(grid_geom, str):
					grid_geom = json.loads(grid_geom)

				if grid_geom.get("type", "").lower() != "featurecollection":
					grid_geom = {
						"type": "FeatureCollection",
						"features": [
							{
								"type": "Feature",
								"id": 0,
								"properties": {},
								"geometry": grid_geom
							}
						]
					}

				# Add trace for this grid
				figure.add_trace(
					go.Choropleth(
						geojson=grid_geom,
						featureidkey="id",
						locations=[0],
						z=[1],
						colorscale=[[0, color], [1, color]],
						marker=dict(
							opacity=fill_opacity,
							line=dict(width=line_width, color=line_color)
						),
						name=f"{name} - {grid_name}",
						showlegend=True,
						hoverinfo='text',
						text=grid_name,
						showscale=False
					)
				)
		else:
			# If grid_geojson is a JSON string, convert it to a dict
			if isinstance(grid_geojson, str):
				grid_geojson = json.loads(grid_geojson)

			# If the provided geojson is a single Polygon rather than a FeatureCollection,
			# wrap it in a FeatureCollection
			if grid_geojson.get("type", "").lower() != "featurecollection":
				grid_geojson = {
					"type": "FeatureCollection",
					"features": [
						{
							"type": "Feature",
							"id": 0,
							"properties": {},
							"geometry": grid_geojson
						}
					]
				}

			# Add a single grid trace
			figure.add_trace(
				go.Choropleth(
					geojson=grid_geojson,
					featureidkey="id",
					locations=[0],
					z=[1],
					colorscale=[[0, color], [1, color]],
					marker=dict(
						opacity=fill_opacity,
						line=dict(width=line_width, color=line_color)
					),
					name=name,
					showlegend=showlegend,
					hoverinfo="skip",
					showscale=False
				)
			)

		return figure

	def add_graph_trace(self, figure, graph, node_color='red', node_size=6, edge_color='blue', edge_width=2,
						name="Pathfinding Graph"):
		"""
		Adds a graph (from NetworkX) as two traces (edges and nodes) on a Plotly Mapbox figure.
		Each edge is drawn as a line and each node as a marker.

		Parameters:
		  figure (go.Figure): The Plotly Mapbox figure to add the traces to.
		  graph (networkx.Graph): The graph whose nodes and edges are plotted.
		  node_color (str): Color for the node markers.
		  node_size (int): Size for the node markers.
		  edge_color (str): Color for the edge lines.
		  edge_width (int): Width for the edge lines.
		  name (str): Name for the node trace (displayed in legend).

		Returns:
		  go.Figure: The updated figure including the graph traces.
		"""
		# Extract edge coordinates.
		edge_x = []
		edge_y = []
		for edge in graph.edges():
			# Each edge is defined by two nodes: (x0, y0) and (x1, y1)
			x0, y0 = edge[0]
			x1, y1 = edge[1]
			edge_x.extend([x0, x1, None])
			edge_y.extend([y0, y1, None])
		edge_trace = go.Scattermapbox(
			lon=edge_x,
			lat=edge_y,
			mode='lines',
			line=dict(color=edge_color, width=edge_width),
			hoverinfo='none',
			showlegend=False
		)

		# Extract node coordinates.
		node_x = []
		node_y = []
		node_text = []
		for node in graph.nodes():

			x, y = node
			node_x.append(x)
			node_y.append(y)
			node_text.append(f"Node: {node}")
		node_trace = go.Scattermapbox(
			lon=node_x,
			lat=node_y,
			mode='markers',
			marker=dict(
				size=node_size,
				color=node_color,
			),
			text=node_text,
			name=name
		)

		# Add both traces to the figure.
		figure.add_trace(edge_trace)
		figure.add_trace(node_trace)
		return figure

	def add_graph_trace_v2(self, figure, graph, node_color='red', node_size=6, edge_color='blue', edge_width=2,
						name="Pathfinding Graph"):
		"""
		Adds a graph (from NetworkX) as two traces (edges and nodes) on a Plotly Mapbox figure.
		Each edge is drawn as a line and each node as a marker.

		Parameters:
		  figure (go.Figure): The Plotly Mapbox figure to add the traces to.
		  graph (networkx.Graph): The graph whose nodes and edges are plotted.
		  node_color (str): Color for the node markers.
		  node_size (int): Size for the node markers.
		  edge_color (str): Color for the edge lines.
		  edge_width (int): Width for the edge lines.
		  name (str): Name for the node trace (displayed in legend).

		Returns:
		  go.Figure: The updated figure including the graph traces.
		"""

		# ✅ Optimize Edge Extraction using NumPy
		edges = list(graph.edges())
		num_edges = len(edges)  # Move len() call outside the loop (performance improvement)

		edge_x, edge_y = [], []
		edge_coords = np.array(edges).reshape(-1, 2, 2)  # Shape: (num_edges, 2 nodes, 2 coords)
		edge_x = np.hstack([edge_coords[:, 0, 0], edge_coords[:, 1, 0], np.full(num_edges, np.nan)]).tolist()
		edge_y = np.hstack([edge_coords[:, 0, 1], edge_coords[:, 1, 1], np.full(num_edges, np.nan)]).tolist()

		edge_trace = go.Scattermapbox(
			lon=edge_x,
			lat=edge_y,
			mode='lines',
			line=dict(color=edge_color, width=edge_width),
			hoverinfo='none',
			showlegend=False
		)

		# ✅ Optimize Node Extraction using List Comprehension
		nodes = list(graph.nodes())
		num_nodes = len(nodes)  # Move len() call outside the loop

		node_x, node_y = zip(*nodes)  # Unpacking tuples into separate lists
		node_text = [f"Node: {node}" for node in nodes]

		node_trace = go.Scattermapbox(
			lon=node_x,
			lat=node_y,
			mode='markers',
			marker=dict(size=node_size, color=node_color),
			text=node_text,
			name=name
		)

		# ✅ Batch Add Traces
		figure.add_traces([edge_trace, node_trace])

		return figure

	def update_enc_bbox_trace(self, figure, changes: dict, enc_bbox: gpd.GeoDataFrame):
		"""
		Updates ENC boundary box traces based on detected changes.

		Args:
			figure: Plotly figure object
			changes: Dictionary containing added/removed ENCs by usage band
			enc_bbox: GeoDataFrame containing all ENC boundaries
		"""
		# Standardize column name in enc_bbox
		enc_bbox = self.misc._standardize_enc_name_column(enc_bbox)

		usage_band_colors = {
			'Overview': 'blue',
			'General': 'green',
			'Coastal': 'orange',
			'Approach': 'red',
			'Harbour': 'purple',
			'Berthing': 'brown'
		}

		# Mapping zoom levels based on usage band modifications
		usage_zoom = {
			'Overview': 2,
			'General': 3,
			'Coastal': 5,
			'Approach': 9,
			'Harbour': 10,
			'Berthing': 10
		}
		default_zoom = 5

		for action in changes:
			print(f" Action: {action}")
			for usage_band, modifications in changes[action].items():
				trace_name = f"{usage_band} ENCs"
				print(f"Updating {trace_name} trace...")
				trace_idx = self.get_trace_by_name(figure, trace_name)

				if trace_idx is not None:
					# Get current trace data
					current_locations = list(figure.data[trace_idx].locations)
					current_text = list(figure.data[trace_idx].text)

					if action == 'removed':
						# Remove ENCs from trace
						for idx, enc_name in modifications:
							while enc_name in current_text:
								text_idx = current_text.index(enc_name)
								current_locations.pop(text_idx)
								current_text.pop(text_idx)

					elif action == 'added':
						# Add new ENCs to trace
						for idx, enc_name in modifications:
							enc_data = enc_bbox[enc_bbox['ENC_NAME'] == enc_name]
							if not enc_data.empty:
								current_locations.extend([idx] * len(enc_data))
								current_text.extend([enc_name] * len(enc_data))

					# Center the Plotly view on the first added ENC if available
					if action:
						first_enc = modifications[0][1]
						activated_enc_data = enc_bbox[enc_bbox['ENC_NAME'] == first_enc]
						if not activated_enc_data.empty:
							centroid = activated_enc_data.unary_union.centroid
							zoom_level = usage_zoom.get(usage_band, default_zoom)
							print(f"Zoom level for {usage_band} ENCs: {zoom_level}")
							self.set_zoom_to(figure, centroid, zoom_level=5)

					# Update trace with new data
					if current_locations:
						figure.data[trace_idx].locations = current_locations
						figure.data[trace_idx].text = current_text



	def add_boundary_trace(self, figure, boundary_df, name: str = 'Boundary',  show_legend=True,
						   line_width: int = 1, fill_opacity: float = 0.5, color: str = 'black'):
		"""
		Adds boundary polygons trace to the given figure from geometry column.

		Args:
			figure: Plotly figure object to add trace to
			boundary_df: GeoDataFrame containing geometry column with polygons
			name: Name of the boundary (default='Boundary')
			line_width: Width of boundary lines (default=1)
			show_legend: Whether to show legend entry (default=True)
		"""
		# Convert geometries to GeoJSON format
		boundary_geojson = json.loads(boundary_df.to_json())

		# Add polygon trace
		figure.add_trace(go.Choroplethmapbox(
			geojson=boundary_geojson,
			locations=boundary_df.index,
			z=[1] * len(boundary_df),  # Uniform fill color
			colorscale=[[0, 'rgba(0,0,0,0.2)'], [1, 'rgba(0,0,0,0.2)']],  # Transparent fill
			showscale=False,
			marker_opacity=fill_opacity,
			marker_line_width=line_width,
			marker_line_color=color,
			name= name,
			showlegend=show_legend,
			hovertemplate='<b>Boundary</b><extra></extra>'
		))

	def add_layer_trace(self, figure, layer_df, name="Layer Trace", color="blue", fill_opacity=0.5, buffer_size: int = 10):
		"""
		Creates a Plotly trace from the provided GeoDataFrame (layer_df) and adds it to the given figure,
		segregating features by user bands based on ENC name. The ENC name can be in a column named either
		"ENC_NAME" or "dsid_dsnm". If only "dsid_dsnm" is available, it is used as the ENC name.

		For point geometries, a 10-meter buffer is applied to make them more visible on the map.

		User Bands:
			'Overview': 1,
			'General': 2,
			'Coastal': 3,
			'Approach': 4,
			'Harbour': 5,
			'Berthing': 6

		Each trace is assigned a legendgroup corresponding to the band label as specified above.
		The trace name is assigned from the provided "name" parameter.
		"""
		# Ensure the layer is in EPSG:4326 (lat/lon)
		try:
			layer_df = layer_df.to_crs(epsg=4326)
		except Exception as e:
			print("Error converting CRS:", e)

		# If 'ENC_NAME' is not available but 'dsid_dsnm' is, use it as ENC_NAME
		if "ENC_NAME" not in layer_df.columns and "dsid_dsnm" in layer_df.columns:
			layer_df = layer_df.copy()
			layer_df["ENC_NAME"] = layer_df["dsid_dsnm"]
		else:
			layer_df = layer_df.copy()  # Create a copy for modification

		# Apply buffer to point geometries
		# First, identify which geometries are points
		is_point = layer_df.geometry.apply(lambda geom: geom.geom_type == 'Point')

		if is_point.any():
			# Create a copy of the point geometries for buffering
			points_df = layer_df[is_point].copy()

			# Project to Web Mercator (EPSG:3857) for consistent buffer size
			points_df = points_df.to_crs(epsg=3857)

			# Apply 10-meter buffer
			points_df['wkb_geometry'] = points_df['wkb_geometry'].buffer(buffer_size)

			# Project back to WGS84
			points_df = points_df.to_crs(epsg=4326)

			# Replace the original point geometries with the buffered ones
			layer_df.loc[is_point, 'wkb_geometry'] = points_df['wkb_geometry']

		# Convert the full GeoDataFrame to GeoJSON (used if ENC name column is not available)
		geojson_layer = json.loads(layer_df.to_json())

		# Define the mapping between band codes and user band labels
		bands = {'1': 'Overview', '2': 'General', '3': 'Coastal', '4': 'Approach', '5': 'Harbour', '6': 'Berthing'}

		# If neither column is available, then add a single trace
		if "ENC_NAME" not in layer_df.columns:
			trace = go.Choroplethmapbox(
				geojson=geojson_layer,
				locations=layer_df.index,
				z=[1] * len(layer_df),
				colorscale=[[0, color], [1, color]],
				marker_opacity=fill_opacity,
				marker_line_width=1,
				marker_line_color=color,
				name=name,
				legendgroup=name,
				showscale=False,
				hoverinfo='text',
				text=name,
				showlegend=True,
			)
			figure.add_trace(trace)
			return figure

		# Otherwise, segregate features by user band (using the 3rd character in the ENC name)
		for band_code, band_label in bands.items():
			filtered_df = layer_df[layer_df['ENC_NAME'].str[2] == band_code]
			if not filtered_df.empty:
				filtered_geojson = json.loads(filtered_df.to_json())
				trace = go.Choroplethmapbox(
					geojson=filtered_geojson,
					locations=filtered_df.index,
					z=[int(band_code)] * len(filtered_df),
					colorscale=[[0, color], [1, color]],
					marker_opacity=fill_opacity,
					marker_line_width=1,
					marker_line_color=color,
					name=name,
					legendgroup=band_label,
					legendgrouptitle_text=band_label,
					showscale=False,
					hoverinfo='text',
					text=name,
					showlegend=True,
				)
				figure.add_trace(trace)

		figure.update_layout(legend=dict(groupclick="toggleitem"))
		return figure

	def add_highlight_layer_trace(self, figure, layer_df, buffer_size=0.005, name="Highlighted Feature",
	                               color="yellow", fill_opacity=0.3, edge_width=1, edge_color=None):
		"""
		Creates a buffer around small objects (like buoys, wrecks, piles) to highlight their location
		with a semi-transparent circle. The buffer makes small point features more visible on the map.

		Parameters:
			figure: Plotly figure object to add traces to
			layer_df: GeoDataFrame containing the ENC objects to highlight
			buffer_size: Size of buffer in degrees (default: 0.005 degrees, roughly 500m at equator)
			name: Name for the trace in the legend
			color: Fill color for the buffer
			fill_opacity: Opacity of the buffer fill (0-1)
			edge_width: Width of the buffer outline
			edge_color: Color of the buffer outline (defaults to same as fill color if None)

		Returns:
			Updated figure with buffer highlights added
		"""
		# Ensure the layer is in EPSG:4326 (lat/lon)
		try:
			layer_df = layer_df.to_crs(epsg=4326)
		except Exception as e:
			print(f"Error converting CRS: {e}")
			return figure

		# If 'ENC_NAME' is not available but 'dsid_dsnm' is, use it as ENC_NAME
		if "ENC_NAME" not in layer_df.columns and "dsid_dsnm" in layer_df.columns:
			layer_df = layer_df.copy()
			layer_df["ENC_NAME"] = layer_df["dsid_dsnm"]

		# Set outline color to match fill color if not specified
		if edge_color is None:
			edge_color = color

		# Apply buffer to all geometries
		buffered_df = layer_df.copy()
		# Project to a suitable projected CRS (UTM or Web Mercator)
		# Web Mercator (EPSG:3857) is a common choice for web maps
		buffered_df = buffered_df.to_crs(epsg=3857)
		buffer_meters = buffer_size if buffer_size > 1 else buffer_size * 111139  # 1 degree ≈ 111.139 km
		buffered_df['wkb_geometry'] = buffered_df['wkb_geometry'].buffer(buffer_size)

		# Convert back to EPSG:4326 for mapping
		buffered_df = buffered_df.to_crs(epsg=4326)

		# Convert the buffered GeoDataFrame to GeoJSON
		geojson_buffered = json.loads(buffered_df.to_json())

		# Define the mapping between band codes and user band labels
		bands = {'1': 'Overview', '2': 'General', '3': 'Coastal', '4': 'Approach', '5': 'Harbour', '6': 'Berthing'}

		# If neither column is available, then add a single trace
		# if "ENC_NAME" not in buffered_df.columns:
		trace = go.Choroplethmapbox(
			geojson=geojson_buffered,
			locations=buffered_df.index,
			z=[1] * len(buffered_df),
			colorscale=[[0, color], [1, color]],
			marker_opacity=fill_opacity,
			marker_line_width=edge_width,
			marker_line_color=edge_color,
			name=f"{name} (Highlight)",
			legendgroup=name,
			showscale=False,
			hoverinfo='text',
			text=name,
			showlegend=True,
		)
		figure.add_trace(trace)
		return figure

	def add_polygon_trace(self, fig, polygon, simplify_tolerance=None, densify_distance=None,
	                      color='rgba(255, 0, 0, 0.8)', width=3, name='Polygon Trace'):
		"""
		Creates a trace from a polygon's exterior boundary and adds it to a Plotly figure.

		Parameters:
			fig (plotly.graph_objects.Figure): Existing Plotly figure to add the trace to
			polygon (shapely.geometry.Polygon or MultiPolygon): The polygon to trace
			simplify_tolerance (float, optional): Distance tolerance for simplifying the polygon
			densify_distance (float, optional): Distance between points when densifying the polygon
			color (str, optional): Color of the trace line in rgba or hex format
			width (int, optional): Width of the trace line
			name (str, optional): Name of the trace for the legend

		Returns:
			plotly.graph_objects.Figure: Updated figure with the polygon trace added
		"""

		# Handle different polygon types
		if isinstance(polygon, dict):  # GeoJSON dictionary
			from shapely.geometry import shape
			polygon = shape(polygon)
		elif isinstance(polygon, str):  # GeoJSON string
			import json
			from shapely.geometry import shape
			polygon = shape(json.loads(polygon))

		# Handle MultiPolygon by taking the largest polygon
		if isinstance(polygon, MultiPolygon):
			# Find the polygon with the largest area
			largest_poly = max(polygon.geoms, key=lambda p: p.area)
			polygon = largest_poly

		# Ensure we're working with a Polygon
		if not isinstance(polygon, Polygon):
			raise TypeError("Input must be a Polygon or convertible to a Polygon")

		# Optionally simplify the polygon to reduce the number of points
		if simplify_tolerance is not None:
			polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

		# Get the exterior coordinates
		coords = list(polygon.exterior.coords)

		# Optionally densify the polygon by adding points along the boundary
		if densify_distance is not None and densify_distance > 0:
			from shapely.geometry import LineString

			# Create a LineString from the exterior coordinates
			line = LineString(coords)

			# Calculate the total length of the perimeter
			total_length = line.length

			# Calculate the number of points needed
			num_points = max(int(total_length / densify_distance), len(coords))

			# Create evenly spaced points along the perimeter
			densified_coords = []
			for i in range(num_points):
				# Calculate the distance along the line for this point
				distance = i * total_length / num_points
				# Get the point at this distance
				point = line.interpolate(distance)
				densified_coords.append((point.x, point.y))

			# Ensure the polygon is closed
			if densified_coords[0] != densified_coords[-1]:
				densified_coords.append(densified_coords[0])

			coords = densified_coords

		# Extract x and y coordinates for Plotly
		x_coords = [coord[0] for coord in coords]
		y_coords = [coord[1] for coord in coords]

		# Create a Plotly trace for the polygon boundary
		trace = go.Scattermapbox(
			lon=x_coords,
			lat=y_coords,
			mode='lines',
			line=dict(color=color, width=width),
			name=name
		)

		# Add the trace to the figure
		fig.add_trace(trace)

		return fig



	def bbox_plot_plotly(self, bbox_df, land_df, port_df, port_names=None, usage_bands=None, show_ports: bool = True, show_bbox: bool = True):

		bbox_df = self.misc._standardize_enc_name_column(bbox_df)
		# Ensure all GeoDataFrames are in EPSG:4326 (latitude and longitude)
		bbox = bbox_df.to_crs(epsg=4326)

		# Categorize ENC boundaries into usage bands
		usage_band_dict = {
			1: {'df': bbox[bbox['ENC_NAME'].str[2] == '1'].copy(), 'label': 'Overview ENCs', 'color': 'blue'},
			2: {'df': bbox[bbox['ENC_NAME'].str[2] == '2'].copy(), 'label': 'General ENCs', 'color': 'green'},
			3: {'df': bbox[bbox['ENC_NAME'].str[2] == '3'].copy(), 'label': 'Coastal ENCs', 'color': 'orange'},
			4: {'df': bbox[bbox['ENC_NAME'].str[2] == '4'].copy(), 'label': 'Approach ENCs', 'color': 'red'},
			5: {'df': bbox[bbox['ENC_NAME'].str[2] == '5'].copy(), 'label': 'Harbor ENCs', 'color': 'purple'},
			6: {'df': bbox[bbox['ENC_NAME'].str[2] == '6'].copy(), 'label': 'Berthing ENCs', 'color': 'brown'}
		}

		# Filter ports based on provided names
		if show_ports and port_names:
			port_names = [name.upper() for name in port_names]
			port_df['PORT_NAME'] = port_df['PORT_NAME'].str.upper()
			ports = port_df[port_df['PORT_NAME'].isin(port_names)]
		else:
			ports = port_df  # Use all ports if none specified

		if not ports.empty:
			ports = ports.to_crs(epsg=4326)
			ports['lon'] = ports.geometry.x
			ports['lat'] = ports.geometry.y

		# Initialize an empty figure
		fig = go.Figure()

		# Plot ports as a single trace
		if show_ports:
			fig.add_trace(go.Scattermapbox(
				lat=ports['lat'],
				lon=ports['lon'],
				mode='markers+text',
				marker=dict(
					size=10,
					color='red',
					symbol='circle'
				),
				text=ports['PORT_NAME'],
				name='Ports',
				legendgroup='Ports',
				textposition='top right',
				hoverinfo='text',
				showlegend=True
			))

		# Plot usage bands
		if usage_bands:
			for band in usage_bands:
				data = usage_band_dict.get(band)
				if data and not data['df'].empty:
					df = data['df']
					label = data['label']
					color = data['color']

					# Convert usage band geometries to GeoJSON
					usage_geojson = json.loads(df.to_json())

					fig.add_trace(go.Choroplethmapbox(
						geojson=usage_geojson,
						locations=df.index,
						z=[band] * len(df),
						colorscale=[[0, color], [1, color]],
						showscale=False,
						marker_opacity=0.5,
						marker_line_width=2,
						marker_line_color=color,
						name=label,
						showlegend=True,
						hovertemplate='<b>%{text}</b><extra></extra>',
						text=df['ENC_NAME']
					))

		# Update the layout
		fig.update_layout(
			mapbox_style="open-street-map",
			mapbox=dict(
				center=dict(
					lat=ports['lat'].mean() if not ports.empty else 0,
					lon=ports['lon'].mean() if not ports.empty else 0
				),
				zoom=2
			),
			margin=dict(l=5, r=5, t=20, b=5, pad=10),
			width=1280,
			height=720,
			showlegend=True,
			legend=dict(
				title=dict(text='Legend', font=dict(size=14)),
				yanchor="top",
				y=0.99,
				xanchor="left",
				x=0.01
			)
		)

		fig.show()