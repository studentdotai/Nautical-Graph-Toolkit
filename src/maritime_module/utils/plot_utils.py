import json
from typing import List, Dict, Union

import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely import wkt
from shapely.geometry import shape, MultiPolygon, Polygon, LineString

from ..utils.misc_utils import Miscellaneous


class PlotlyChart:
	def __init__(self, misc_utils: Miscellaneous = None):
		"""
		Initializes the PlotlyChart helper.
		Args:
			misc_utils: An instance of the Miscellaneous utility class.
						If not provided, a new one will be created.
		"""
		self.misc = misc_utils or Miscellaneous()
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

	@staticmethod
	def get_trace_by_name(fig, trace_name: str) -> Union[int, None]:
		trace_list = self.get_trace_list(fig)
		for idx, name in trace_list:
			if name == trace_name:
				print(f"Found trace with name '{trace_name}' at index {idx}")
				return idx
		print(f"Error: Trace with name '{trace_name}' not found.")
		return None

	@staticmethod
	def get_trace_item_by_name(fig, trace_name: str, param_name: str):
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

	@staticmethod
	def remove_trace(fig, trace_name: str):
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

	@staticmethod
	def plotly_base_config(figure) -> dict:
		plotly_config={
		  'displayModeBar': 'hover',  # Show mode bar on hover
		  'responsive': True,  # Make the chart responsive
		  'scrollZoom': True,  # Enable scroll to zoom
		  'displaylogo': False,
		  'modeBarButtonsToRemove': ['zoomIn', 'zoomOut', 'pan', 'select', 'lassoSelect'],
		  'modeBarButtonsToAdd': ['autoScale', 'hoverCompareCartesian']
		}
		return plotly_config

	@staticmethod
	def plotly_dinamic_zoom(distance: float) -> int:
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

	@staticmethod
	def set_zoom_to(figure, geometry, zoom_level: int = 7):
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

	@staticmethod
	def update_legend(figure, traceorder='normal', legend_text="Legend", font_size=16, font_color="black"):

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

	@staticmethod
	def set_trace_visibility(figure, trace_name: str) -> None:
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

	@staticmethod
	def create_base_map(title: str ="Global map", mapbox_token: str  = "MAP_BOX_PUBLIC_TOKEN",
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

	@staticmethod
	def add_geo_grids(figure):
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


	@staticmethod
	def add_ports_trace(figure, port_df, name="Ports", color: str = 'black', show_legend=True):
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

	@staticmethod
	def add_single_port_trace(figure, port_series, name="Ports", leg_group="Ports", leg_title="", color: str = 'black', show_legend=True):
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

	@staticmethod
	def add_node_trace(figure, nodes_data):
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

	@staticmethod
	def add_edge_trace(figure, edges_data):
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

	@staticmethod
	def add_edge_trace_batch(figure, edges_data, batch_size=400000, line_color='blue', line_width=2,
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

	@staticmethod
	def add_weighted_edge_trace(figure, edges_data, weight_column_index=None, weight_attribute=None,
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

	@staticmethod
	def add_weighted_edge_trace_gl(figure, edges_data, edge_width=1, colorscale='Viridis',
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

	@staticmethod
	def add_route_trace(figure, line, name="Route", color="blue", width=3, showlegend=True):
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

	def add_enc_bbox_trace(self, figure, bbox_df, usage_bands: List[int], line_width: int = 1, show_legend=True):
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

	def add_grid_trace(self, figure, grid_geojson, name="Grid", color="blue", fill_opacity=0.5,
					   line_width=2, line_color="black", showlegend=True, plot_type="mapbox"):
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
		  plot_type (str): Type of plot ('mapbox' or 'scattergl').

		Returns:
		  go.Figure: The updated figure.
		"""
		# --- REFACTORED to handle list of grids and different plot types ---
		if isinstance(grid_geojson, list):
			for grid in grid_geojson:
				grid_name = grid['name']
				grid_geom = grid['geom']

				# Convert to FeatureCollection if needed
				if isinstance(grid_geom, str):
					try:
						grid_geom = json.loads(grid_geom)
					except json.JSONDecodeError:
						continue

				self._add_single_grid_trace(figure, grid_geom, f"{name} - {grid_name}", color, fill_opacity,
											line_width, line_color, showlegend, plot_type)
		else:
			self._add_single_grid_trace(figure, grid_geojson, name, color, fill_opacity,
										line_width, line_color, showlegend, plot_type)
		return figure

	def _add_single_grid_trace(self, figure, grid_geojson, name, color, fill_opacity, line_width, line_color, showlegend, plot_type):
		"""Helper function to add a single grid trace for either mapbox or scattergl."""
		if isinstance(grid_geojson, str):
			try:
				grid_geojson = json.loads(grid_geojson)
			except json.JSONDecodeError:
				return

		if grid_geojson.get("type", "").lower() != "featurecollection":
			grid_geojson = {
				"type": "FeatureCollection",
				"features": [{"type": "Feature", "id": 0, "properties": {}, "geometry": grid_geojson}]
			}

		if plot_type == "mapbox":
			trace = go.Choroplethmapbox(
				geojson=grid_geojson, featureidkey="id", locations=[0], z=[1],
				colorscale=[[0, color], [1, color]], marker_line_color=line_color,
				marker_line_width=line_width, marker_opacity=fill_opacity, name=name,
				showlegend=showlegend, hoverinfo="skip", showscale=False
			)
		else:  # scattergl
			trace = go.Choropleth(
				geojson=grid_geojson, featureidkey="id", locations=[0], z=[1],
				colorscale=[[0, color], [1, color]],
				marker=dict(opacity=fill_opacity, line=dict(width=line_width, color=line_color)),
				name=name, showlegend=showlegend, hoverinfo="skip", showscale=False
			)
		figure.add_trace(trace)

	@staticmethod
	def add_graph_trace(figure, graph, node_color='red', node_size=6, edge_color='blue', edge_width=2,
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

	@staticmethod
	def add_graph_trace_v2(figure, graph, node_color='red', node_size=6, edge_color='blue', edge_width=2,
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

	def update_enc_bbox_trace(self, figure, changes: dict, enc_bbox: 'gpd.GeoDataFrame'):
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



	@staticmethod
	def add_boundary_trace(figure, boundary_df, name: str = 'Boundary', show_legend=True,
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

	@staticmethod
	def add_layer_trace(figure, layer_df, name="Layer Trace", color="blue", fill_opacity=0.5, buffer_size: int = 10):
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
			points_df['geometry'] = points_df['geometry'].buffer(buffer_size)

			# Project back to WGS84
			points_df = points_df.to_crs(epsg=4326)

			# Replace the original point geometries with the buffered ones
			layer_df.loc[is_point, 'geometry'] = points_df['geometry']

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

	@staticmethod
	def add_highlight_layer_trace(figure, layer_df, buffer_size=0.005, name="Highlighted Feature",
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

	@staticmethod
	def add_polygon_trace(fig, polygon, simplify_tolerance=None, densify_distance=None,
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



	def bbox_plot_plotly(self, bbox_df, port_df, port_names=None, usage_bands=None, show_ports: bool = True):

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