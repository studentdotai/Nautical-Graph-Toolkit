import ast
from datetime import datetime
import json
from decimal import Decimal
import logging

import h3
import math
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.geo import shape
from shapely.geometry.linestring import LineString
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from sqlalchemy import text
from shapely import wkt
from shapely.vectorized import contains
import geopandas as gpd

import json
import os





from Data import PostGIS
from MARITIME_MODULE import Miscellaneous


class Misceleaneous:
	@staticmethod
	def haversine( lon1, lat1, lon2, lat2):
		R = 3440.065
		dlon = math.radians(lon2 - lon1)
		dlat = math.radians(lat2 - lat1)
		a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
			dlon / 2) ** 2
		c = 2 * math.asin(math.sqrt(a))
		return R * c


class BaseGraph:
	"""
	The class RouteAssistant wraps the entire workflow starting from base grid creation.
	• The create_base_grid method executes the SQL (with extra combined grid) and saves grids to PostGIS if required.
	• The create_grid_graph method builds the grid graph (by merging main and extra grids) using the helper create_grid_subgraph_v2.
	• Methods save_graph, clean_graph, and load_graph handle PostGIS persistence and reloading of the graph.
	• The route method computes the first route using A* (through the astar_r
	"""


	def __init__(self, departure_port, arrival_port, port_boundary, enc_schema_name: str, graph_schema_name:str):
		# Use the global PostGIS connection and Miscellaneous conversion functions.
		self.pg = PostGIS()
		self.misc = Miscellaneous()
		self.departure_point = departure_port
		self.arrival_point = arrival_port
		self.port_boundary = port_boundary
		self.enc_schema = enc_schema_name
		self.graph_schema = graph_schema_name



	def create_base_grid(self, layer_table="seaare", extra_grids=["fairwy", "tsslpt", "prcare"],
						 reduce_distance=2, save_to_db=True):
		"""
		Creates a base grid over the port boundary (using main and extra grids) and returns GeoJSON
		for points (departure, start, end, arrival) as well as the main grid and extra grid.
		Additionally, the combined grid is saved to PostGIS (if save_to_db is True) for later use.
		"""
		reduce_distance = self.misc.miles_to_decimal(reduce_distance)

		# Build dynamic UNION ALL query for extra grids
		extra_grid_union = " UNION ALL ".join(
			[f"SELECT wkb_geometry AS geom, dsid_dsnm, '{table}' AS grid_name FROM \"{self.enc_schema}\".\"{table}\""
			 for table in extra_grids]
		)

		with self.pg.connect() as conn:
			grid_query = text(f"""
			WITH grid_enc AS (
				SELECT ST_Union(s.wkb_geometry) AS grid_geom
				FROM "{self.enc_schema}"."{layer_table}" s
				WHERE substring(s.dsid_dsnm from 3 for 1)  IN ('1','2')
				  AND ST_Intersects(s.wkb_geometry, ST_GeomFromText(:port_boundary_geom, 4326))
			), reduced_grid AS (
				SELECT CASE 
						 WHEN :reduce_distance > 0 THEN ST_Buffer(grid_geom, -:reduce_distance)
						 ELSE grid_geom
					   END AS grid_geom
				FROM grid_enc
			), extra_grid AS (
				SELECT grid_name, ST_Union(tbl.geom) AS grid_geom
				FROM (
					{extra_grid_union}
				) tbl
				WHERE substring(tbl.dsid_dsnm from 3 for 1) IN ('4','3')
				  AND ST_Intersects(tbl.geom, ST_GeomFromText(:port_boundary_geom, 4326))
				GROUP BY grid_name
			), combined_grid AS (
				SELECT ST_Union(rg.grid_geom, eg.grid_geom) AS grid_geom
				FROM reduced_grid rg, extra_grid eg
			), dumped AS (
				SELECT (dp).geom AS comp
				FROM (SELECT ST_Dump(combined_grid.grid_geom) AS dp FROM combined_grid) d
			), connected AS (
				SELECT d1.comp
				FROM dumped d1
				WHERE EXISTS (
					 SELECT 1 FROM dumped d2 
					 WHERE ST_DWithin(d1.comp, d2.comp, 0.0001) AND ST_Area(d1.comp) > 0.01
				)
			), filtered_grid AS (
				SELECT ST_Union(comp) AS grid_geom
				FROM connected
			), adjusted_points AS (
				SELECT
					CASE 
						WHEN ST_Contains(fg.grid_geom, ST_GeomFromText(:departure_point, 4326))
						  THEN ST_GeomFromText(:departure_point, 4326)
						ELSE ST_ClosestPoint(fg.grid_geom, ST_GeomFromText(:departure_point, 4326))
					END AS start_point,
					CASE 
						WHEN ST_Contains(fg.grid_geom, ST_GeomFromText(:arrival_point, 4326))
						  THEN ST_GeomFromText(:arrival_point, 4326)
						ELSE ST_ClosestPoint(fg.grid_geom, ST_GeomFromText(:arrival_point, 4326))
					END AS end_point,
					fg.grid_geom
				FROM filtered_grid fg
			)
			SELECT 
				ST_AsGeoJSON(ST_GeomFromText(:departure_point, 4326)) AS departure_point,
				ST_AsGeoJSON(ap.start_point) AS start_point,
				ST_AsGeoJSON(ap.end_point) AS end_point,
				ST_AsGeoJSON(ST_GeomFromText(:arrival_point, 4326)) AS arrival_point,
				ST_AsGeoJSON(rg.grid_geom) AS main_grid_geojson,
				ST_AsGeoJSON(ST_Union(eg.grid_geom)) AS extra_grid,
				ST_AsGeoJSON(fg.grid_geom) AS combined_grid
			FROM adjusted_points ap, reduced_grid rg, extra_grid eg, filtered_grid fg
			GROUP BY departure_point, ap.start_point, ap.end_point, arrival_point, rg.grid_geom, fg.grid_geom;
			""")

			params = {
				'port_boundary_geom': self.port_boundary.wkt if hasattr(self.port_boundary,
																		'wkt') else self.port_boundary,
				'reduce_distance': reduce_distance,
				'departure_point': self.departure_point.wkt if hasattr(self.departure_point, 'wkt') else self.departure_point,
				'arrival_point': self.arrival_point.wkt if hasattr(self.arrival_point, 'wkt') else self.arrival_point,
			}

			result = conn.execute(grid_query, params)
			row = result.fetchone()

			final_result = {
				"points": {
					"dep_point": row[0],
					"start_point": row[1],
					"end_point": row[2],
					"arr_point": row[3]
				},
				"main_grid": row[4],
				"extra_grids": row[5],
				"combined_grid": row[6]
			}

			if save_to_db:
				schema = self.graph_schema
				table_main = "grid_main"
				table_extra = "grid_extra"
				table_combined = "grid_combined"
				create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema}";'

				drop_table_main_sql = 'DROP TABLE IF EXISTS "{}"."grid_main"'.format(schema)
				drop_table_extra_sql = 'DROP TABLE IF EXISTS "{}"."grid_extra"'.format(schema)
				drop_table_combined_sql = 'DROP TABLE IF EXISTS "{}"."grid_combined"'.format(schema)

				create_table_main_sql = f"""
					CREATE TABLE IF NOT EXISTS "{schema}"."{table_main}" (
						 id SERIAL PRIMARY KEY,
						 grid GEOMETRY(MultiPolygon,4326)
					);
					"""
				create_table_extra_sql = f"""
					CREATE TABLE IF NOT EXISTS "{schema}"."{table_extra}" (
						id SERIAL PRIMARY KEY,
						grid GEOMETRY(MultiPolygon,4326)
					);
					"""
				create_table_combined_sql = f"""
					CREATE TABLE IF NOT EXISTS "{schema}"."{table_combined}" (
						id SERIAL PRIMARY KEY,
						grid GEOMETRY(MultiPolygon,4326)
					);
					"""
				insert_main_sql = f"""
					INSERT INTO "{schema}"."{table_main}" (grid)
					VALUES (ST_GeomFromGeoJSON(:geojson));
					"""
				insert_extra_sql = f"""
					INSERT INTO "{schema}"."{table_extra}" (grid)
					VALUES (ST_GeomFromGeoJSON(:geojson));
					"""
				insert_combined_sql = f"""
					INSERT INTO "{schema}"."{table_combined}" (grid)
					VALUES (ST_GeomFromGeoJSON(:geojson));
					"""

				with self.pg.connect() as conn:
					conn.execute(text(create_schema_sql))

					conn.execute(text(drop_table_main_sql))
					conn.execute(text(drop_table_extra_sql))
					conn.execute(text(drop_table_combined_sql))

					conn.execute(text(create_table_main_sql))
					conn.execute(text(create_table_extra_sql))
					conn.execute(text(create_table_combined_sql))
					conn.execute(text(insert_main_sql), {"geojson": row[4]})
					conn.execute(text(insert_extra_sql), {"geojson": row[5]})
					conn.execute(text(insert_combined_sql), {"geojson": row[6]})
					conn.commit()
				print(f"Saved main_grid, extra_grid and combined_grid to PostGIS in schema '{schema}'.")
			return final_result

	def create_base_graph(self, grid_geojson, spacings: float = None) -> nx.Graph:
		"""
		Constructs multiple graphs from grid GeoJSONs from the base grid creation.
		Merges the main grid and extra grids using a mesh grid approach defined in create_grid_subgraph_v2.
		"""
		if spacings:
			spacing = spacings
		else:
			spacing = 0.1

		# Handle both string and dictionary inputs
		if isinstance(grid_geojson, str):
			grid = json.loads(grid_geojson)
		elif isinstance(grid_geojson, dict):
			# If it's a dict from create_base_grid, it might contain GeoJSON strings
			if any(key in grid_geojson for key in ['main_grid', 'combined_grid']):
				# Use combined_grid if available, otherwise main_grid
				grid_key = 'combined_grid' if 'combined_grid' in grid_geojson else 'main_grid'
				grid = json.loads(grid_geojson[grid_key])
			else:
				# Assume it's already a parsed GeoJSON object
				grid = grid_geojson
		else:
			raise TypeError("grid_geojson must be either a GeoJSON string or a dictionary")



		if grid["type"] == "Polygon":
			polygon = Polygon(grid['coordinates'][0])
		elif grid["type"] == "MultiPolygon":
			polygon = MultiPolygon([Polygon(coords[0]) for coords in grid['coordinates']])
		else:
			raise ValueError("Invalid GeoJSON type. Expected 'Polygon' or 'MultiPolygon'.")
		print(f"{datetime.now()} - Subgraph v1 Started")

		graph = self.create_grid_subgraph(polygon, spacing)
		print(f"{datetime.now()} - Subgraph v1 Completed")
		return graph

	def create_grid_graph(self, grid_geojsons: dict, spacings: dict = None) -> nx.Graph:
		"""
		Constructs multiple graphs from grid GeoJSONs from the base grid creation.
		Merges the main grid and extra grids using a mesh grid approach defined in create_grid_subgraph_v2.
		spacing should be int by step of 3
		"""
		if spacings is None:
			spacings = {
				'main_grid': 0.1,
				'extra_grids': 0.2,
			}
		default_sp = self.misc.miles_to_decimal(3)
		main_geojson = json.loads(grid_geojsons['main_grid'])
		if main_geojson["type"] == "Polygon":
			main_polygon = Polygon(main_geojson['coordinates'][0])
		elif main_geojson["type"] == "MultiPolygon":
			main_polygon = MultiPolygon([Polygon(coords[0]) for coords in main_geojson['coordinates']])
		else:
			raise ValueError("Invalid GeoJSON type for main_grid. Expected 'Polygon' or 'MultiPolygon'.")
		main_graph = self.create_grid_subgraph(main_polygon, spacings['main_grid'])

		extra_graphs = []
		extra_grids = grid_geojsons.get('extra_grids', None)
		if extra_grids:
			if isinstance(extra_grids, str):
				extra_polygon = shape(json.loads(extra_grids))
				spacing = spacings.get('extra_grid', default_sp)
				extra_graphs.append(self.create_grid_subgraph(extra_polygon, spacing))
			elif isinstance(extra_grids, list):
				for grid in extra_grids:
					grid_name = grid.get('name', 'unknown_grid')
					grid_polygon = shape(json.loads(grid['geom']))
					spacing = spacings.get(grid_name, default_sp)
					extra_graphs.append(self.create_grid_subgraph(grid_polygon, spacing))
		return nx.compose_all([main_graph] + extra_graphs)

	def create_grid_subgraph(self, polygon, spacing, max_edge_factor=3):
		"""
		Creates a graph for a single grid with specified spacing, using a maximum edge length
		threshold (e.g. 1.5x the spacing) to limit connectivity, thereby avoiding expensive
		spatial operations.

		Parameters:
			polygon (shapely.geometry.Polygon): Polygon geometry for the grid.
			spacing (float): Grid spacing in degrees.
			max_edge_factor (float): Maximum allowed edge length relative to spacing (e.g. 1.5 or 2).

		Returns:
			networkx.Graph: Graph for the specified grid.
		"""
		# Get polygon bounds
		minx, miny, maxx, maxy = polygon.bounds

		# Generate grid points using numpy vectorized arrays
		x_coords, y_coords = np.meshgrid(
			np.arange(minx, maxx + spacing, spacing),
			np.arange(miny, maxy + spacing, spacing)
		)
		print(f"{datetime.now()} - NP Mesh Created \nSpacing: {spacing}")
		# Flatten the meshgrid into coordinate pairs
		points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
		print(f"{datetime.now()} - Column Stack Created")


		# Build nodes only if they fall inside the polygon (OLD APPROACH)
		# nodes = {tuple(pt): Point(pt) for pt in points if polygon.contains(Point(pt))}

		# NEW Use shapely.vectorized.contains to get a boolean mask for points inside the polygon
		mask = contains(polygon, points[:, 0], points[:, 1])
		valid_points = points[mask]

		# Build nodes dictionary (each valid point becomes a node)
		nodes = {tuple(pt): Point(pt) for pt in valid_points}

		print(f"{datetime.now()} - Nodes created: {len(nodes)}")
		# Create a graph and add the nodes
		G = nx.Graph()
		G.add_nodes_from(nodes.keys())
		print(f"{datetime.now()} - Nodes added to Graph")
		# Define eight neighbor directions
		directions = np.array([
			(-spacing, 0), (spacing, 0),
			(0, -spacing), (0, spacing),
			(-spacing, -spacing), (-spacing, spacing),
			(spacing, -spacing), (spacing, spacing)
		])

		# Compute maximum allowed edge length
		max_edge_length = spacing * max_edge_factor

		print(f"{datetime.now()} - Edge Creation Started")
		# Iterate through nodes and add edges if the neighbor exists and distance is within threshold
		for (x, y) in nodes.keys():
			# Build potential neighbor coordinates
			neighbors = [(x + dx, y + dy) for dx, dy in directions if (x + dx, y + dy) in nodes]
			if not neighbors:
				continue  # Skip if no valid neighbors
			# Compute distances vectorized for all neighbors
			distances = np.sqrt(np.sum((np.array(neighbors) - np.array([x, y])) ** 2, axis=1))
			# Filter edges that are within the threshold distance
			valid_edges = [((x, y), nb, {"weight": d}) for nb, d in zip(neighbors, distances) if d <= max_edge_length]
			G.add_edges_from(valid_edges)
		print(f"{datetime.now()} - Edge Creation Complete")
		return G

	def create_grid_subgraph_v2(self, polygon, spacing, max_edge_factor=5, precision=0.01):
		"""
		Creates a graph for a single grid with specified spacing, using a maximum edge length
		threshold (e.g. 1.5x the spacing) to limit connectivity, thereby avoiding expensive
		spatial operations.

		Parameters:
			polygon (shapely.geometry.Polygon): Polygon geometry for the grid.
			spacing (float): Grid spacing in degrees.
			max_edge_factor (float): Maximum allowed edge length relative to spacing (e.g. 1.5 or 2).
			precision (float): Precision for coordinate snapping to avoid floating point issues.

		Returns:
			networkx.Graph: Graph for the specified grid.
		"""
		# Get polygon bounds
		minx, miny, maxx, maxy = polygon.bounds

		# Generate grid points using numpy vectorized arrays
		x_coords, y_coords = np.meshgrid(
			np.arange(minx, maxx + spacing, spacing),
			np.arange(miny, maxy + spacing, spacing)
		)
		# Flatten the meshgrid into coordinate pairs
		points = np.column_stack([x_coords.ravel(), y_coords.ravel()])

		# Snap each coordinate to the fixed precision
		points = np.round(points / precision) * precision

		# Build nodes only if they fall inside the polygon
		nodes = {tuple(pt): Point(pt) for pt in points if polygon.contains(Point(pt))}
		print(f"{datetime.now()} - Nodes found: {len(nodes)}")

		# Create a graph and add the nodes
		G = nx.Graph()
		G.add_nodes_from(nodes.keys())

		# Define eight neighbor directions
		directions = np.array([
			(-spacing, 0), (spacing, 0),
			(0, -spacing), (0, spacing),
			(-spacing, -spacing), (-spacing, spacing),
			(spacing, -spacing), (spacing, spacing)
		])

		# Compute maximum allowed edge length
		max_edge_length = spacing * max_edge_factor

		print(f"{datetime.now()} - Edge Creation Started")
		# Iterate through nodes and add edges if the neighbor exists and distance is within threshold
		for (x, y) in nodes.keys():
			# Build potential neighbor coordinates
			neighbors = [(x + dx, y + dy) for dx, dy in directions if (x + dx, y + dy) in nodes]
			if not neighbors:
				continue  # Skip if no valid neighbors
			# Compute distances vectorized for all neighbors
			distances = np.sqrt(np.sum((np.array(neighbors) - np.array([x, y])) ** 2, axis=1))
			# Filter edges that are within the threshold distance
			valid_edges = [((x, y), nb, {"weight": d}) for nb, d in zip(neighbors, distances) if d <= max_edge_length]
			G.add_edges_from(valid_edges)
		print(f"{datetime.now()} - Edge Creation Complete")
		return G

	def save_graph(self, graph: nx.Graph,
				   nodes_table: str = "graph_nodes", edges_table: str = "graph_edges"):
		"""
		Loads the provided graph into PostGIS by creating nodes and edges tables.
		"""
		print(f"Graph Saved to {nodes_table} and {edges_table} tables")

		create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{self.graph_schema}";'
		drop_table_nodes_sql = f"""DROP TABLE IF EXISTS "{self.graph_schema}"."{nodes_table}" """
		drop_table_edges_sql = f"""DROP TABLE IF EXISTS "{self.graph_schema}"."{edges_table}" """

		create_nodes_table_sql = f"""
			CREATE TABLE IF NOT EXISTS "{self.graph_schema}"."{nodes_table}" (
				 id SERIAL PRIMARY KEY,
				 node VARCHAR,
				 geom GEOMETRY(Point,4326)
			);
			"""
		create_edges_table_sql = f"""
			CREATE TABLE IF NOT EXISTS "{self.graph_schema}"."{edges_table}" (
				 id SERIAL PRIMARY KEY,
				 source VARCHAR, -- String representation, Used for NetworkX Graph
				 target VARCHAR,
				 source_id INTEGER REFERENCES "{self.graph_schema}"."{nodes_table}"(id),  -- FK reference, to provide faster node reference  
				 target_id INTEGER REFERENCES "{self.graph_schema}"."{nodes_table}"(id),  
				 weight DOUBLE PRECISION,
				 geom GEOMETRY(LineString,4326)
			);
			"""
		# Prepare node data
		nodes_data = []
		for node in graph.nodes():
			nodes_data.append((str(node), Point(node)))

		# Create lookup dictionary for node IDs
		node_to_id = {}

		with self.pg.connect() as conn:
			conn.execute(text(create_schema_sql))
			conn.execute(text(drop_table_edges_sql))
			conn.execute(text(drop_table_nodes_sql))
			conn.execute(text(create_nodes_table_sql))
			conn.execute(text(create_edges_table_sql))

			# Insert nodes and collect their generated IDs
			for node_val, point in nodes_data:
				insert_node_sql = f"""
						INSERT INTO "{self.graph_schema}"."{nodes_table}" (node, geom)
						VALUES (:node, ST_GeomFromText(:wkt, 4326))
						RETURNING id;
						"""
				node_id = conn.execute(text(insert_node_sql),
									   {"node": node_val, "wkt": point.wkt}).fetchone()[0]
				node_to_id[node_val] = node_id

			# Insert edges with both string and ID references
			for u, v, data in graph.edges(data=True):
				line = LineString([u, v])
				weight = data.get('weight', 0)
				source_str = str(u)
				target_str = str(v)
				source_id = node_to_id[source_str]
				target_id = node_to_id[target_str]

				insert_edge_sql = f"""
						INSERT INTO "{self.graph_schema}"."{edges_table}" 
						(source, target, source_id, target_id, weight, geom)
						VALUES (:source, :target, :source_id, :target_id, :weight, 
								ST_GeomFromText(:wkt, 4326));
						"""
				conn.execute(text(insert_edge_sql),
							 {"source": source_str, "target": target_str,
							  "source_id": source_id, "target_id": target_id,
							  "weight": weight, "wkt": line.wkt})
			conn.commit()
		print(f"Graph saved into schema '{self.graph_schema}': {len(nodes_data)} nodes, {len(graph.edges())} edges.")

	def clean_graph(self, nodes_table: str = "graph_nodes", edges_table: str = "graph_edges",
					grid_table: str = "grid_combined",
					distance_threshold: float = 5, rearrang_graph: bool = False):
		"""
		Cleans graph edges in PostGIS by removing nodes outside the grid and rearranging node connections for close nodes.
		rearange_graph: bool = False, expensive operation, use with caution
		"""
		distance = self.misc.miles_to_decimal(distance_threshold)
		delete_edges_sql = f"""
				DELETE FROM "{self.graph_schema}"."{edges_table}"
				WHERE source_id IN (
					SELECT id FROM "{self.graph_schema}"."{nodes_table}"
					WHERE NOT EXISTS (
						SELECT 1
						FROM "{self.graph_schema}"."{grid_table}" mg
						WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
					)
				) OR target_id IN (
					SELECT id FROM "{self.graph_schema}"."{nodes_table}"
					WHERE NOT EXISTS (
						SELECT 1
						FROM "{self.graph_schema}"."{grid_table}" mg
						WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
					)
				);
			"""

		# Then delete nodes outside the grid
		delete_nodes_sql = f"""
				DELETE FROM "{self.graph_schema}"."{nodes_table}"
				WHERE NOT EXISTS (
					SELECT 1
					FROM "{self.graph_schema}"."{grid_table}" mg
					WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
				);
			"""

		with self.pg.connect() as conn:
			# First delete the edges to maintain referential integrity
			conn.execute(text(delete_edges_sql))
			# Then delete the nodes
			conn.execute(text(delete_nodes_sql))
			conn.commit()
		print("Edges not within the grid have been removed.")
		print("Nodes outside the grid have been removed.")


		if rearrang_graph:
			insert_sql = f"""
				INSERT INTO "{self.graph_schema}"."{edges_table}" (source, target, weight, geom)
				SELECT n1.node::text AS source, n2.node::text AS target,
					   ST_Distance(n1.geom, n2.geom) AS weight,
					   ST_MakeLine(n1.geom, n2.geom) AS geom
				FROM "{self.graph_schema}"."{nodes_table}" n1, "{self.graph_schema}"."{nodes_table}" n2
				WHERE n1.node != n2.node
				  AND ST_Distance(n1.geom, n2.geom) < :dist_threshold
				  AND NOT EXISTS (
						SELECT 1
						FROM "{self.graph_schema}"."{edges_table}" e
						WHERE (e.source = n1.node::text AND e.target = n2.node::text)
						   OR (e.source = n2.node::text AND e.target = n1.node::text)
				  );
				"""
			with self.pg.connect() as conn:
				conn.execute(text(insert_sql), {"dist_threshold": distance})
				conn.commit()
			print("Node connections have been rearranged based on proximity.")

		# Remove orphan nodes(nodes without connected edges)
		delete_orphans_sql = f"""
					DELETE FROM "{self.graph_schema}"."{nodes_table}" n
					WHERE NOT EXISTS (
						SELECT 1 FROM "{self.graph_schema}"."{edges_table}" e
						WHERE e.source_id = n.id
					)
					AND NOT EXISTS (
						SELECT 1 FROM "{self.graph_schema}"."{edges_table}" e
						WHERE e.target_id = n.id
					);
					"""
		with self.pg.connect() as conn:
			conn.execute(text(delete_orphans_sql))
			conn.commit()
		print("Orphan nodes (nodes without connected edges) have been removed.")

	def clean_graph_h3(self, nodes_table: str = "graph_nodes", edges_table: str = "graph_edges",
					grid_table: str = "grid_combined", land_table: str = "lndare",
					usage_bands: list = ["4", "5", "6"]):
		"""
		Efficiently cleans graph by removing:
		1. Nodes outside the grid
		2. Edges intersecting with land polygons
		3. Orphan nodes (nodes without connected edges)

		Parameters:
			nodes_table (str): Name of nodes table in graph schema
			edges_table (str): Name of edges table in graph schema
			grid_table (str): Name of grid table in graph schema
			land_table (str): Name of land area table in ENC schema
			usage_bands (list): List of usage bands to consider for land areas. Default is ["4", "5", "6"]
		"""
		usage_bands_str = "', '".join(usage_bands)

		# 1. Delete edges with nodes outside the grid
		delete_edges_outside_grid_sql = f"""
			DELETE FROM "{self.graph_schema}"."{edges_table}"
			WHERE source_id IN (
				SELECT id FROM "{self.graph_schema}"."{nodes_table}"
				WHERE NOT EXISTS (
					SELECT 1
					FROM "{self.graph_schema}"."{grid_table}" mg
					WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
				)
			) OR target_id IN (
				SELECT id FROM "{self.graph_schema}"."{nodes_table}"
				WHERE NOT EXISTS (
					SELECT 1
					FROM "{self.graph_schema}"."{grid_table}" mg
					WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
				)
			);
		"""

		# 2. Delete nodes outside the grid
		delete_nodes_outside_grid_sql = f"""
			DELETE FROM "{self.graph_schema}"."{nodes_table}"
			WHERE NOT EXISTS (
				SELECT 1
				FROM "{self.graph_schema}"."{grid_table}" mg
				WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
			);
		"""

		# 3. Delete edges intersecting with land areas
		delete_land_intersection_sql = f"""
			DELETE FROM "{self.graph_schema}"."{edges_table}" e
			WHERE EXISTS (
				SELECT 1 
				FROM "{self.enc_schema}"."{land_table}" l
				WHERE substring(l.dsid_dsnm from 3 for 1) IN ('{usage_bands_str}')
				AND ST_Intersects(e.geom, l.wkb_geometry)
			);
		"""

		# 4. Remove orphan nodes (nodes without connected edges)
		delete_orphans_sql = f"""
			DELETE FROM "{self.graph_schema}"."{nodes_table}" n
			WHERE NOT EXISTS (
				SELECT 1 FROM "{self.graph_schema}"."{edges_table}" e
				WHERE e.source_id = n.id
			)
			AND NOT EXISTS (
				SELECT 1 FROM "{self.graph_schema}"."{edges_table}" e
				WHERE e.target_id = n.id
			);
		"""

		with self.pg.connect() as conn:
			# First delete edges that would cause orphan nodes
			conn.execute(text(delete_edges_outside_grid_sql))
			print("Edges with nodes outside the grid have been removed.")

			# Then delete nodes outside the grid
			conn.execute(text(delete_nodes_outside_grid_sql))
			print("Nodes outside the grid have been removed.")

			# Delete edges intersecting with land
			conn.execute(text(delete_land_intersection_sql))
			print("Edges intersecting with land areas have been removed.")

			# Finally, remove any orphaned nodes
			conn.execute(text(delete_orphans_sql))
			print("Orphan nodes (nodes without connected edges) have been removed.")

			conn.commit()

	def pg_get_graph_nodes_edges(self, schema_name=None, graph_name=None, weighted=False, all_columns: bool = False):
		"""
		Connects to PostGIS and retrieves nodes and edges as GeoJSON.

		Parameters:
			schema_name (str, optional): Schema to use. Defaults to self.graph_schema.
			graph_name (str, optional): Suffix for the graph tables. Defaults to using the base graph.
			weighted (bool, optional): If True, fetches the adjusted_weight for edges.
			all_columns (bool, optional): If True, retrieves all columns in the nodes and edges tables.

		Returns:
		  tuple: (nodes_data, edges_data) where each is a list of rows.
		"""
		if schema_name is None:
			schema_name = self.graph_schema

		if graph_name is None or graph_name == "base":
			nodes_table = "graph_nodes"
			edges_table = "graph_edges"
		else:
			nodes_table = f"graph_nodes_{graph_name}"
			edges_table = f"graph_edges_{graph_name}"

		print(f"Retrieving Graph from tables {nodes_table} and {edges_table}")

		if all_columns:
			nodes_query = f'SELECT * FROM "{schema_name}"."{nodes_table}"'
			edges_query = f'SELECT * FROM "{schema_name}"."{edges_table}"'
		else:
			nodes_query = f'SELECT id, node, ST_AsGeoJSON(geom) AS geom FROM "{schema_name}"."{nodes_table}"'
			if weighted:
				edges_query = f'SELECT source, target, weight, ST_AsGeoJSON(geom) AS geom, adjusted_weight FROM "{schema_name}"."{edges_table}"'
			else:
				edges_query = f'SELECT source, target, weight, ST_AsGeoJSON(geom) AS geom FROM "{schema_name}"."{edges_table}"'

		with self.pg.connect() as conn:
			nodes_data = conn.execute(text(nodes_query)).fetchall()
			edges_data = conn.execute(text(edges_query)).fetchall()
		return nodes_data, edges_data

	def is_truly_undirected(self, schema_name=None, graph_name=None):
		"""
		Checks if the graph is truly undirected by verifying that for every edge (a,b),
		there exists a corresponding edge (b,a) with the same properties using SQLAlchemy.

		Returns:
			bool: True if the graph is truly undirected, False otherwise.
		"""
		if schema_name is None:
			schema_name = self.graph_schema

		if graph_name is None or graph_name == "base":
			edges_table = "graph_edges"
		else:
			edges_table = f"graph_edges_{graph_name}"

		try:
			# Assuming self.session is a SQLAlchemy session
			# or self.engine is a SQLAlchemy engine

			# Query to check for directed edges
			query = text(f"""
	        WITH edges AS (
	            SELECT source, target, weight
	            FROM "{schema_name}"."{edges_table}"
	        ),
	        directed_edges AS (
	            SELECT e1.source, e1.target
	            FROM edges e1
	            LEFT JOIN edges e2 ON e1.source = e2.target AND e1.target = e2.source
	            WHERE e2.source IS NULL
	            UNION
	            SELECT e1.source, e1.target
	            FROM edges e1
	            JOIN edges e2 ON e1.source = e2.target AND e1.target = e2.source
	            WHERE ABS(e1.weight - e2.weight) > 0.000001
	        )
	        SELECT COUNT(*) FROM directed_edges;
	        """)

			# Use existing session if available
			with self.pg.connect() as conn:
				result = conn.execute(query).scalar()

			# If count is 0, then the graph is truly undirected
			if result == 0:
				print(f"{datetime.now()} - Is undirected {graph_name} graph?")
			return result


		except Exception as e:
			# Use logger if available, otherwise fall back to print
			if hasattr(self, 'logger'):
				self.logger.error(f"Error checking if graph is undirected: {e}")
			else:
				print(f"Error checking if graph is undirected: {e}")
			return False

	def pg_export_graph_to_geopackage(self, graph_name=None, output_path=None, include_nodes=True, include_edges=True, weighted=False, all_columns: bool = False):
		"""
		Exports graph nodes and/or edges from PostGIS to a GeoPackage file.

		Parameters:
			graph_name (str, optional): Name of the graph to export. If None, uses the base graph.
			output_path (str, optional): Path where the GeoPackage will be saved.
										If None, uses 'graph_{graph_name}.gpkg'.
			include_nodes (bool): Whether to include nodes in the export (default: True)
			include_edges (bool): Whether to include edges in the export (default: True)
			weighted (bool, optional): If True, fetches the adjusted_weight for edges.
			all_columns (bool, optional): If True, retrieves all columns in the nodes and edges tables.

		Returns:
			str: Path to the created GeoPackage file.
		"""
		# Updated helper to parse geometry data from a column.
		def parse_geometry(x):
			from shapely import wkb, wkt
			try:
				if isinstance(x, dict):
					return shape(x)
				elif isinstance(x, str):
					# First, try loading as GeoJSON.
					try:
						geo = json.loads(x)
						return shape(geo)
					except json.JSONDecodeError:
						pass
					# Next, try to see if it is a hex-encoded WKB.
					# If the string contains only hex digits and has an even length, assume WKB hex.
					if all(c in "0123456789ABCDEFabcdef" for c in x.strip()) and len(x.strip()) % 2 == 0:
						try:
							return wkb.loads(x, hex=True)
						except Exception as wkb_err:
							print("Error parsing geometry as WKB hex:", wkb_err)
					# Fallback: try to load as WKT.
					try:
						return wkt.loads(x)
					except Exception as wkt_err:
						print("Error parsing geometry as WKT:", wkt_err)
				return x
			except Exception as e:
				print("Error parsing geometry:", e)
				return None

		if output_path is None:
			graph_suffix = graph_name if graph_name else "base"
			output_path = f"graph_{graph_suffix}.gpkg"

		# Retrieve nodes and edges using existing function.
		nodes_data, edges_data = self.pg_get_graph_nodes_edges(
			schema_name=self.graph_schema,
			graph_name=graph_name,
			weighted=weighted,
			all_columns=all_columns
		)

		if not nodes_data and not edges_data:
			print(f"No graph data found for graph_name: {graph_name}")
			return None

		# Process nodes if requested.
		if include_nodes and nodes_data:
			if all_columns:
				try:
					nodes_columns = nodes_data[0].keys()
				except AttributeError:
					nodes_columns = None
				nodes_df = pd.DataFrame(nodes_data, columns=nodes_columns)
				geo_col = "geom" if "geom" in nodes_df.columns else "geom_json"
			else:
				nodes_df = pd.DataFrame(nodes_data, columns=['id', 'node', 'geom_json'])
				geo_col = "geom_json"

			nodes_df['geometry'] = nodes_df[geo_col].apply(parse_geometry)
			if geo_col in nodes_df.columns and geo_col != 'geometry':
				nodes_df = nodes_df.drop(columns=[geo_col])
			nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry='geometry', crs="EPSG:4326")
			nodes_gdf.to_file(output_path, driver="GPKG", layer='nodes')
			print(f"Exported {len(nodes_gdf)} nodes to {output_path}, layer 'nodes'")

		# Process edges if requested.
		if include_edges and edges_data:
			if all_columns:
				try:
					edges_columns = edges_data[0].keys()
				except AttributeError:
					edges_columns = None
				edges_df = pd.DataFrame(edges_data, columns=edges_columns)
				geo_col = "geom" if "geom" in edges_df.columns else "geom_json"
			else:
				if weighted:
					edges_df = pd.DataFrame(edges_data, columns=['source', 'target', 'weight', 'geom_json', 'adjusted_weight'])
				else:
					edges_df = pd.DataFrame(edges_data, columns=['source', 'target', 'weight', 'geom_json'])
				geo_col = "geom_json"

			edges_df['geometry'] = edges_df[geo_col].apply(parse_geometry)
			if geo_col in edges_df.columns and geo_col != 'geometry':
				edges_df = edges_df.drop(columns=[geo_col])
			edges_gdf = gpd.GeoDataFrame(edges_df, geometry='geometry', crs="EPSG:4326")

			if include_nodes and nodes_data:
				edges_gdf.to_file(output_path, driver="GPKG", layer='edges', mode='a')
			else:
				edges_gdf.to_file(output_path, driver="GPKG", layer='edges')
			print(f"Exported {len(edges_gdf)} edges to {output_path}, layer 'edges'")

		return output_path

	def pg_load_graph(self, nodes_data, edges_data):
		"""
		Loads the graph from PostGIS by querying the "graph_nodes" and "graph_edges" tables.
		Converts node string representations into tuple keys and forms a NetworkX graph.
		"""
		G = nx.Graph()
		for row in nodes_data:
			try:
				# Convert node string (e.g. "(lon, lat)") into a tuple.
				node_key = ast.literal_eval(row[1])
				geom_json = json.loads(row[2])
				point = shape(geom_json)
				G.add_node(node_key, point=point)
			except Exception as e:
				print("Error processing node:", e)
		for row in edges_data:
			try:
				# row: (source, target, weight, geojson)
				source = ast.literal_eval(row[0])
				target = ast.literal_eval(row[1])
				weight = row[2]
				geom_json = json.loads(row[3])
				G.add_edge(source, target, weight=weight, geom=geom_json)
			except Exception as e:
				print("Error processing edge:", e)
		return G




class FineGraph(BaseGraph):
	"""
		FineGraph extends BaseGraph to provide additional capabilities
		for detailed routing and graph manipulation around specific areas.
		"""

	def __init__(self, enc_schema_name: str, graph_schema_name: str, route_schema_name: str):
		# Initialize the parent BaseGraph class
		# Pass None for points if they're not available at initialization
		super().__init__(
			departure_port=None,
			arrival_port=None,
			port_boundary=None,
			enc_schema_name=enc_schema_name,
			graph_schema_name=graph_schema_name,
		)
		self.route_schema = route_schema_name



	def pg_fine_grid(self, layer_name: str, enc_names: list, route_buffer, geom_column: str = "wkb_geometry",
					 save_to_db: bool = False, schema_name: str = None, table_name: str = "grid_fine"):
		"""
		Creates a fine grid by slicing and combining layer geometries that intersect with a buffer.
		Optimized to return only the combined geometry as GeoJSON for use with NetworkX and Plotly.
		Removes land areas from usage bands 4 and 5 from the resulting geometry.

		Parameters:
		  layer_name (str): The name of the table (in the ENC schema) containing layer geometries.
		  enc_names (list): List of ENC identifier strings to filter the features.
		  route_buffer (shapely.geometry.Polygon): A buffer polygon used for slicing geometries.
		  geom_column (str): Name of the geometry column in the table. (Default is "wkb_geometry".)
		  save_to_db (bool): Whether to save the grid to PostGIS (Default is False).
		  schema_name (str): Schema where the grid will be saved. If None, uses graph_schema.
		  table_name (str): Table name for the saved grid (Default is "fine_grid").

		Returns:
		  str: GeoJSON string representation of the combined sliced geometries with land areas removed.
		"""
		formated_names = self.pg._format_enc_names(enc_names)

		# Use the graph_schema as default if schema_name is None
		if schema_name is None:
			schema_name = self.graph_schema

		# Query to get combined geometry with land areas removed directly from PostgreSQL

		union_query = f"""
		   WITH combined_geometry AS (
			   SELECT ST_Union(
				   ST_Intersection({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
			   ) as geom
			   FROM "{self.enc_schema}"."{layer_name}"
			   WHERE dsid_dsnm = ANY(:enc_names)
				 AND ST_Intersects({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
		   ),
		   land_areas AS (
			   SELECT ST_Union(wkb_geometry) as geom
			   FROM "{self.enc_schema}"."lndare"
			   WHERE substring(dsid_dsnm from 3 for 1) IN ('4','5')
				 AND ST_Intersects(wkb_geometry, ST_GeomFromText(:wkt_buffer, 4326))
		   )
		   SELECT ST_AsGeoJSON(
			   CASE 
				   WHEN (SELECT geom FROM land_areas) IS NOT NULL 
				   THEN ST_Difference((SELECT geom FROM combined_geometry), (SELECT geom FROM land_areas))
				   ELSE (SELECT geom FROM combined_geometry)
			   END
		   ) as combined_geojson
		   """

		params = {"enc_names": formated_names, "wkt_buffer": route_buffer.wkt}

		with self.pg.connect() as conn:
			result = conn.execute(text(union_query), params).fetchone()

		geojson_string = None
		if result and result[0]:
			# Keep as string instead of parsing into a Python dict
			geojson_string = result[0]

			# Save to database if requested
			if save_to_db:
				create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";'

				drop_table_sql = f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"'

				create_table_sql = f"""
					CREATE TABLE IF NOT EXISTS "{schema_name}"."{table_name}" (
						id SERIAL PRIMARY KEY,
						grid GEOMETRY(Geometry,4326),
						created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
					);
				"""

				insert_sql = f"""
					INSERT INTO "{schema_name}"."{table_name}" (grid)
					VALUES (ST_GeomFromGeoJSON(:geojson));
				"""

				with self.pg.connect() as conn:
					conn.execute(text(create_schema_sql))
					conn.execute(text(drop_table_sql))
					conn.execute(text(create_table_sql))
					conn.execute(text(insert_sql), {"geojson": geojson_string})
					conn.commit()

				print(f"Fine grid saved to PostGIS in {schema_name}.{table_name}")

			return geojson_string
		else:
			# Return empty GeoJSON string if no geometries were found
			return '{"type": "GeometryCollection", "geometries": []}'


	def pg_filter_layer_by_buffer(self, layer_name: str, enc_names: list, route_buffer,
								  geom_column: str = "wkb_geometry"):
		"""
		Filters ENC layer geometries by ENC names and restricts the result to only those
		features that lie within the given route buffer polygon. In addition to returning
		a GeoDataFrame, this function also outputs the GeoJSON representation for use in
		Plotly visualizations and graph building.

		Parameters:
		  layer_name (str): The name of the table (in the ENC schema) containing layer geometries.
		  enc_names (list): List of ENC identifier strings to filter the features.
		  route_buffer (shapely.geometry.Polygon): A buffer polygon (e.g., route buffer) used for filtering.
		  geom_column (str): Name of the geometry column in the table. (Default is "wkb_geometry".)

		Returns:
		  tuple: (gdf, geojson_output)
			gdf (gpd.GeoDataFrame): GeoDataFrame of features filtered by ENC names and route buffer.
			geojson_output (str): GeoJSON string representing the filtered features.
		"""
		formated_names = self.pg._format_enc_names(enc_names)

		query = text(f"""
			   SELECT *, ST_AsText({geom_column}) as geom_wkt
			   FROM "{self.enc_schema}"."{layer_name}"
			   WHERE dsid_dsnm = ANY(:enc_names)
				 AND ST_Intersects({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
		   """)

		params = {"enc_names": formated_names, "wkt_buffer": route_buffer.wkt}

		with self.pg.connect() as conn:
			result = conn.execute(query, params)
			rows = result.fetchall()
			columns = result.keys()

		# Convert to DataFrame then GeoDataFrame
		df = pd.DataFrame(rows, columns=columns)
		if not df.empty and 'geom_wkt' in df.columns:
			# Convert the geometry column from WKT to shapely objects
			df["geometry"] = gpd.GeoSeries.from_wkt(df['geom_wkt'])

			# Convert Decimal types to float to avoid JSON serialization issues
			for col in df.select_dtypes(include=['object']).columns:
				try:
					if df[col].apply(lambda x: isinstance(x, Decimal)).any():
						df[col] = df[col].astype(float)
				except:
					pass

			gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
		else:
			# Create an empty GeoDataFrame with a geometry column to satisfy GeoPandas
			gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

		# Custom function to convert Decimal to float for JSON serialization
		def decimal_default(obj):
			if isinstance(obj, Decimal):
				return float(obj)
			raise TypeError

		# Convert the GeoDataFrame into GeoJSON with custom conversion for Decimal
		geojson_output = gdf.to_json(default=decimal_default)
		return gdf, geojson_output

	def pg_slice_layer_by_buffer(self, layer_name: str, enc_names: list, route_buffer,
								 geom_column: str = "wkb_geometry", merge_geometries: bool = False):
		"""
		Slices layer geometries by a buffer polygon using ST_Intersection.
		Returns only the portions of geometries that lie within the buffer.

		Parameters:
		  layer_name (str): The name of the table (in the ENC schema) containing layer geometries.
		  enc_names (list): List of ENC identifier strings to filter the features.
		  route_buffer (shapely.geometry.Polygon): A buffer polygon used for slicing geometries.
		  geom_column (str): Name of the geometry column in the table. (Default is "wkb_geometry".)
		  merge_geometries (bool): If True, returns an additional merged geometry for graph creation.

		Returns:
		  tuple: If merge_geometries is False: (gdf, geojson_output)
				 If merge_geometries is True: (gdf, geojson_output, merged_geometry)
			gdf (gpd.GeoDataFrame): GeoDataFrame of sliced features.
			geojson_output (str): GeoJSON string representing the sliced features.
			merged_geometry (shapely.geometry): Combined geometry of all sliced features (if merge_geometries is True)
		"""
		formated_names = self.pg._format_enc_names(enc_names)

		# Query that slices geometries using ST_Intersection
		base_query = f"""
			SELECT 
				*, 
				ST_AsText(
					ST_Intersection({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
				) as sliced_geom_wkt
			FROM "{self.enc_schema}"."{layer_name}"
			WHERE dsid_dsnm = ANY(:enc_names)
			  AND ST_Intersects({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
		"""

		# Add query for merged geometry if requested
		merged_geom = None
		if merge_geometries:
			merge_query = f"""
				SELECT ST_AsText(
					ST_Union(
						ST_Intersection({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
					)
				) as merged_geometry
				FROM "{self.enc_schema}"."{layer_name}"
				WHERE dsid_dsnm = ANY(:enc_names)
				  AND ST_Intersects({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
			"""

		params = {"enc_names": formated_names, "wkt_buffer": route_buffer.wkt}

		with self.pg.connect() as conn:
			result = conn.execute(text(base_query), params)
			rows = result.fetchall()
			columns = result.keys()

			# Get merged geometry if requested
			if merge_geometries:
				merge_result = conn.execute(text(merge_query), params).fetchone()
				if merge_result and merge_result[0]:
					merged_geom = wkt.loads(merge_result[0])

		# Convert to DataFrame then GeoDataFrame
		df = pd.DataFrame(rows, columns=columns)
		if not df.empty and 'sliced_geom_wkt' in df.columns:
			# Convert the sliced geometry column from WKT to shapely objects
			df["geometry"] = gpd.GeoSeries.from_wkt(df['sliced_geom_wkt'])

			# Convert Decimal types to float to avoid JSON serialization issues
			for col in df.select_dtypes(include=['object']).columns:
				try:
					if df[col].apply(lambda x: isinstance(x, Decimal)).any():
						df[col] = df[col].astype(float)
				except:
					pass

			gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
		else:
			# Create an empty GeoDataFrame with a geometry column to satisfy GeoPandas
			gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

		# Custom function to convert Decimal to float for JSON serialization
		def decimal_default(obj):
			if isinstance(obj, Decimal):
				return float(obj)
			raise TypeError

		# Convert the GeoDataFrame into GeoJSON with correct parameter name for custom handler
		geojson_output = gdf.to_json(default=decimal_default)

		if merge_geometries:
			return gdf, geojson_output, merged_geom
		else:
			return gdf, geojson_output

	def pg_get_graphs_list(self, schema_name=None):
		"""
		Lists all available graphs in the specified schema by identifying
		pairs of node and edge tables with matching suffixes.

		Parameters:
			schema_name (str, optional): Schema to search in. Defaults to self.graph_schema.

		Returns:
			list: List of dictionaries containing graph information with keys:
				  - graph_name: The suffix identifying the graph
				  - nodes_table: Full name of the nodes table
				  - edges_table: Full name of the edges table
		"""
		if schema_name is None:
			schema_name = self.graph_schema

		# Query to get all tables in the schema
		query = text("""
			SELECT table_name 
			FROM information_schema.tables 
			WHERE table_schema = :schema
			AND table_name LIKE 'graph_nodes%' OR table_name LIKE 'graph_edges%'
		""")

		available_graphs = []

		with self.pg.connect() as conn:
			tables = [row[0] for row in conn.execute(query, {"schema": schema_name}).fetchall()]

			# Find node tables
			node_tables = [t for t in tables if t.startswith('graph_nodes')]
			# Find edge tables
			edge_tables = [t for t in tables if t.startswith('graph_edges')]

			# Group by graph name suffix
			for node_table in node_tables:
				# Extract suffix (everything after "graph_nodes_")
				if len(node_table) > 11:  # Longer than just "graph_nodes"
					suffix = node_table[12:]
					matching_edge_table = f"graph_edges_{suffix}"

					if matching_edge_table in edge_tables:

						available_graphs.append(suffix)
				else:
					# Handle the case of default graph tables without suffix
					if "graph_edges" in edge_tables:
						available_graphs.append("base")

		# Print the available graphs in a formatted way
		if available_graphs:
			print(f"Available graphs in schema '{schema_name}':")
			for i, graph in enumerate(available_graphs, 1):
				print(f"{i}. Graph: {graph}")
		else:
			print(f"No graphs found in schema '{schema_name}'")

		return available_graphs

	def _enhance_graph_schema(self, graph_name="base", layer_tables=None):
		"""
		Enhances the graph edges table with additional columns for weight management,
		creating individual columns for each maritime feature layer.

		Parameters:
			graph_name (str): Name of the graph to enhance
			layer_tables (dict): Dictionary of layer names to include as weight columns
		"""
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Add base weight columns if they don't exist
		base_columns_sql = f"""
		DO $$ 
		BEGIN
			IF NOT EXISTS (
				SELECT 1 FROM information_schema.columns 
				WHERE table_schema = '{self.graph_schema}' 
				AND table_name = '{edges_table}' 
				AND column_name = 'base_weight'
			) THEN
				ALTER TABLE "{self.graph_schema}"."{edges_table}" 
				ADD COLUMN base_weight FLOAT,
				ADD COLUMN adjusted_weight FLOAT;

				-- Initialize base_weight from existing weight
				UPDATE "{self.graph_schema}"."{edges_table}"
				SET base_weight = weight,
					adjusted_weight = weight;
			END IF;
		END
		$$;
		"""

		# Generate SQL to add layer-specific weight columns
		layer_columns_sql = []
		if layer_tables:
			for layer_name in layer_tables:
				# Sanitize layer name for column name
				col_name = f"wt_{layer_name.lower()}"

				# Add column if it doesn't exist
				layer_sql = f"""
				DO $$
				BEGIN
					IF NOT EXISTS (
						SELECT 1 FROM information_schema.columns 
						WHERE table_schema = '{self.graph_schema}' 
						AND table_name = '{edges_table}' 
						AND column_name = '{col_name}'
					) THEN
						ALTER TABLE "{self.graph_schema}"."{edges_table}" 
						ADD COLUMN {col_name} FLOAT DEFAULT 1.0;
					END IF;
				END
				$$;
				"""
				layer_columns_sql.append(layer_sql)

		with self.pg.connect() as conn:
			# Add base columns
			conn.execute(text(base_columns_sql))

			# Add layer-specific columns
			if layer_columns_sql:
				for sql in layer_columns_sql:
					conn.execute(text(sql))

			conn.commit()

		print(f"Enhanced graph schema with weight management columns")

		# Return list of column names that were added (useful for apply_feature_weights)
		column_names = []
		if layer_tables:
			column_names = [f"wt_{layer_name.lower()}" for layer_name in layer_tables]
			print(f"Added columns: {', '.join(column_names)}")
		return column_names

	def pg_apply_feature_weights(self, graph_name="base", layer_tables=None, usage_bands=None, apply_to_weight=False):
		"""
		Calculates adjusted weights based on maritime features using individual columns
		for each layer. Original weights are preserved in base_weight column.

		Parameters:
			graph_name (str): Name of the graph to modify
			layer_tables (dict): Dictionary mapping layer names to their weight factors
			usage_bands (list): List of usage bands to include (e.g., ['1','3','4'])
			apply_to_weight (bool): If True, updates the main weight column with the calculated result. Defaults to False.
		"""
		if layer_tables is None:
			layer_tables = {
				'fairwy': {'attr': None, 'factor': 0.8},
				'tsslpt': {'attr': None, 'factor': 0.7},
				'depcnt': {'attr': 'valdco', 'values': {'5': 1.5, '10': 1.2, '20': 0.9}},
				'resare': {'attr': None, 'factor': 2.0},
				'obstrn': {'attr': None, 'factor': 5.0}
			}

		# Default usage bands (all) if not specified
		if usage_bands is None:
			usage_bands = ['1', '2', '3', '4', '5', '6']

		# Format the usage bands for SQL IN clause
		usage_bands_str = "'" + "','".join(usage_bands) + "'"

		# Ensure the enhanced schema exists with columns for each layer
		column_names = self._enhance_graph_schema(graph_name, layer_tables)

		# Determine table name
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Reset all weight columns to 1.0 (neutral factor)
		reset_weights_sql = ", ".join([f"{col} = 1.0" for col in column_names]) if column_names else ""
		if reset_weights_sql:
			reset_sql = f"""
			UPDATE "{self.graph_schema}"."{edges_table}"
			SET {reset_weights_sql};
			"""
			with self.pg.connect() as conn:
				conn.execute(text(reset_sql))
				conn.commit()

		# Generate SQL for each layer's weight tracking
		weight_calculations = []

		for layer_name, config in layer_tables.items():
			# Sanitize layer name for column usage
			col_name = f"wt_{layer_name.lower()}"

			if config.get('attr') and config.get('values'):
				# Attribute-based weights
				for attr_value, factor in config['values'].items():
					# Create column name for this specific attribute value
					attr_col_name = f"wt_{layer_name.lower()}_{attr_value.lower()}"

					# Make sure this column exists
					attr_col_sql = f"""
					DO $$
					BEGIN
						IF NOT EXISTS (
							SELECT 1 FROM information_schema.columns 
							WHERE table_schema = '{self.graph_schema}' 
							AND table_name = '{edges_table}' 
							AND column_name = '{attr_col_name}'
						) THEN
							ALTER TABLE "{self.graph_schema}"."{edges_table}" 
							ADD COLUMN {attr_col_name} FLOAT DEFAULT 1.0;
						END IF;
					END
					$$;
					"""


					# Get actual table name by stripping any suffix after underscore
					actual_layer = layer_name.split('_')[0]
					# SQL to apply weight factor if edge intersects with this feature
					# Add usage band filter to the WHERE clause
					weight_sql = f"""
					UPDATE "{self.graph_schema}"."{edges_table}" e
					SET {attr_col_name} = {factor}
					FROM "{self.enc_schema}"."{actual_layer}" l
					WHERE ST_Intersects(e.geom, l.wkb_geometry)
					AND l.{config['attr']} = '{attr_value}'
					AND substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str});
					"""

					weight_calculations.append((attr_col_sql, weight_sql))
			else:
				# Fixed factor for entire layer
				factor = config.get('factor', 1.0)

				# SQL to apply weight factor if edge intersects with this feature
				# Add usage band filter to the WHERE clause
				weight_sql = f"""
				UPDATE "{self.graph_schema}"."{edges_table}" e
				SET {col_name} = {factor}
				FROM "{self.enc_schema}"."{layer_name}" l
				WHERE ST_Intersects(e.geom, l.wkb_geometry)
				AND substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str});
				"""

				weight_calculations.append((None, weight_sql))

		# Calculate the final adjusted weight based on all individual weight columns
		# Get all weight columns from the table
		get_cols_sql = f"""
		SELECT column_name FROM information_schema.columns 
		WHERE table_schema = '{self.graph_schema}' 
		AND table_name = '{edges_table}'
		AND column_name LIKE 'wt_%';
		"""

		# Execute all SQL in transaction
		with self.pg.connect() as conn:
			try:
				conn.execute(text("BEGIN;"))

				# Ensure base_weight is populated
				conn.execute(text(f"""
					UPDATE "{self.graph_schema}"."{edges_table}" 
					SET base_weight = weight 
					WHERE base_weight IS NULL;
				"""))

				# Create any missing attribute-specific columns and apply weight factors
				for col_sql, weight_sql in weight_calculations:
					if col_sql:
						conn.execute(text(col_sql))
					conn.execute(text(weight_sql))

				# Get all weight columns
				weight_cols = [row[0] for row in conn.execute(text(get_cols_sql)).fetchall()]

				# Build the multiplication expression for all weight columns
				if weight_cols:
					weight_expr = " * ".join(["base_weight"] + weight_cols)

					# Update adjusted_weight
					update_sql = f"""
					UPDATE "{self.graph_schema}"."{edges_table}"
					SET adjusted_weight = {weight_expr}
					"""
					conn.execute(text(update_sql))

					# Optionally update main weight
					if apply_to_weight:
						weight_sql = f"""
						UPDATE "{self.graph_schema}"."{edges_table}"
						SET weight = adjusted_weight
						"""
						conn.execute(text(weight_sql))

				conn.execute(text("COMMIT;"))
				print(f"Successfully applied feature weights to graph '{graph_name}' using usage bands: {usage_bands}")
				return True

			except Exception as e:
				conn.execute(text("ROLLBACK;"))
				print(f"Error calculating weights: {str(e)}")
				return False

	# New function to load graph with specific weighting
	def pg_load_graph_with_weights(self, nodes_data, edges_data, use_adjusted_weights=True):
		"""
		Loads the graph from PostGIS data, with option to use original or adjusted weights

		Parameters:
			nodes_data, edges_data: Data from pg_get_graph_nodes_edges()
			use_adjusted_weights: If True, uses adjusted_weight instead of base_weight

		Returns:
			nx.Graph: NetworkX graph with selected weight values
		"""
		G = nx.Graph()

		for row in nodes_data:
			try:
				node_key = ast.literal_eval(row[1])
				geom_json = json.loads(row[2])
				point = shape(geom_json)
				G.add_node(node_key, point=point)
			except Exception as e:
				print("Error processing node:", e)

		for row in edges_data:
			try:
				source = ast.literal_eval(row[0])
				target = ast.literal_eval(row[1])

				# Extract weights - assume row structure includes adjusted_weight and base_weight
				if len(row) >= 4:  # Original format with just weight
					default_weight = row[2]
					geom_json = json.loads(row[3])

					# Simple case - just use the provided weight
					G.add_edge(source, target, weight=default_weight, geom=geom_json)
				else:  # Enhanced format with both weights
					base_weight = row[2]
					adjusted_weight = row[3] if row[3] is not None else base_weight
					factors = row[4]  # JSONB stored as string
					geom_json = json.loads(row[5])

					# Choose which weight to use
					weight_to_use = adjusted_weight if use_adjusted_weights else base_weight

					G.add_edge(source, target,
							   weight=weight_to_use,
							   base_weight=base_weight,
							   adjusted_weight=adjusted_weight,
							   weight_factors=factors,
							   geom=geom_json)
			except Exception as e:
				print("Error processing edge:", e)

		return G

	def pg_connect_nodes(self, source_id, target_id, custom_weight=None,
	                     graph_name=None
	                     ):
		"""
		Creates a new edge between two existing nodes in the graph database using their primary key IDs.

		Parameters:
			source_id (int): Primary key ID of the source node
			target_id (int): Primary key ID of the target node
			custom_weight (float, optional): Custom weight for the edge. If None, calculated based on distance.
			nodes_table (str): Name of the nodes table (default "graph_nodes")
			edges_table (str): Name of the edges table (default "graph_edges")

		Returns:
			bool: True if edge creation was successful, False otherwise
		"""
		print(f'Graph input name: {graph_name}')
		# if graph_name is None or "base":
		# 	nodes_table = "graph_nodes"
		# 	edges_table = "graph_edges"
		# else:

		nodes_table = f"graph_nodes_{graph_name}"
		edges_table = f"graph_edges_{graph_name}"


		print(nodes_table)
		# SQL to get node details based on their IDs
		node_query = f"""
		SELECT id, node, ST_AsText(geom) as geom_wkt
		FROM "{self.graph_schema}"."{nodes_table}"
		WHERE id IN (:source_id, :target_id)
		"""

		with self.pg.connect() as conn:
			try:
				# Get node information
				nodes_result = conn.execute(text(node_query),
											{"source_id": int(source_id), "target_id": int(target_id)})
				print(nodes_result)
				nodes = nodes_result.fetchall()
				print(len(nodes))
				if len(nodes) != 2:
					print(f"Error: One or both nodes (IDs: {source_id}, {target_id}) not found.")
					return False

				# Map node details
				node_map = {row[0]: {"node_str": row[1], "geom": row[2]} for row in nodes}

				# Check if edge already exists
				check_edge_sql = f"""
				SELECT COUNT(*) FROM "{self.graph_schema}"."{edges_table}"
				WHERE (source_id = :source_id AND target_id = :target_id)
				OR (source_id = :target_id AND target_id = :source_id)
				"""

				edge_exists = conn.execute(text(check_edge_sql),
										   {"source_id": source_id, "target_id": target_id}).scalar()

				if edge_exists > 0:
					print(f"Edge between nodes {source_id} and {target_id} already exists.")
					return False

				# Calculate weight based on distance if not provided
				if custom_weight is None:
					weight_sql = f"""
					SELECT ST_Distance(
						ST_GeomFromText(:source_geom, 4326),
						ST_GeomFromText(:target_geom, 4326)
					) as weight
					"""
					weight_result = conn.execute(text(weight_sql), {
						"source_geom": node_map[source_id]["geom"],
						"target_geom": node_map[target_id]["geom"]
					}).scalar()
					weight = weight_result
				else:
					weight = custom_weight

				# Create the edge
				insert_edge_sql = f"""
				INSERT INTO "{self.graph_schema}"."{edges_table}" 
				(source, target, source_id, target_id, weight, geom)
				VALUES (
					:source_node_str, 
					:target_node_str, 
					:source_id, 
					:target_id, 
					:weight, 
					ST_MakeLine(
						(SELECT geom FROM "{self.graph_schema}"."{nodes_table}" WHERE id = :source_id),
						(SELECT geom FROM "{self.graph_schema}"."{nodes_table}" WHERE id = :target_id)
					)
				)
				"""

				conn.execute(text(insert_edge_sql), {
					"source_node_str": node_map[source_id]["node_str"],
					"target_node_str": node_map[target_id]["node_str"],
					"source_id": source_id,
					"target_id": target_id,
					"weight": weight
				})

				conn.commit()
				print(f"Successfully created edge between nodes {source_id} and {target_id} with weight {weight}")
				return True

			except Exception as e:
				print(f"Error creating edge: {str(e)}")
				return False

	@staticmethod
	def verify_graph(G, nodes_gdf, edges_gdf):
		"""
		Verify that the graph has the correct number of nodes and edges
		and consists of a single connected component.

		Parameters:
		-----------
		G : NetworkX Graph
			The graph to verify
		nodes_gdf : GeoDataFrame
			Original nodes GeoDataFrame
		edges_gdf : GeoDataFrame
			Original edges GeoDataFrame

		Returns:
		--------
		is_valid : bool
			True if the graph is valid, False otherwise
		"""
		# Check node count
		expected_nodes = len(nodes_gdf)
		actual_nodes = G.number_of_nodes()
		nodes_match = expected_nodes == actual_nodes

		# Check edge count
		expected_edges = len(edges_gdf)
		actual_edges = G.number_of_edges()
		edges_match = expected_edges == actual_edges

		# Check connectivity
		is_connected = nx.is_connected(G)

		# Print verification results
		print(f"Node count: Expected {expected_nodes}, Actual {actual_nodes}, Match: {nodes_match}")
		print(f"Edge count: Expected {expected_edges}, Actual {actual_edges}, Match: {edges_match}")
		print(f"Graph is connected: {is_connected}")

		if not is_connected:
			# Find connected components
			components = list(nx.connected_components(G))
			print(f"Number of connected components: {len(components)}")
			print(f"Sizes of components: {[len(c) for c in components]}")

		return nodes_match and edges_match and is_connected

	def pg_copy_table(self, source_table, target_table, schema_name):
		"""
		Creates a copy of a PostgreSQL table using SQLAlchemy.

		Args:
			source_table (str): Name of the source table
			target_table (str): Name of the target table
			schema_name (str): Schema name

		Returns:
			bool: True if successful, False otherwise
		"""
		try:
			# Clean up table names - remove any spaces
			source_table = source_table.strip()
			target_table = target_table.strip()

			# Check if target table already exists
			check_sql = text(f"""
	            SELECT EXISTS (
	                SELECT FROM information_schema.tables 
	                WHERE table_schema = :schema 
	                AND table_name = :table
	            )
	        """)

			# Drop table if it exists
			drop_sql = text(f'DROP TABLE IF EXISTS "{schema_name}"."{target_table}"')

			# Create new table as copy
			copy_sql = text(f'CREATE TABLE "{schema_name}"."{target_table}" AS TABLE "{schema_name}"."{source_table}"')

			with self.pg.connect() as conn:
				# Check if table exists
				exists = conn.execute(check_sql, {"schema": schema_name, "table": target_table}).scalar()

				# Start transaction
				conn.execute(text("BEGIN;"))

				# Drop if exists
				if exists:
					conn.execute(drop_sql)

				# Create copy
				conn.execute(copy_sql)

				# Commit transaction
				conn.execute(text("COMMIT;"))

			print(f"Table {schema_name}.{target_table} created as a copy of {schema_name}.{source_table}")
			return True

		except Exception as e:
			print(f"Error copying table: {str(e)}")
			with self.pg.connect() as conn:
				conn.execute(text("ROLLBACK;"))
			return False

class H3Graph(BaseGraph):
	def __init__(self, enc_schema_name: str, graph_schema_name: str, route_schema_name: str):
		# Initialize the parent BaseGraph class
		# Pass None for points if they're not available at initialization
		super().__init__(
			departure_port=None,
			arrival_port=None,
			port_boundary=None,
			enc_schema_name=enc_schema_name,
			graph_schema_name=graph_schema_name,
		)
		self.route_schema = route_schema_name

	def pg_create_h3_grid(self, base_layer="seaare", optional_layers=None, enc_names: list = None, usage_bands=None,
						  route_buffer=None, save_to_db=False, schema_name=None, table_name="grid_h3"):
		"""
		Creates a grid suitable for H3 cell generation by slicing and combining layer geometries.
		Uses seaare as base layer and filters by usage bands, with optional additional layers.
		Removes land areas from the resulting geometry.

		Parameters:
			base_layer (str): The name of the base table (default is "seaare")
			enc_names (list): List of ENC identifier strings to filter the features.
			optional_layers (list): List of additional layers to include (e.g., ["fairwy", "tsslpt", "prcare"])
			usage_bands (dict): Dictionary mapping layer names to usage bands to include
							   (e.g., {"seaare": ["1", "2"], "fairwy": ["3", "4"]})
			route_buffer (shapely.geometry.Polygon): Optional buffer polygon to restrict the area
			save_to_db (bool): Whether to save the grid to PostGIS (Default is False)
			schema_name (str): Schema where the grid will be saved. If None, uses graph_schema
			table_name (str): Table name for the saved grid (Default is "grid_h3")

		Returns:
			dict: A dictionary mapping keys of the form "<layer>_band<usage_band>" to a GeoJSON string.
		"""
		# Set defaults
		if optional_layers is None:
			optional_layers = ["fairwy", "tsslpt", "prcare"]

		if usage_bands is None:
			usage_bands = {
				"seaare": ["1", "2", "3", "4", "5", "6"],
				"fairwy": ["3", "4", "5"],
				"tsslpt": ["3", "4", "5"],
				"prcare": ["3", "4", "5"]
			}

		# Use the graph_schema as default if schema_name is None
		if schema_name is None:
			schema_name = self.graph_schema

		# If provided, format the ENC names filter as in pg_fine_grid
		formated_names = None
		if enc_names is not None:
			formated_names = self.pg._format_enc_names(enc_names)

		# Initialize results dictionary
		results = {}
		collection = {}

		combined_query = f"""
				WITH combined_seaare AS (
					SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					) as geom
					FROM "{self.enc_schema}"."{base_layer}"
					WHERE substring(dsid_dsnm from 3 for 1) IN ('1','2','3','4','5','6')
					{"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					{f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				),
				land_areas AS (
					SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					) as geom
					FROM "{self.enc_schema}"."lndare"
					WHERE substring(dsid_dsnm from 3 for 1) IN ('4','5','6')
					{"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					{f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				)
				SELECT ST_AsGeoJSON(
					CASE 
						WHEN (SELECT geom FROM land_areas) IS NOT NULL 
						THEN ST_Difference((SELECT geom FROM combined_seaare), (SELECT geom FROM land_areas))
						ELSE (SELECT geom FROM combined_seaare)
					END
				) as geojson
			"""

		with self.pg.connect() as conn:
			params = {}
			if formated_names:
				params["enc_names"] = formated_names
			combined_result = conn.execute(text(combined_query), params).fetchone()

			if combined_result and combined_result[0]:
				results["combined_grid"] = combined_result[0]
			else:
				collection["combined_grid"] = '{"type": "GeometryCollection", "geometries": []}'

		# Process base layer (seaare) by usage bands
		for usage_band in usage_bands.get(base_layer, ["1", "2", "3", "4", "5", "6"]):
			# Build the spatial query with usage band filter and optional ENC filtering
			base_query = f"""
				 WITH base_geometry AS (
					 SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					) as geom
					 FROM "{self.enc_schema}"."{base_layer}"
					 WHERE substring(dsid_dsnm from 3 for 1) = '{usage_band}'
					 {"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					 {f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				 ),
				 land_areas AS (
					 SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					) as geom
					 FROM "{self.enc_schema}"."lndare"
					 WHERE substring(dsid_dsnm from 3 for 1) IN ('4','5','6')
					 {"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					 {f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				 )
				 SELECT ST_AsGeoJSON(
					 CASE 
						 WHEN (SELECT geom FROM land_areas) IS NOT NULL 
						 THEN ST_Difference((SELECT geom FROM base_geometry), (SELECT geom FROM land_areas))
						 ELSE (SELECT geom FROM base_geometry)
					 END
				 ) as geojson
				 """

			with self.pg.connect() as conn:
				params = {}
				if formated_names:
					params["enc_names"] = formated_names
				cur_result = conn.execute(text(base_query), params).fetchone()

				if cur_result and cur_result[0]:
					results[f"{base_layer}_band{usage_band}"] = cur_result[0]
				else:
					collection[f"{base_layer}_band{usage_band}"] = '{"type": "GeometryCollection", "geometries": []}'

		# Process optional layers
		for layer in optional_layers:

			# Build the spatial query for optional layer using the same ENC filtering if provided
			layer_query = f"""
				 WITH layer_geometry AS (
					 SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					 ) as geom
					 FROM "{self.enc_schema}"."{layer}"
					 WHERE substring(dsid_dsnm from 3 for 1) IN ('3','4', '5', '6')
					 {"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					 {f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				 ),
				 land_areas AS (
					 SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					 ) as geom
					 FROM "{self.enc_schema}"."lndare"
					 WHERE substring(dsid_dsnm from 3 for 1) IN ('4','5','6')
					 {"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					 {f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				 )
				 SELECT ST_AsGeoJSON(
					 CASE 
						 WHEN (SELECT geom FROM land_areas) IS NOT NULL 
						 THEN ST_Difference((SELECT geom FROM layer_geometry), (SELECT geom FROM land_areas))
						 ELSE (SELECT geom FROM layer_geometry)
					 END
				 ) as geojson
				 """

			with self.pg.connect() as conn:
				params = {}
				if formated_names:
					params["enc_names"] = formated_names
				cur_result = conn.execute(text(layer_query), params).fetchone()

				if cur_result and cur_result[0] and cur_result[0] != '{"type":"GeometryCollection","geometries":[]}':
					results[f"{layer}"] = cur_result[0]
				else:
					collection[f"{layer}"] = '{"type": "GeometryCollection", "geometries": []}'

		# Save to database if requested
		if save_to_db and results:
			create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";'
			drop_table_sql = f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"'

			create_table_sql = f"""
				 CREATE TABLE IF NOT EXISTS "{schema_name}"."{table_name}" (
				 id SERIAL PRIMARY KEY,
				 layer_name VARCHAR(50),
				 usage_band VARCHAR(10),
				 grid GEOMETRY(Geometry,4326),
				 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				 );
				 """

			base_table = f'{table_name}_base'
			# SQL for combined grid table
			drop_combined_table_sql = f'DROP TABLE IF EXISTS "{schema_name}"."{base_table}"'
			create_combined_table_sql = f"""
					 CREATE TABLE IF NOT EXISTS "{schema_name}"."{base_table}" (
						 id SERIAL PRIMARY KEY,
						 grid GEOMETRY(Geometry,4326),
						 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
					 );
					 """

			with self.pg.connect() as conn:
				conn.execute(text(create_schema_sql))

				# Create regular grid table
				conn.execute(text(drop_table_sql))
				conn.execute(text(create_table_sql))

				# Create combined grid table
				conn.execute(text(drop_combined_table_sql))
				conn.execute(text(create_combined_table_sql))

				# Insert each result into the regular table (except combined grid)
				for key, geojson in results.items():
					if key != "combined_grid":
						# Parse the key to get layer name and usage band
						if "_band" in key:
							parts = key.split('_band')
							layer_name = parts[0]
							usage_band = parts[1]
						else:
							layer_name = key
							usage_band = ''

						insert_sql = f"""
								 INSERT INTO "{schema_name}"."{table_name}" (layer_name, usage_band, grid)
								 VALUES (:layer_name, :usage_band, ST_GeomFromGeoJSON(:geojson));
								 """

						conn.execute(text(insert_sql), {
							"layer_name": layer_name,
							"usage_band": usage_band,
							"geojson": geojson
						})

				# Insert combined grid into the dedicated table
				if "combined_grid" in results:
					insert_combined_sql = f"""
							 INSERT INTO "{schema_name}"."{base_table}" (grid)
							 VALUES (ST_GeomFromGeoJSON(:geojson));
							 """
					conn.execute(text(insert_combined_sql), {
						"geojson": results["combined_grid"]
					})

				conn.commit()

			print(f"H3 grid layers saved to PostGIS in {schema_name}.{table_name}")
			print(f"Combined H3 grid saved to PostGIS in {schema_name}.{base_table}")

		return results

	def create_h3_graph(self, base_resolution=7, detail_resolution=11,
					   base_layer="seaare", optional_layers=None, enc_names: list = None, usage_bands=None,
					   route_buffer=None, save_to_db=False, table_name="grid_h3"):
		"""
		Creates a multi-resolution H3 grid based on maritime features.

		Parameters:
			base_resolution (int): H3 resolution for base areas (5-6 recommended)
			detail_resolution (int): H3 resolution for detailed areas (7-9 recommended)
			base_layer (str): The name of the base table (default is "seaare")
			optional_layers (list): List of additional layers for detailed resolution
			enc_names (list): List of ENC identifier strings to filter the features.
			usage_bands (dict): Dictionary mapping layer names to usage bands
			route_buffer (shapely.geometry.Polygon): Optional buffer to restrict the area
			save_to_db (bool): Whether to save the H3 cells to PostGIS
			table_name (str): Table name for the saved grid (Default is "grid_h3")
		Returns:
			nx.Graph: NetworkX graph built from the multi-resolution H3 grid
		"""

		print(f"{datetime.now()} - Starting H3 grid creation")

		# Retrieve grid polygons for different layers
		grid_geojsons = self.pg_create_h3_grid(
			base_layer=base_layer,
			optional_layers=optional_layers,
			enc_names = enc_names,
			usage_bands=usage_bands,
			route_buffer=route_buffer,
			save_to_db=save_to_db,
			table_name = table_name
		)
		print(f"{datetime.now()} - Retrieved {len(grid_geojsons)} grid polygons")

		# Initialize sets for H3 cells
		base_hexagons = set()
		detail_hexagons = set()

		# Process each polygon only once, differentiating base and detail layers.
		for key, geojson_str in grid_geojsons.items():
			if key != "combined_grid":
				try:
					geojson = json.loads(geojson_str)

					if key.startswith(base_layer):
						band = key.split('_')[1]
						print(f"Band: {band}")
						if band in ["band1", "band2"]:
							cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), base_resolution)
						elif band == "band3":
							cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution - 2)
						elif band == "band4":
							cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution - 1)
						else:
							cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution)
						base_hexagons.update(cells)
						print(f"{datetime.now()} - Added {len(cells)} base cells from {key}")
					elif key in ["prcare"]:
						cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution - 1)
						detail_hexagons.update(cells)
						print(f"{datetime.now()} - Added {len(cells)} detail cells from {key}")
					else:
						cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution)
						detail_hexagons.update(cells)
						print(f"{datetime.now()} - Added {len(cells)} detail cells from {key}")
				except Exception as e:
					print(f"Error processing {key}: {str(e)}")

		# Remove base cells that are covered by detail cells using a set comprehension
		print(f'Cleaning Child - Parrent Cells')
		base_to_remove = set()
		for detail_cell in detail_hexagons:
			# Get the cell's current resolution
			current_res = h3.get_resolution(detail_cell)

			# Find the appropriate parent at base_resolution
			if current_res > base_resolution:
				parent = h3.cell_to_parent(detail_cell, base_resolution)
				base_to_remove.add(parent)

		# Remove the identified base cells
		base_hexagons -= base_to_remove

		print(
			f"{datetime.now()} - Final grid has {len(base_hexagons)} base cells and {len(detail_hexagons)} detail cells")

		# Create graph from hexagons
		G = nx.Graph()

		# Helper: cache cell centers for reuse
		def get_center(cell):
			# Returns (lng, lat) as used by the graph nodes
			lat, lng = h3.cell_to_latlng(cell)
			return (lng, lat)

		# Add nodes for base and detail resolution
		for h3_idx in base_hexagons:
			G.add_node(get_center(h3_idx), h3_index=h3_idx, resolution=base_resolution)
		for h3_idx in detail_hexagons:
			G.add_node(get_center(h3_idx), h3_index=h3_idx, resolution=detail_resolution)

		print(f"{datetime.now()} - Added {len(G.nodes)} nodes to graph")

		edges_added = 0

		# Function to add an edge between two cells given their H3 indexes and centers.
		def add_edge(cell_a, cell_b):
			center_a = get_center(cell_a)
			center_b = get_center(cell_b)
			weight = Misceleaneous.haversine(center_a[0], center_a[1], center_b[0], center_b[1])
			G.add_edge(center_a, center_b, weight=weight, h3_edge=(cell_a, cell_b))

		# Add edges for base resolution cells
		for h3_idx in base_hexagons:
			for neighbor in h3.grid_ring(h3_idx, 1):
				if neighbor in base_hexagons:
					add_edge(h3_idx, neighbor)
					edges_added += 1

		# Add edges for detail resolution cells
		for h3_idx in detail_hexagons:
			for neighbor in h3.grid_ring(h3_idx, 1):
				if neighbor in detail_hexagons:
					add_edge(h3_idx, neighbor)
					edges_added += 1

		# Optimized: Connect cross-resolution edges by iterating over detail cells only.
		# For each detail cell, compute its parent at the base resolution, then check the neighbors
		# of that parent. If a neighboring base cell exists, add an edge.
		for detail_idx in detail_hexagons:
			detail_parent = h3.cell_to_parent(detail_idx, base_resolution)
			# Get neighbors (fixed, at most 6) of the parent cell
			parent_neighbors = h3.grid_ring(detail_parent, 1)
			for base_candidate in parent_neighbors:
				if base_candidate in base_hexagons:
					add_edge(base_candidate, detail_idx)
					edges_added += 1

		print(f"{datetime.now()} - Added a total of {edges_added} edges to graph")

		return G, grid_geojsons['combined_grid']

	def create_smooth_h3_transition(self, boundary_poly, base_resolution=5, detail_resolution=8):
		"""
		Creates a multi-resolution H3 grid with smooth transitions between resolutions.
		"""
		# Get base and detail hexagons
		base_hexagons = h3.polygon_to_cells(boundary_poly.__geo_interface__, base_resolution)
		detail_areas = self._get_detail_areas()  # Your function to identify harbor/approach areas
		detail_hexagons = []
		for area in detail_areas:
			detail_hexagons.extend(h3.polygon_to_cells(area.__geo_interface__, detail_resolution))

		# Calculate how many transition levels we need
		transition_levels = detail_resolution - base_resolution
		all_hexagons = set(base_hexagons)

		# Find parent cells of detailed hexagons at base resolution
		base_parents = set()
		for hex_id in detail_hexagons:
			parent = h3.cell_to_parent(hex_id, base_resolution)
			base_parents.add(parent)

		# Create transition rings at each intermediate resolution
		for i in range(1, transition_levels):
			current_res = base_resolution + i
			# For each base cell at the boundary
			transition_ring = set()

			# Get ring of cells around detailed area
			for parent in base_parents:
				# Get neighbors at base resolution
				neighbors = h3.grid_disk(parent, 1)
				for neighbor in neighbors:
					# If neighbor is not already a parent of a detailed cell
					if neighbor not in base_parents:
						# Add children at the current transition resolution
						children = h3.cell_to_children(neighbor, current_res)
						transition_ring.update(children)

			# Add this ring to our collection
			all_hexagons.update(transition_ring)

		# Add detailed hexagons
		all_hexagons.update(detail_hexagons)

		# Remove base hexagons that have been replaced by higher resolution cells
		for h in base_parents:
			if h in all_hexagons:
				all_hexagons.remove(h)

		return list(all_hexagons)

	def fix_h3_graph_directionality(self, G):
		"""
		Fixes directionality issues in an H3 graph by ensuring all edges are bidirectional.
		For each edge (u,v) that doesn't have a corresponding (v,u), adds the missing edge
		with the same weight and attributes.

		Parameters:
			G (nx.Graph): The H3 graph to fix

		Returns:
			nx.Graph: The fixed graph
		"""
		print(f"{datetime.now()} - Checking H3 graph directionality...")

		# Create a copy to avoid modifying the original during iteration
		G_fixed = G.copy()

		# Track edges to add
		edges_to_add = []

		# Check each edge
		for u, v, data in G.edges(data=True):
			if not G.has_edge(v, u):
				# Edge is directional, need to add the reverse
				edges_to_add.append((v, u, data.copy()))

		# Add missing edges
		if edges_to_add:
			print(f"{datetime.now()} - Adding {len(edges_to_add)} missing reverse edges")
			G_fixed.add_edges_from(edges_to_add)
		else:
			print(f"{datetime.now()} - All edges are already bidirectional")

		return G_fixed



class Weights:
	def __init__(self, enc_schema_name: str, graph_schema_name:str):
		self.pg = PostGIS()
		self.enc_schema = enc_schema_name
		self.graph_schema = graph_schema_name

	def pg_create_node_features(self, graph_name="base"):
		"""
		Creates new node feature columns by analyzing depth area (from the 'depare' table)
		and dredged area (from the 'drgare' table). It adds (if missing) and populates:
		  • depare_feature – minimum depth value from 'depare' (using attribute drval1)
		  • drgare_feature  – minimum dredged value from 'drgare' (using attribute dred_val)
		  • min_feature     – the minimum of the two above values for each node.
		"""
		nodes_table = f"graph_nodes_{graph_name}" if graph_name != "base" else "graph_nodes"

		schema = self.graph_schema
		enc_schema = self.enc_schema  # ENC schema in which the 'depare' and 'drgare' tables reside

		with self.pg.connect() as conn:
			# Add columns to the nodes table if they do not exist
			conn.execute(text(f'''
					ALTER TABLE "{schema}"."{nodes_table}" 
					ADD COLUMN IF NOT EXISTS dval_depare FLOAT;
				'''))
			conn.execute(text(f'''
					ALTER TABLE "{schema}"."{nodes_table}" 
					ADD COLUMN IF NOT EXISTS dval_drgare FLOAT;
				'''))
			conn.execute(text(f'''
					ALTER TABLE "{schema}"."{nodes_table}" 
					ADD COLUMN IF NOT EXISTS min_dval FLOAT;
				'''))
			conn.commit()
			print(f"{datetime.now()} -  Added columns to the nodes table")

			# Update depare_feature from the "depare" ENC layer using its drval1 attribute
			update_depare_sql = f'''
					UPDATE "{schema}"."{nodes_table}" n
					SET dval_depare = sub.min_depth
					FROM (
						SELECT n.id AS node_id,
							   MIN(l.drval1::FLOAT) AS min_depth
						FROM "{schema}"."{nodes_table}" n
						JOIN "{enc_schema}"."depare" l 
						  ON ST_Intersects(l.wkb_geometry, n.geom)
						GROUP BY n.id
					) sub
					WHERE n.id = sub.node_id;
				'''
			conn.execute(text(update_depare_sql))
			print(f"{datetime.now()} -  Depth Area column updated")

			# Update drgare_feature from the "drgare" ENC layer using its dred_val attribute
			update_drgare_sql = f'''
					UPDATE "{schema}"."{nodes_table}" n
					SET dval_drgare = sub.min_dredge
					FROM (
						SELECT n.id AS node_id,
							   MIN(l.drval1::FLOAT) AS min_dredge
						FROM "{schema}"."{nodes_table}" n
						JOIN "{enc_schema}"."drgare" l 
						  ON ST_Intersects(l.wkb_geometry, n.geom)
						GROUP BY n.id
					) sub
					WHERE n.id = sub.node_id;
				'''
			conn.execute(text(update_drgare_sql))
			print(f"{datetime.now()} -  Dredged Area column updated")

			# Update min_feature as the least (minimum) of depare_feature and drgare_feature
			update_min_sql = f'''
					UPDATE "{schema}"."{nodes_table}"
					SET min_dval = LEAST(dval_depare, dval_drgare);
				'''
			conn.execute(text(update_min_sql))
			conn.commit()

		print("Node features Depth Area, Dredged Area, and min_dval have been created and populated.")

	def pg_hybrid_edge_schema(self, graph_name="base", static_layers=None, dynamic_properties=None):
		"""
		Enhances the graph edges table with columns for both:
		1. Pre-calculated weights for static features
		2. Raw property values for dynamic features (draft-dependent, vessel-dependent)

		Parameters:
			graph_name (str): Name of the graph to enhance
			static_layers (dict): Dictionary of static layers to include as weight columns
			dynamic_properties (dict): Dictionary of properties to store raw values for
		"""
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Add base weight columns if they don't exist
		base_columns_sql = f"""
		DO $$ 
		BEGIN
			-- Check and add base_weight
			IF NOT EXISTS (
				SELECT 1 FROM information_schema.columns 
				WHERE table_schema = '{self.graph_schema}' 
				AND table_name = '{edges_table}' 
				AND column_name = 'base_weight'
			) THEN
				ALTER TABLE "{self.graph_schema}"."{edges_table}" ADD COLUMN base_weight FLOAT;
				UPDATE "{self.graph_schema}"."{edges_table}" SET base_weight = weight;
			END IF;

			-- Check and add static_weight_factor
			IF NOT EXISTS (
				SELECT 1 FROM information_schema.columns 
				WHERE table_schema = '{self.graph_schema}' 
				AND table_name = '{edges_table}' 
				AND column_name = 'static_weight_factor'
			) THEN
				ALTER TABLE "{self.graph_schema}"."{edges_table}" ADD COLUMN static_weight_factor FLOAT DEFAULT 1.0;
			END IF;

			-- Check and add dynamic_factor
			IF NOT EXISTS (
				SELECT 1 FROM information_schema.columns 
				WHERE table_schema = '{self.graph_schema}' 
				AND table_name = '{edges_table}' 
				AND column_name = 'dynamic_factor'
			) THEN
				ALTER TABLE "{self.graph_schema}"."{edges_table}" ADD COLUMN dynamic_factor FLOAT DEFAULT 1.0;
			END IF;

			-- Check and add adjusted_weight
			IF NOT EXISTS (
				SELECT 1 FROM information_schema.columns 
				WHERE table_schema = '{self.graph_schema}' 
				AND table_name = '{edges_table}' 
				AND column_name = 'adjusted_weight'
			) THEN
				ALTER TABLE "{self.graph_schema}"."{edges_table}" ADD COLUMN adjusted_weight FLOAT;
				UPDATE "{self.graph_schema}"."{edges_table}" SET adjusted_weight = weight;
			END IF;
		END
		$$;
		"""

		# Generate SQL for static feature weight columns
		static_columns_sql = []
		if static_layers:
			for layer_name in static_layers:
				col_name = f"wt_{layer_name.lower()}"
				static_sql = f"""
				DO $$
				BEGIN
					IF NOT EXISTS (
						SELECT 1 FROM information_schema.columns 
						WHERE table_schema = '{self.graph_schema}' 
						AND table_name = '{edges_table}' 
						AND column_name = '{col_name}'
					) THEN
						ALTER TABLE "{self.graph_schema}"."{edges_table}" 
						ADD COLUMN {col_name} FLOAT DEFAULT 1.0;
					END IF;
				END
				$$;
				"""
				static_columns_sql.append(static_sql)

		# Generate SQL for dynamic property columns (storing the raw feature values)
		dynamic_columns_sql = []
		if dynamic_properties:
			for prop_name, prop_config in dynamic_properties.items():
				col_name = f"ft_{prop_name.lower()}"
				data_type = prop_config.get('data_type', 'FLOAT')
				dynamic_sql = f"""
					DO $$
					BEGIN
						IF NOT EXISTS (
							SELECT 1 FROM information_schema.columns 
							WHERE table_schema = '{self.graph_schema}' 
							AND table_name = '{edges_table}' 
							AND column_name = '{col_name}'
						) THEN
							ALTER TABLE "{self.graph_schema}"."{edges_table}" 
							ADD COLUMN {col_name} {data_type};
						END IF;
					END
					$$;
					"""
				dynamic_columns_sql.append(dynamic_sql)

		# NEW: Generate SQL for dynamic weight columns (for final weight factors)
		dynamic_weight_columns_sql = []
		if dynamic_properties:
			for prop_name, prop_config in dynamic_properties.items():
				wt_col_name = f"wt_{prop_name.lower()}"
				# You could allow different data types for weight columns via config; here default to FLOAT with DEFAULT 1.0.
				wt_data_type = prop_config.get('weight_data_type', 'FLOAT')
				dynamic_wt_sql = f"""
					DO $$
					BEGIN
						IF NOT EXISTS (
							SELECT 1 FROM information_schema.columns 
							WHERE table_schema = '{self.graph_schema}' 
							AND table_name = '{edges_table}' 
							AND column_name = '{wt_col_name}'
						) THEN
							ALTER TABLE "{self.graph_schema}"."{edges_table}" 
							ADD COLUMN {wt_col_name} {wt_data_type} DEFAULT 1.0;
						END IF;
					END
					$$;
					"""
				dynamic_weight_columns_sql.append(dynamic_wt_sql)

		with self.pg.connect() as conn:
			# Add base columns
			conn.execute(text(base_columns_sql))

			# Add static weight columns
			if static_columns_sql:
				for sql in static_columns_sql:
					conn.execute(text(sql))

			# Add dynamic property columns
			if dynamic_columns_sql:
				for sql in dynamic_columns_sql:
					conn.execute(text(sql))

			# Add dynamic weight columns for each dynamic property
			if dynamic_weight_columns_sql:
				for sql in dynamic_weight_columns_sql:
					conn.execute(text(sql))

			conn.commit()

		print(f"{datetime.now()} - Updated graph schema with hybrid weight management approach")

		# Return lists of column names that were added
		static_columns = []
		dynamic_columns = []

		if static_layers:
			static_columns = [f"wt_{layer_name.lower()}" for layer_name in static_layers]
			print(f"Added static weight columns: {', '.join(static_columns)}")

		if dynamic_properties:
			dynamic_columns = [f"ft_{prop_name.lower()}" for prop_name in dynamic_properties]
			print(f"Added dynamic property columns: {', '.join(dynamic_columns)}")

		return static_columns, dynamic_columns

	def pg_static_edge_weights(self, graph_name="base", static_layers=None, usage_bands=None):
		"""
		Calculates weight factors for static features and stores them in the graph.

		Parameters:
			graph_name (str): Name of the graph to modify
			static_layers (dict): Dictionary mapping static layer names to their weight factors
			usage_bands (list): List of usage bands to include (e.g., ['1','3','4'])
		"""
		if static_layers is None:
			static_layers = {
				'lndare': {'attr': None, 'factor': 999.0},  # Land - impassable
				'obstrn': {'attr': None, 'factor': 5.0},  # Obstruction
				'pilpnt': {'attr': None, 'factor': 0.9},  # Pilot boarding places - favorable
				'morfac': {'attr': None, 'factor': 3.0},  # Mooring facility
				'bcnlat': {'attr': None, 'factor': 2.0}  # Lateral beacons
			}

		# Default usage bands (all) if not specified
		if usage_bands is None:
			usage_bands = ['3', '4', '5', '6']

		# Format the usage bands for SQL IN clause
		usage_bands_str = "'" + "','".join(usage_bands) + "'"

		# Ensure the enhanced schema exists with columns for static layers
		static_columns, _ = self.pg_hybrid_edge_schema(graph_name, static_layers)

		# Determine table name
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Reset all static weight columns to 1.0 (neutral factor)
		reset_weights_sql = ", ".join([f"{col} = 1.0" for col in static_columns]) if static_columns else ""
		if reset_weights_sql:
			reset_sql = f"""
			UPDATE "{self.graph_schema}"."{edges_table}"
			SET {reset_weights_sql};
			"""
			with self.pg.connect() as conn:
				conn.execute(text(reset_sql))
				conn.commit()

		# Process each static layer
		for layer_name, config in static_layers.items():
			col_name = f"wt_{layer_name.lower()}"
			factor = config.get('factor', 1.0)

			# SQL to apply weight factor if edge intersects with this feature
			weight_sql = f"""
			UPDATE "{self.graph_schema}"."{edges_table}" e
			SET {col_name} = {factor}
			FROM "{self.enc_schema}"."{layer_name}" l
			WHERE ST_Intersects(e.geom, l.wkb_geometry)
			AND substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str});
			"""

			# Execute the update
			with self.pg.connect() as conn:
				conn.execute(text(weight_sql))
				conn.commit()

		# Calculate combined static weight factor from all static weight columns
		with self.pg.connect() as conn:
			# Get all static weight columns
			get_cols_sql = f"""
			SELECT column_name FROM information_schema.columns 
			WHERE table_schema = '{self.graph_schema}' 
			AND table_name = '{edges_table}'
			AND column_name LIKE 'wt_%';
			"""

			weight_cols = [row[0] for row in conn.execute(text(get_cols_sql)).fetchall()]

			# Build the multiplication expression for all static weight columns
			if weight_cols:
				weight_expr = " * ".join(weight_cols)

				# Update static_weight_factor
				update_sql = f"""
				UPDATE "{self.graph_schema}"."{edges_table}"
				SET static_weight_factor = {weight_expr}
				"""
				conn.execute(text(update_sql))
				conn.commit()

		print(f"{datetime.now()} - Successfully applied static feature weights to graph '{graph_name}'")

	def pg_dynamic_edge_features(self, graph_name="base", dynamic_properties=None, usage_bands=None):
		"""
		Populates raw property values for dynamic features that depend on vessel characteristics.

		Parameters:
			graph_name (str): Name of the graph to modify
			dynamic_properties (dict): Dictionary of dynamic properties to populate
			usage_bands (list): List of usage bands to include
		"""

		if dynamic_properties is None:
			dynamic_properties = {
				'depth_min': {
					'layer': 'depare',
					'attr': 'drval1',
					'data_type': 'FLOAT',
					'query_type': 'min'
				},
				'depth_contour': {
					'layer': 'depcnt',
					'attr': 'valdco',
					'data_type': 'FLOAT[]',
					'query_type': 'array'
				},
				'anchorage_category': {
					'layer': 'achare',
					'attr': 'catach',
					'data_type': 'VARCHAR[]',
					'query_type': 'value'
				},
				'wreck': {
					'layer': 'wrecks',
					'attr': 'valsou',
					'data_type': 'FLOAT',
					'query_type': 'min'
				}
			}

		# Default usage bands (all) if not specified
		if usage_bands is None:
			usage_bands = ['3', '4', '5', '6']

		# Format the usage bands for SQL IN clause
		usage_bands_str = "'" + "','".join(usage_bands) + "'"

		# Ensure the enhanced schema exists with columns for dynamic properties
		_, dynamic_columns = self.pg_hybrid_edge_schema(graph_name, None, dynamic_properties)

		# Determine table name
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Process each dynamic property
		for prop_name, config in dynamic_properties.items():
			layer = config['layer']
			attr = config['attr']
			col_name = f"ft_{prop_name.lower()}"
			query_type = config.get('query_type', 'value')

			# Construct query based on query_type
			if query_type == 'min':
				# Get minimum value of the attribute where edge intersects feature
				query_sql = f"""
				UPDATE "{self.graph_schema}"."{edges_table}" e
				SET {col_name} = subquery.min_val
				FROM (
					SELECT e.id, MIN(l.{attr}::float) as min_val
					FROM "{self.graph_schema}"."{edges_table}" e
					JOIN "{self.enc_schema}"."{layer}" l ON ST_Intersects(e.geom, l.wkb_geometry)
					WHERE substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str})
					GROUP BY e.id
				) subquery
				WHERE e.id = subquery.id;
				"""
			elif query_type == 'array':
				# Collect all values as an array
				query_sql = f"""
				UPDATE "{self.graph_schema}"."{edges_table}" e
				SET {col_name} = subquery.val_array
				FROM (
					SELECT e.id, array_agg(DISTINCT l.{attr}::float) as val_array
					FROM "{self.graph_schema}"."{edges_table}" e
					JOIN "{self.enc_schema}"."{layer}" l ON ST_Intersects(e.geom, l.wkb_geometry)
					WHERE substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str})
					GROUP BY e.id
				) subquery
				WHERE e.id = subquery.id;
				"""
			else:  # 'value'
				# Get the attribute value (for categorical properties)
				if query_type == 'value':
					# Modify the data type if handling arrays
					if prop_name == 'anchorage_category':
						# Keep the array as is without trying to cast to integer
						query_sql = f"""
						UPDATE "{self.graph_schema}"."{edges_table}" e
						SET {col_name} = l.{attr}
						FROM "{self.enc_schema}"."{layer}" l
						WHERE ST_Intersects(e.geom, l.wkb_geometry)
						AND substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str});
						"""
					else:
						# For non-array types, use the original approach
						query_sql = f"""
						UPDATE "{self.graph_schema}"."{edges_table}" e
						SET {col_name} = l.{attr}::integer
						FROM "{self.enc_schema}"."{layer}" l
						WHERE ST_Intersects(e.geom, l.wkb_geometry)
						AND substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str});
						"""

			# Execute the update
			with self.pg.connect() as conn:
				conn.execute(text(query_sql))
				conn.commit()

		print(f"{datetime.now()} - Successfully populated dynamic properties for graph '{graph_name}'")

	def pg_calculate_dynamic_weights(self, graph_name="base", vessel_parameters=None):
		"""
		Calculates dynamic edge weights in PostGIS based on vessel parameters.
		Instead of iterating over edges in Python, this function updates the graph_edges table
		using a series of SQL UPDATE statements.

		Parameters:
		  graph_name (str): Name of the graph. For the default graph use "base"; otherwise, a suffix is appended.
		  vessel_parameters (dict): Vessel parameters containing:
				- draft (float): Vessel draft in meters.
				- vessel_type (str): Vessel type (e.g. "cargo", "passenger").
				- safety_margin (float): Additional safety margin in meters.
		Returns:
		  bool: True if the updates succeeded, False otherwise.
		"""
		if vessel_parameters is None:
			vessel_parameters = {}
		vessel_type = vessel_parameters.get("vessel_type", "cargo")
		draft = vessel_parameters.get("draft", 8.0)
		safety_margin = vessel_parameters.get("safety_margin", 1.0)
		safe_depth = draft + safety_margin

		# Determine the name of the edges table
		if graph_name == "base":
			edges_table = "graph_edges"
		else:
			edges_table = f"graph_edges_{graph_name}"

		try:
			with self.pg.connect() as conn:
				conn.execute(text("BEGIN;"))
				# 1. Update dynamic_factor based on ft_depth_min.
				#    - If ft_depth_min is below vessel draft, set factor to 999.0 (impassable).
				#    - If ft_depth_min is between draft and safe_depth, scale factor between 1.0 and 5.0.
				#    - Otherwise, set factor to 0.9 (favor deeper water).
				update_depth_sql = f"""
				           UPDATE "{self.graph_schema}"."{edges_table}"
				           SET 
				               wt_depth_min = 
				                   CASE 
				                       -- Calculate effective depth (prioritize dredged area values)
				                       WHEN (
				                           COALESCE(ft_drgare, ft_depth_min) - :draft
				                       ) <= 0 THEN 999.0  -- Impassable if UKC <= 0

				                       -- Scale weight based on under-keel clearance when between 0 and safety margin
				                       WHEN (
				                           COALESCE(ft_drgare, ft_depth_min) - :draft
				                       ) < :safety_margin THEN 
				                           1.0 + (4.0 * ((:safety_margin - (COALESCE(ft_drgare, ft_depth_min) - :draft)) / :safety_margin))

				                       -- Slightly favor deeper water when UKC is good
				                       ELSE 0.9
				                   END,

				               -- Store the calculated UKC for reference
				               dynamic_factor = 
				                   CASE 
				                       WHEN COALESCE(ft_drgare, ft_depth_min) IS NOT NULL THEN 
				                           CASE 
				                               WHEN (COALESCE(ft_drgare, ft_depth_min) - :draft) <= 0 THEN 999.0
				                               WHEN (COALESCE(ft_drgare, ft_depth_min) - :draft) < :safety_margin 
				                                   THEN 1.0 + (4.0 * ((:safety_margin - (COALESCE(ft_drgare, ft_depth_min) - :draft)) / :safety_margin))
				                               ELSE 0.9
				                           END
				                       ELSE 1.0
				                   END
				           WHERE ft_depth_min IS NOT NULL OR ft_drgare IS NOT NULL;
				           """
				conn.execute(text(update_depth_sql),
				             {"draft": draft, "safe_depth": safe_depth, "safety_margin": safety_margin})

				# 2. Update dynamic_factor based on depth contours.
				#    If any value in the ft_depth_contour array is shallower than safe_depth,
				#    ensure the factor is at least 1.5.
				update_contour_sql = f"""
				UPDATE "{self.graph_schema}"."{edges_table}"
				SET dynamic_factor = CASE 
						WHEN EXISTS (
							SELECT 1 FROM unnest(ft_depth_contour) AS contour 
							WHERE contour::float < :safe_depth
						) THEN GREATEST(dynamic_factor, 1.5)
						ELSE dynamic_factor
					END
				WHERE ft_depth_contour IS NOT NULL;
				"""
				conn.execute(text(update_contour_sql), {"safe_depth": safe_depth})

				# 3. Apply anchorage-category adjustments.
				#    For cargo vessels, favor categories 1 and 2 using a multiplier of 0.8;
				#    for passenger vessels, favor categories 5 and 6 using a multiplier of 0.7.
				if vessel_type.lower() == "cargo":
					preferred = "1,2"
					multiplier = 0.8
				elif vessel_type.lower() == "passenger":
					preferred = "5,6"
					multiplier = 0.7
				else:
					preferred = None
					multiplier = 1.0
				if preferred:
					update_anchorage_sql = f"""
					UPDATE "{self.graph_schema}"."{edges_table}"
					SET dynamic_factor = dynamic_factor * :multiplier
					WHERE ft_anchorage_category IS NOT NULL 
					  AND EXISTS (
						  SELECT 1 FROM unnest(ft_anchorage_category) AS cat 
						  WHERE cat::int IN ({preferred})
					  );
					"""
					conn.execute(text(update_anchorage_sql), {"multiplier": multiplier})

				# 4. Apply hazard adjustments for wrecks.
				#    If ft_wreck is present and is less than safe_depth,
				#    ensure the dynamic_factor is at least 8.0.
				update_wreck_sql = f"""
				UPDATE "{self.graph_schema}"."{edges_table}"
				SET dynamic_factor = GREATEST(dynamic_factor, 8.0)
				WHERE ft_wreck IS NOT NULL AND ft_wreck < :safe_depth;
				"""
				conn.execute(text(update_wreck_sql), {"safe_depth": safe_depth})

				# 5. Update final adjusted weight based on base_weight, static_weight_factor, and dynamic_factor.
				update_final_sql = f"""
				UPDATE "{self.graph_schema}"."{edges_table}"
				SET adjusted_weight = base_weight * static_weight_factor * dynamic_factor,
					weight = base_weight * static_weight_factor * dynamic_factor;
				"""
				conn.execute(text(update_final_sql))
				conn.execute(text("COMMIT;"))
			print("Dynamic weights calculated and updated successfully in PostGIS.")
			return True
		except Exception as e:
			with self.pg.connect() as conn:
				conn.execute(text("ROLLBACK;"))
			print(f"Error calculating dynamic weights in PostGIS: {str(e)}")
			return False

	def calculate_dynamic_weights(self, graph, vessel_parameters):
		"""
		Dynamically calculates weights based on vessel parameters and stored properties.
		This function operates on a loaded NetworkX graph, not directly on the database.

		Parameters:
			graph (nx.Graph): NetworkX graph loaded from the database
			vessel_parameters (dict): Dictionary containing vessel parameters:
				- draft: Vessel draft in meters
				- vessel_type: Type of vessel (e.g., 'cargo', 'tanker', 'passenger')
				- safety_margin: Safety margin for depth calculations (default: 2.0m)

		Returns:
			nx.Graph: Graph with updated edge weights
		"""
		# Extract vessel parameters with defaults
		vessel_type = vessel_parameters.get('vessel_type', 'cargo')
		draft = vessel_parameters.get('draft', 8.0)
		safety_margin = vessel_parameters.get('safety_margin', 1.0)

		# Calculate minimum safe depth
		safe_depth = draft + safety_margin

		# Create a copy of the graph to avoid modifying the original
		G = graph.copy()

		# Process each edge
		for u, v, data in G.edges(data=True):
			# Get static weight factor (default to 1.0 if not present)
			static_factor = data.get('static_weight_factor', 1.0)

			# Initialize dynamic factor to 1.0
			dynamic_factor = 1.0

			# Apply depth constraints
			if 'ft_depth_min' in data:
				depth_min = data['ft_depth_min']
				if depth_min is not None:
					if depth_min < draft:
						# Impassable - set very high weight
						dynamic_factor = 999.0
					elif depth_min < safe_depth:
						# Unsafe - scale weight based on how close to minimum draft
						safety_ratio = (safe_depth - depth_min) / safety_margin
						dynamic_factor = 1.0 + (4.0 * safety_ratio)  # Scale from 1.0 to 5.0
					else:
						# Safe depth - slightly favor deeper water
						dynamic_factor = 0.9

			# Apply depth contour check if available
			if 'ft_depth_contour' in data and data['ft_depth_contour']:
				try:
					contours = data['ft_depth_contour']
					# If there are shallow contours, adjust weight
					if any(c < safe_depth for c in contours):
						shallow_factor = 1.5
						dynamic_factor = max(dynamic_factor, shallow_factor)
				except (TypeError, ValueError):
					# Handle case where contours aren't properly formatted
					pass

			# Apply anchorage category adjustments
			if 'ft_anchorage_category' in data and data['ft_anchorage_category'] is not None:
				# Handle the case where catach is an array
				cat_values = data['ft_anchorage_category']

				# Check if any values in the array match preferred anchorage types
				if isinstance(cat_values, list):
					preferred_categories = [1, 2] if vessel_type == 'cargo' else [5,
																				  6] if vessel_type == 'passenger' else []
					if any(int(cat) in preferred_categories for cat in cat_values if cat is not None):
						factor = 0.7 if vessel_type == 'passenger' else 0.8
						dynamic_factor *= factor

			# Apply isolated danger adjustments
			if 'ft_wreck' in data and data['ft_wreck'] is not None:
				danger_depth = data['ft_wreck']
				if danger_depth < safe_depth:
					# Avoid hazards with depth less than safe depth
					dynamic_factor = max(dynamic_factor, 8.0)

			# Calculate final weight adjustment
			base_weight = data.get('base_weight', data.get('weight', 1.0))
			adjusted_weight = base_weight * static_factor * dynamic_factor

			# Update edge with the calculated weights
			G.edges[u, v]['dynamic_factor'] = dynamic_factor
			G.edges[u, v]['adjusted_weight'] = adjusted_weight
			G.edges[u, v]['weight'] = adjusted_weight  # Set main weight for path algorithms

		print(
			f"{datetime.now()} - Applied dynamic weight adjustments for vessel: {vessel_type}, draft: {draft}m, safety margin: {safety_margin}m")
		return G

	def pg_update_wind_current_directions(self, graph_name="base", timestamp=None):
		"""
		Updates wind and current directional factors based on forecast time.

		Parameters:
			graph_name (str): Name of the graph to modify
			timestamp (datetime): Forecast timestamp (uses current time if None)
		"""
		if timestamp is None:
			timestamp = datetime.now()

		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Query wind and current data for the specified time
		# (This assumes you have tables with wind/current forecasts)
		update_wind_sql = f"""
		UPDATE "{self.graph_schema}"."{edges_table}" e
		SET dir_fwd_wind = 
			CASE
				WHEN cosine_similarity(
					ST_Azimuth(ST_StartPoint(e.geom), ST_EndPoint(e.geom)),
					w.direction
				) > 0.7 THEN 0.8  -- With the wind (20% reduction)
				WHEN cosine_similarity(
					ST_Azimuth(ST_StartPoint(e.geom), ST_EndPoint(e.geom)),
					w.direction
				) < -0.7 THEN 1.5  -- Against the wind (50% increase)
				ELSE 1.0  -- Neutral
			END,
		dir_rev_wind = 
			CASE
				WHEN cosine_similarity(
					ST_Azimuth(ST_EndPoint(e.geom), ST_StartPoint(e.geom)),
					w.direction
				) > 0.7 THEN 0.8
				WHEN cosine_similarity(
					ST_Azimuth(ST_EndPoint(e.geom), ST_StartPoint(e.geom)),
					w.direction
				) < -0.7 THEN 1.5
				ELSE 1.0
			END
		FROM wind_forecast w
		WHERE ST_Intersects(e.geom, w.geom)
		AND w.forecast_time = :timestamp;
		"""

		# Similar approach for currents...

		with self.pg.connect() as conn:
			conn.execute(text(update_wind_sql), {"timestamp": timestamp})
			# Execute current SQL
			conn.commit()

	def pg_delete_node_feature_columns(self, graph_name: str = "base"):
		"""
		Deletes all columns from the nodes table except 'node' and 'geom'.
		This is useful for cleaning up extra attributes that might have been added.
		"""
		# Retrieve the list of columns for the given nodes table in the graph schema.
		nodes_table = f"graph_nodes_{graph_name}" if graph_name != "base" else "graph_nodes"
		query = """
			  SELECT column_name 
			  FROM information_schema.columns 
			  WHERE table_schema = :schema 
				AND table_name = :table
		  """
		with self.pg.connect() as conn:
			result = conn.execute(text(query), {"schema": self.graph_schema, "table": nodes_table})
			columns = [row[0] for row in result.fetchall()]

			# Identify columns to drop (i.e. all columns except 'node' and 'geom')
			columns_to_drop = [col for col in columns if col not in ['id','node', 'geom']]
			if columns_to_drop:
				drop_stmt = ", ".join([f'DROP COLUMN IF EXISTS "{col}"' for col in columns_to_drop])
				alter_sql = f'ALTER TABLE "{self.graph_schema}"."{nodes_table}" {drop_stmt};'
				conn.execute(text(alter_sql))
				conn.commit()
				print(f"Deleted columns {columns_to_drop} from {nodes_table}")
			else:
				print("No extra columns to delete.")

	def pg_delete_edge_feature_columns(self, graph_name="base", columns=None, column_type=None):
		"""
		Deletes specific weight columns from the graph edges table.

		Parameters:
			graph_name (str): Name of the graph to modify
			columns (list): List of specific column names to delete (will override column_type)
			column_type (str): Type of columns to delete ('static', 'dynamic', or 'all')
							  - 'static' removes all 'wt_*' columns
							  - 'dynamic' removes all 'prop_*' columns
							  - 'all' removes both types

		Returns:
			bool: True if operation succeeded, False otherwise
		"""
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Determine columns to delete
		target_columns = []

		if columns is not None:
			# Specific columns provided - use these directly
			target_columns = columns
		elif column_type is not None:
			# Query the database to get all weight-related columns
			with self.pg.connect() as conn:
				query = f"""
				SELECT column_name FROM information_schema.columns 
				WHERE table_schema = '{self.graph_schema}' 
				AND table_name = '{edges_table}'
				"""

				if column_type == 'static':
					query += " AND column_name LIKE 'wt_%'"
				elif column_type == 'dynamic':
					query += " AND column_name LIKE 'ft_%'"
				elif column_type == 'all':
					query += " AND (column_name LIKE 'wt_%' OR column_name LIKE 'ft_%')"
				else:
					print(f"Invalid column_type: {column_type}. Must be 'static', 'dynamic', or 'all'.")
					return False

				result = conn.execute(text(query))
				target_columns = [row[0] for row in result]
		else:
			print("Error: Either 'columns' or 'column_type' must be provided.")
			return False

		if not target_columns:
			print("No weight columns found to delete.")
			return False

		# Generate ALTER TABLE statement to drop the columns
		drop_columns = []
		for col in target_columns:
			drop_columns.append(f'DROP COLUMN IF EXISTS "{col}"')

		alter_sql = f"""
		ALTER TABLE "{self.graph_schema}"."{edges_table}"
		{', '.join(drop_columns)};
		"""

		try:
			with self.pg.connect() as conn:
				conn.execute(text(alter_sql))
				conn.commit()

			print(f"Successfully deleted the following columns from {edges_table}:")
			for col in target_columns:
				print(f"  - {col}")

			return True
		except Exception as e:
			print(f"Error deleting weight columns: {str(e)}")
			return False

	def pg_directional_features(self, graph_name="base", direction_features=None, usage_bands=None):
		"""
		Processes directional features like Traffic Separation Schemes (TSS) where direction of travel matters.
		Adds forward and reverse headings to edges and applies appropriate weights based on feature orientation.
		"""
		# Setup logger
		logger = logging.getLogger(__name__)

		# Default parameters if none provided
		if direction_features is None:
			direction_features = {
				'tsslpt': {'attr': 'orient', 'penalty_against': 1000.0, 'penalty_cross': 5.0},
				# 'fairwy': {'attr': 'orient', 'penalty_against': 10.0, 'penalty_cross': 2.0}
			}
		if usage_bands is None:
			usage_bands = ['3', '4', '5']

		# Format usage bands for SQL IN clause
		usage_bands_str = "'" + "','".join(usage_bands) + "'"
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Consolidated SQL commands
		heading_columns_sql = f"""
			DO $$ 
			BEGIN
				IF NOT EXISTS (
					SELECT 1 FROM information_schema.columns 
					WHERE table_schema = :graph_schema 
					AND table_name = :edges_table 
					AND column_name = 'forward_heading'
				) THEN
					ALTER TABLE "{self.graph_schema}"."{edges_table}" 
					ADD COLUMN forward_heading double precision,
					ADD COLUMN reverse_heading double precision,
					ADD COLUMN wt_directional double precision DEFAULT 1.0,
					ADD COLUMN wt_rev_directional double precision DEFAULT 1.0;

					UPDATE "{self.graph_schema}"."{edges_table}"
					SET forward_heading = degrees(ST_Azimuth(ST_StartPoint(geom), ST_EndPoint(geom))),
						reverse_heading = degrees(ST_Azimuth(ST_EndPoint(geom), ST_StartPoint(geom)));
				END IF;
			END
			$$;
			"""

		angle_diff_function_sql = """
		CREATE OR REPLACE FUNCTION angle_diff(a1 double precision, a2 double precision) RETURNS double precision AS $$
		DECLARE
		  diff1 double precision;
		  diff2 double precision;
		BEGIN
		  diff1 := a1 - a2 + 360 - floor((a1 - a2 + 360)/360)*360;
		  diff2 := a2 - a1 + 360 - floor((a2 - a1 + 360)/360)*360;
		  RETURN LEAST(ABS(diff1), ABS(diff2));
		END;
		$$ LANGUAGE plpgsql;
		"""

		reset_sql = f"""
		UPDATE "{self.graph_schema}"."{edges_table}"
		SET wt_directional = 1.0, wt_rev_directional = 1.0;
		"""

		try:
			# Open a single connection and begin a transaction
			with self.pg.connect() as conn:
				# Execute schema updates and function creation
				conn.execute(text(heading_columns_sql), {"graph_schema": self.graph_schema, "edges_table": edges_table})
				conn.execute(text(angle_diff_function_sql))
				conn.execute(text(reset_sql))

				# Process each directional feature
				for feature_name, config in direction_features.items():
					orientation_attr = config.get('attr', 'orient')
					penalty_against = config.get('penalty_against', 1000.0)
					penalty_cross = config.get('penalty_cross', 5.0)

					update_sql = f"""
					UPDATE "{self.graph_schema}"."{edges_table}" e
					SET 
						wt_directional = 
							CASE
								WHEN angle_diff(e.forward_heading::FLOAT, l.orient::FLOAT) <= 30 THEN 1.0
								WHEN angle_diff(e.forward_heading::FLOAT, l.orient::FLOAT) >= 150 THEN :penalty_against
								ELSE :penalty_cross
							END,
						wt_rev_directional =
							CASE
								WHEN angle_diff(e.reverse_heading::FLOAT, l.orient::FLOAT) <= 30 THEN 1.0
								WHEN angle_diff(e.reverse_heading::FLOAT, l.orient::FLOAT) >= 150 THEN :penalty_against
								ELSE :penalty_cross
							END
					FROM "{self.enc_schema}"."{feature_name}" l
					WHERE ST_Intersects(e.geom, l.wkb_geometry)
					  AND substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str})
					  AND l.{orientation_attr} IS NOT NULL;
					"""
					conn.execute(text(update_sql), {"penalty_against": penalty_against, "penalty_cross": penalty_cross})

				# Update static_weight_factor
				update_factor_sql = f"""
				UPDATE "{self.graph_schema}"."{edges_table}"
				SET static_weight_factor = static_weight_factor * wt_directional;
				"""
				conn.execute(text(update_factor_sql))
				conn.commit()

			logger.info(f"{datetime.now()} - Successfully processed directional features for graph '{graph_name}'")
			return True

		except Exception as e:
			logger.error(f"Error processing directional features for graph '{graph_name}': {e}")
			# Optionally, rollback if your library doesn't automatically rollback on exception
			return False
