import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

from shapely import wkt
from shapely.geometry import shape, mapping, Point, Polygon, LineString, MultiPolygon
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import Geometry
import networkx as nx
import numpy as np
from MARITIME_MODULE import Miscellaneous
import torch
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data as tourchData
import numpy as np
import geopandas as gpd
from GRAPH import BaseGraph, Misceleaneous

import math
import ast

from Data import PostGIS

pg = PostGIS()
misc = Miscellaneous()


class Route:
	def __init__(self, schema_name: str):
		self.route_schema = schema_name
		self.pg = PostGIS()

	# def haversine(self, lon1, lat1, lon2, lat2):
	# 	R = 3440.065
	# 	dlon = math.radians(lon2 - lon1)
	# 	dlat = math.radians(lat2 - lat1)
	# 	a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
	# 		dlon / 2) ** 2
	# 	c = 2 * math.asin(math.sqrt(a))
	# 	return R * c

	def base_route(self, Graph, departure_point: Point, arrival_point: Point) -> tuple:
		"""
		Computes the A* route from departure_point to arrival_point by loading the graph from PostGIS.
		Calculates the total distance from the computed route.
		Returns:
		  tuple: (route: LineString, total_distance: float)
		"""
		astar = Astar(Graph)
		route = astar.compute_route(departure_point, arrival_point)

		# Helper function to compute haversine distance between two points in nautical miles.
		# Earth radius in nautical miles (~3440.065 NM)


		# Compute total distance by summing the nautical mile distance for each segment along the route.
		total_distance_nm = 0.0
		coords = list(route.coords)
		for i in range(len(coords) - 1):
			lon1, lat1 = coords[i]
			lon2, lat2 = coords[i + 1]
			total_distance_nm += Misceleaneous.haversine(lon1, lat1, lon2, lat2)

		print(f"Total route distance: {total_distance_nm} nautical miles")
		return route, total_distance_nm

	def pg_get_route_names(self, table_name: str = "routes") -> list:
		"""
		Retrieves a list of route names from the specified routes table in the route schema.

		Parameters:
			table (str): Name of the routes table (default "routes").

		Returns:
			list: A list containing the route names.
		"""
		# Build SQL to select route names from the table
		query = f'SELECT route_name FROM "{self.route_schema}"."{table_name}"'
		with self.pg.connect() as conn:
			result = conn.execute(text(query))
			route_names = [row[0] for row in result.fetchall()]
		return route_names

	def load_route(self, route_name: str, route_schema_name: str = "ROUTE", table_name: str = "routes"):
		"""
		Loads a route from PostGIS by its name and returns it as a Shapely LineString.

		Parameters:
			route_name (str): The name of the route to load.
			schema (str): The PostGIS schema where the routes table is stored (default "ROUTE").
			table (str): The PostGIS table containing route information (default "routes").

		Returns:
			A Shapely LineString of the route if found; otherwise, None.
		"""
		query = text(f'''
	        SELECT ST_AsText(geom)
	        FROM "{route_schema_name}"."{table_name}"
	        WHERE route_name = :route_name
	    ''')
		with self.pg.connect() as conn:
			row = conn.execute(query, {"route_name": route_name}).fetchone()
			if row and row[0]:
				route_geom = wkt.loads(row[0])
				return route_geom
			else:
				print(f"Route '{route_name}' not found in {route_schema_name}.{table_name}")
				return None

	def pg_save_route(self, route: LineString, route_name: str = "Unnamed Route", schema_name = "ROUTE", table_name = "routes", overwrite: bool = False) -> None:
		"""
		Saves the provided route (as a LineString) into a PostGIS database.
		The route is saved in a custom schema "Route" within a table "routes".
		If a route_name is provided, it will be saved along with the route; otherwise, a default name is used.
		"""

		# SQL to create the schema if it doesn't exist.
		create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";'

		# SQL to create the routes table if it doesn't exist.
		create_table_sql = f"""
					CREATE TABLE IF NOT EXISTS "{schema_name}"."{table_name}" (
						id SERIAL PRIMARY KEY,
						route_name VARCHAR,
						geom GEOMETRY(LineString,4326)
					);
				"""

		with self.pg.connect() as conn:
			# Create schema and table if needed.
			conn.execute(text(create_schema_sql))
			conn.execute(text(create_table_sql))
			conn.commit()

			# Check if the route name already exists.
			check_sql = text(f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}" WHERE route_name = :route_name')
			result = conn.execute(check_sql, {"route_name": route_name}).fetchone()
			name_exists = result[0] > 0

			# If the name exists, update or generate a unique name.
			if name_exists:
				if overwrite:
					# Overwrite the existing route geometry.
					update_sql = text(f'''
								UPDATE "{schema_name}"."{table_name}"
								SET geom = ST_GeomFromText(:wkt,4326)
								WHERE route_name = :route_name
							''')
					conn.execute(update_sql, {"wkt": route.wkt, "route_name": route_name})
					conn.commit()
					print(f"Route '{route_name}' successfully overwritten in schema '{schema_name}'.")
					return
				else:
					# Generate a new route name by appending a suffix until a unique name is found.
					counter = 1
					new_route_name = f"{route_name}_{counter}"
					while True:
						result = conn.execute(check_sql, {"route_name": new_route_name}).fetchone()
						if result[0] == 0:
							route_name = new_route_name
							break
						counter += 1
						new_route_name = f"{route_name}_{counter}"

			# If the route name does not exist or a new name was generated, insert the new route.
			insert_sql = text(f'''
						INSERT INTO "{schema_name}"."{table_name}" (route_name, geom)
						VALUES (:route_name, ST_GeomFromText(:wkt,4326))
					''')
			conn.execute(insert_sql, {"route_name": route_name, "wkt": route.wkt})
			conn.commit()
			print(f"Route '{route_name}' successfully saved into schema '{schema_name}' in table '{table_name}'.")

class Astar:
	def __init__(self, graph):
		self.graph = graph

	@staticmethod
	def heuristic(node1, node2):
		"""
		Euclidean distance heuristic for A* route planning between two nodes.
		Each node is expected to be a tuple (lon, lat).
		"""
		return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

	def find_nearest_node(self, point):
		"""
		Finds the node in the graph that is closest to the given shapely Point.

		Parameters:
		  point (shapely.geometry.Point): Location from which to search.

		Returns:
		  tuple: A node (as a coordinate tuple) that is nearest to the point.
		"""
		best_node = None
		best_distance = float('inf')
		for node, data in self.graph.nodes(data=True):
			node_point = data.get('point')
			if node_point is None:
				continue
			dist = point.distance(node_point)
			if dist < best_distance:
				best_distance = dist
				best_node = node
		return best_node

	def compute_route(self, start_point, end_point, save_geojson: bool = False, output_file: str = "route.geojson"):
		"""
		Computes the shortest route between start_point and end_point using
		the A* algorithm on the provided graph.

		Parameters:
		  start_point (shapely.geometry.Point): The starting point.
		  end_point (shapely.geometry.Point): The destination point.
		  save_geojson (bool): If True, saves the resulting route in GeoJSON format.
		  output_file (str): File path to save the GeoJSON (default "route.geojson").

		Returns:
		  shapely.geometry.LineString: A LineString representing the computed route.

		Raises:
		  ValueError: If a route cannot be computed.
		"""
		start_node = self.find_nearest_node(start_point)
		print(f"Start Node: {start_node}")
		end_node = self.find_nearest_node(end_point)
		print(f"End Node: {end_node}")
		if start_node is None or end_node is None:
			raise ValueError("Could not find a nearest node for the given start or end point.")
		try:
			route = nx.astar_path(self.graph, start_node, end_node, heuristic=Astar.heuristic, weight='weight')
		except nx.NetworkXNoPath:
			raise ValueError("No route found between the specified start and end points.")

		# Combine Departure and Arrival points with the route
		full_route = [tuple(start_point.coords)[0]] + route + [tuple(end_point.coords)[0]]
		route_linestring = LineString(full_route)
		# Optionally convert the LineString to GeoJSON and save to a file
		if save_geojson:
			geojson_obj = mapping(route_linestring)
			with open(output_file, "w") as f:
				json.dump(geojson_obj, f)
			print(f"Route saved to {output_file}")

		return route_linestring




class AstarImproved(Astar):
	def safety_weight(self, n):
		"""
		Computes the safety weight based on the number of risky neighbors (n).
		Adjust this function based on how you determine risk.
		"""
		return 1 + 0.5 * (2 ** (n - 1))

	def sailing_cost(self, u, v):
		"""
		Custom edge cost function.
		Assumes each node has a 'pos' attribute as a (x, y) tuple and a 'safety' attribute.
		"""
		x1, y1 = self.graph.nodes[u]['pos']
		x2, y2 = self.graph.nodes[v]['pos']
		distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
		# Use the safety attribute of the destination node v.
		safety_value = self.graph.nodes[v].get('safety', 1)
		return distance * safety_value

	def pilot_quantity(self, start, target, current):
		"""
		Calculates the pilot quantity for the current node based on the angle
		between the start-target vector and the current-target vector.
		"""
		sx, sy = self.graph.nodes[start]['pos']
		tx, ty = self.graph.nodes[target]['pos']
		cx, cy = self.graph.nodes[current]['pos']

		# Vector from start to target
		st = (tx - sx, ty - sy)
		# Vector from current to target
		ct = (tx - cx, ty - cy)

		norm_st = math.sqrt(st[0] ** 2 + st[1] ** 2)
		norm_ct = math.sqrt(ct[0] ** 2 + ct[1] ** 2)

		# Avoid division by zero
		if norm_st == 0 or norm_ct == 0:
			return 4
		# Calculate sine of the angle using the magnitude of the cross product
		cross = abs(st[0] * ct[1] - st[1] * ct[0])
		sin_theta = cross / (norm_st * norm_ct)
		return 4 - sin_theta

	def improved_heuristic(self, current, target, start):
		"""
		Custom heuristic function that multiplies the Euclidean distance from the
		current node to the target by the pilot quantity.
		"""
		cx, cy = self.graph.nodes[current]['pos']
		tx, ty = self.graph.nodes[target]['pos']
		distance = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
		pq = self.pilot_quantity(start, target, current)
		return distance * pq

	def compute_route_improved(self, departure_point, arrival_point):
		"""
		Computes an improved route between departure_point and arrival_point using
		custom sailing cost and improved heuristic functions.
		The departure and arrival points are assumed to be shapely Point objects.
		Returns:
			A shapely.geometry.LineString representing the computed route.
		"""
		start_node = self.find_nearest_node(departure_point)
		end_node = self.find_nearest_node(arrival_point)
		if start_node is None or end_node is None:
			raise ValueError("Could not find a nearest node for the given departure or arrival point.")

		# Use NetworkX A* with the custom heuristic and sailing cost functions.
		route_nodes = nx.astar_path(
			self.graph,
			start_node,
			end_node,
			heuristic=lambda current, target: self.improved_heuristic(current, target, start_node),
			weight=lambda u, v, data=None: self.sailing_cost(u, v)
		)

		# Combine the departure and arrival coordinates with the route node sequence.
		full_route = [tuple(departure_point.coords)[0]] + route_nodes + [tuple(arrival_point.coords)[0]]
		return LineString(full_route)


class PgRoutingAStar(BaseGraph):
	"""
	Performs A* routing using pgRouting on maritime graphs stored in PostGIS.
	Extends BaseGraph to leverage existing graph management capabilities.
	"""

	def __init__(self, departure_port=None, arrival_port=None, port_boundary=None,
	             enc_schema_name: str = "enc", graph_schema_name: str = "graph",
	             route_schema_name: str = "routes"):

		super().__init__(
			departure_port=departure_port,
			arrival_port=arrival_port,
			port_boundary=port_boundary,
			enc_schema_name=enc_schema_name,
			graph_schema_name=graph_schema_name
		)
		self.route_schema = route_schema_name
		self._ensure_pgrouting_extensions()

	def _ensure_pgrouting_extensions(self):
		"""Ensures that the necessary pgRouting extension is installed in PostgreSQL."""
		with self.pg.connect() as conn:
			conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgrouting;"))
			conn.commit()
			print("Checked pgRouting extension")

	def find_route_astar(self, start_id: int, end_id: int,
	                     graph_name: str = "base",
	                     heuristic_factor: float = 1.0,
	                     use_adjusted_weights: bool = True,
	                     reverse_cost: bool = False) -> Dict:
		"""
		Finds a route using pgRouting's A* algorithm between two nodes.

		Parameters:
			start_id: ID of the start node
			end_id: ID of the end node
			graph_name: Name of the graph to use (base is default)
			heuristic_factor: Factor to apply to the heuristic (1.0 = default A* behavior)
			use_adjusted_weights: If True, uses adjusted_weight column instead of weight
			reverse_cost: If True, uses bidirectional cost calculation

		Returns:
			Dictionary containing route information and GeoJSON geometry
		"""
		print(f"{datetime.now()} - Executing A* routing from node {start_id} to node {end_id}")

		# Determine table names based on graph name
		nodes_table = f"graph_nodes_{graph_name}" if graph_name != "base" else "graph_nodes"
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Determine which weight column to use
		weight_col = "adjusted_weight" if use_adjusted_weights else "weight"

		# Prepare the pgRouting A* query
		astar_query = f"""
        WITH route AS (
            SELECT * FROM pgr_aStar(
                'SELECT id, 
                        source_id AS source, 
                        target_id AS target, 
                        {weight_col} AS cost,
                        {f"{weight_col} AS reverse_cost" if reverse_cost else "NULL AS reverse_cost"},
                        ST_X(ST_StartPoint(geom)) AS x1, 
                        ST_Y(ST_StartPoint(geom)) AS y1,
                        ST_X(ST_EndPoint(geom)) AS x2,
                        ST_Y(ST_EndPoint(geom)) AS y2
                 FROM "{self.graph_schema}"."{edges_table}"',
                :start_id, 
                :end_id,
                directed := {str(reverse_cost).lower()},
                heuristic := 2,
                factor := :heuristic_factor
            )
        ),
        edges_geom AS (
            SELECT 
                r.seq,
                r.path_seq,
                r.edge,
                r.node,
                r.cost,
                r.agg_cost,
                e.geom
            FROM route r
            LEFT JOIN "{self.graph_schema}"."{edges_table}" e ON r.edge = e.id
            WHERE r.edge != -1
            ORDER BY r.path_seq
        ),
        route_geom AS (
            SELECT ST_LineMerge(ST_Union(geom)) AS geometry
            FROM edges_geom
        )
        SELECT 
            json_build_object(
                'type', 'Feature',
                'geometry', ST_AsGeoJSON(rg.geometry)::json,
                'properties', json_build_object(
                    'total_cost', (SELECT SUM(cost) FROM edges_geom),
                    'total_distance', ST_Length(rg.geometry::geography)/1852, -- nautical miles
                    'num_edges', (SELECT COUNT(*) FROM edges_geom),
                    'start_id', :start_id,
                    'end_id', :end_id
                )
            ) AS geojson,
            (SELECT SUM(cost) FROM edges_geom) AS total_cost,
            (SELECT array_agg(edge) FROM edges_geom) AS edge_ids,
            (SELECT array_agg(node) FROM route) AS node_ids,
            rg.geometry
        FROM route_geom rg;
        """

		params = {
			"start_id": start_id,
			"end_id": end_id,
			"heuristic_factor": heuristic_factor
		}

		with self.pg.connect() as conn:
			try:
				result = conn.execute(text(astar_query), params).fetchone()
				if result:
					geojson, total_cost, edge_ids, node_ids, geometry = result

					# Convert to GeoDataFrame for additional analysis
					if geometry:
						gdf = gpd.GeoDataFrame(
							{'geometry': [geometry]},
							crs="EPSG:4326"
						)

						return {
							'geojson': geojson,
							'total_cost': total_cost,
							'edge_ids': edge_ids,
							'node_ids': node_ids,
							'gdf': gdf,
							'success': True
						}
					else:
						print(f"No route found between nodes {start_id} and {end_id}")
						return {'success': False, 'message': 'No route found'}
				else:
					print(f"No route found between nodes {start_id} and {end_id}")
					return {'success': False, 'message': 'No result returned'}
			except Exception as e:
				print(f"Error executing A* routing: {e}")
				return {'success': False, 'message': str(e)}

	def get_nearest_node(self, point: Union[Point, Tuple[float, float]],
	                     graph_name: str = "base",
	                     max_distance: float = 0.1) -> Optional[int]:
		"""
		Find the nearest node in the graph to the given point.

		Parameters:
			point: A Point object or (longitude, latitude) tuple
			graph_name: Name of the graph to use
			max_distance: Maximum search distance in degrees

		Returns:
			Node ID if found, None otherwise
		"""
		nodes_table = f"graph_nodes_{graph_name}" if graph_name != "base" else "graph_nodes"

		# Convert tuple to point if needed
		if isinstance(point, tuple):
			point_wkt = f"POINT({point[0]} {point[1]})"
		else:
			point_wkt = point.wkt

		query = f"""
        SELECT id, node, ST_Distance(geom, ST_GeomFromText(:point_wkt, 4326)) as distance
        FROM "{self.graph_schema}"."{nodes_table}"
        WHERE ST_DWithin(geom, ST_GeomFromText(:point_wkt, 4326), :max_distance)
        ORDER BY distance
        LIMIT 1;
        """

		with self.pg.connect() as conn:
			result = conn.execute(text(query), {
				"point_wkt": point_wkt,
				"max_distance": max_distance
			}).fetchone()

			if result:
				node_id, node_str, distance = result
				print(f"Found nearest node {node_id} at distance {distance}")
				return node_id
			else:
				print(f"No node found within {max_distance} degrees of {point_wkt}")
				return None

	def save_route(self, route_data: Dict, route_name: str,
	               vessel_info: Dict = None) -> int:
		"""
		Saves a route to the database for future reference.

		Parameters:
			route_data: Dictionary with route information from find_route_astar
			route_name: Name for the route
			vessel_info: Optional dictionary with vessel parameters

		Returns:
			ID of the saved route
		"""
		# Create route schema if it doesn't exist
		with self.pg.connect() as conn:
			conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.route_schema}";'))

			# Create routes table if it doesn't exist
			conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS "{self.route_schema}"."routes" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                geom GEOMETRY(LineString, 4326),
                total_cost FLOAT,
                distance_nm FLOAT,
                edge_ids INTEGER[],
                node_ids INTEGER[],
                vessel_info JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """))

			# Insert the route
			if route_data.get('success', False):
				geojson = json.loads(route_data['geojson'])

				insert_query = f"""
                INSERT INTO "{self.route_schema}"."routes" 
                (name, geom, total_cost, distance_nm, edge_ids, node_ids, vessel_info)
                VALUES (
                    :name,
                    ST_GeomFromGeoJSON(:geometry),
                    :total_cost,
                    :distance_nm,
                    :edge_ids,
                    :node_ids,
                    :vessel_info
                )
                RETURNING id;
                """

				result = conn.execute(text(insert_query), {
					"name": route_name,
					"geometry": json.dumps(geojson['geometry']),
					"total_cost": route_data['total_cost'],
					"distance_nm": geojson['properties']['total_distance'],
					"edge_ids": route_data['edge_ids'],
					"node_ids": route_data['node_ids'],
					"vessel_info": json.dumps(vessel_info) if vessel_info else None
				}).fetchone()

				conn.commit()

				route_id = result[0]
				print(f"Route '{route_name}' saved with ID {route_id}")
				return route_id
			else:
				print("Cannot save route: No valid route data provided")
				return None

	def route_from_points(self, start_point: Union[Point, Tuple[float, float]],
	                      end_point: Union[Point, Tuple[float, float]],
	                      graph_name: str = "base",
	                      max_search_distance: float = 0.1,
	                      heuristic_factor: float = 1.0,
	                      use_adjusted_weights: bool = True) -> Dict:
		"""
		Convenience method to find a route between two geographic points.

		Parameters:
			start_point: Starting point as (longitude, latitude) or Point
			end_point: Ending point as (longitude, latitude) or Point
			graph_name: Name of the graph to use
			max_search_distance: Maximum distance to search for nearest nodes
			heuristic_factor: Factor for A* heuristic
			use_adjusted_weights: Whether to use adjusted weights

		Returns:
			Route data dictionary or error information
		"""
		# Find nearest nodes to start and end points
		start_node_id = self.get_nearest_node(start_point, graph_name, max_search_distance)
		end_node_id = self.get_nearest_node(end_point, graph_name, max_search_distance)

		if start_node_id is None or end_node_id is None:
			return {
				'success': False,
				'message': 'Could not find nodes near the specified points'
			}

		# Find route using A*
		return self.find_route_astar(
			start_id=start_node_id,
			end_id=end_node_id,
			graph_name=graph_name,
			heuristic_factor=heuristic_factor,
			use_adjusted_weights=use_adjusted_weights
		)

class GNNGraph(BaseGraph):
	"""
	Extends BaseGraph to provide Graph Neural Network capabilities
	for maritime route planning and pathfinding.
	"""

	def __init__(self, departure_port, arrival_port, enc_schema_name, graph_schema_name):
		super().__init__(
			departure_port=None,
			arrival_port=None,
			port_boundary=None,
			enc_schema_name=enc_schema_name,
			graph_schema_name=graph_schema_name,
		)
		self.model = None

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print(f"Device used: {self.device}")

	def convert_nx_to_pytorch_geometric(self, G):
		"""
		Convert NetworkX graph to PyTorch Geometric Data object
		"""
		# Create node mapping (nx node -> index)
		node_mapping = {node: i for i, node in enumerate(G.nodes())}

		# Extract node features
		node_features = []
		for node in G.nodes():
			# Extract relevant node features from the graph
			# For example: coordinates, depth, etc.
			features = [node[0], node[1]]  # lon, lat as basic features
			# Add more features if available in your graph
			node_features.append(features)

		# Convert to tensor
		x = torch.tensor(node_features, dtype=torch.float)

		# Extract edges and edge features
		edge_index = []
		edge_attr = []

		for u, v, data in G.edges(data=True):
			# Add edge in both directions (for undirected graph)
			edge_index.append([node_mapping[u], node_mapping[v]])
			edge_index.append([node_mapping[v], node_mapping[u]])

			# Extract edge features
			weight = data.get('weight', 1.0)
			base_weight = data.get('base_weight', weight)
			adjusted_weight = data.get('adjusted_weight', weight)

			# Use available weights as features
			edge_features = [weight, base_weight, adjusted_weight]
			edge_attr.append(edge_features)
			edge_attr.append(edge_features)  # Same features for reverse edge

		# Convert to tensors
		edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
		edge_attr = torch.tensor(edge_attr, dtype=torch.float)

		# Create PyTorch Geometric Data object
		data = tourchData(x=x, edge_index=edge_index, edge_attr=edge_attr)

		# Store node mapping for later reference
		data.node_mapping = node_mapping
		data.reverse_mapping = {v: k for k, v in node_mapping.items()}

		return data

	def create_gnn_model(self, input_dim, hidden_dim=64, output_dim=32):
		"""
		Create a Graph Neural Network model for pathfinding
		"""

		class GNNPathfinder(torch.nn.Module):
			def __init__(self, input_dim, hidden_dim, output_dim):
				super(GNNPathfinder, self).__init__()
				self.conv1 = GCNConv(input_dim, hidden_dim)
				self.conv2 = GCNConv(hidden_dim, hidden_dim)
				self.conv3 = GCNConv(hidden_dim, output_dim)

				# Edge weight prediction layer
				self.edge_predictor = torch.nn.Sequential(
					torch.nn.Linear(output_dim * 2 + 3, hidden_dim),  # 2 node embeddings + 3 edge features
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_dim, 1)
				)

			def forward(self, x, edge_index, edge_attr):
				# Node embedding
				x = self.conv1(x, edge_index)
				x = torch.relu(x)
				x = self.conv2(x, edge_index)
				x = torch.relu(x)
				x = self.conv3(x, edge_index)

				# For each edge, predict a new weight/cost
				edge_weights = []
				for i in range(edge_index.size(1)):
					# Get source and target node embeddings
					src, dst = edge_index[0, i], edge_index[1, i]
					src_emb = x[src]
					dst_emb = x[dst]

					# Concatenate with edge features
					edge_features = edge_attr[i]
					combined = torch.cat([src_emb, dst_emb, edge_features])

					# Predict new weight
					weight = self.edge_predictor(combined)
					edge_weights.append(weight)

				edge_weights = torch.cat(edge_weights)

				return x, edge_weights

		self.model = GNNPathfinder(input_dim, hidden_dim, output_dim).to(self.device)
		return self.model

	def train_gnn(self, data, optimal_paths, epochs=100, lr=0.001):
		"""
		Train the GNN model using historical optimal paths

		Parameters:
			data: PyTorch Geometric Data object
			optimal_paths: List of optimal paths (node sequences)
			epochs: Number of training epochs
			lr: Learning rate
		"""
		if self.model is None:
			input_dim = data.x.size(1)
			self.create_gnn_model(input_dim)

		optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

		# Move data to device
		data = data.to(self.device)

		# Create target edge weights (1 for edges in optimal paths, higher values for others)
		target_weights = torch.ones(data.edge_index.size(1), device=self.device) * 10.0  # Default high weight

		# Set weights for edges in optimal paths to 1.0
		for path in optimal_paths:
			for i in range(len(path) - 1):
				src_idx = data.node_mapping[path[i]]
				dst_idx = data.node_mapping[path[i + 1]]

				# Find this edge in edge_index
				for e_idx in range(data.edge_index.size(1)):
					if (data.edge_index[0, e_idx] == src_idx and
							data.edge_index[1, e_idx] == dst_idx):
						target_weights[e_idx] = 1.0
						break

		# Training loop
		self.model.train()
		for epoch in range(epochs):
			optimizer.zero_grad()

			# Forward pass
			_, pred_weights = self.model(data.x, data.edge_index, data.edge_attr)

			# Loss: make predicted weights close to target weights
			loss = torch.nn.functional.mse_loss(pred_weights.squeeze(), target_weights)

			# Backward pass
			loss.backward()
			optimizer.step()

			if (epoch + 1) % 10 == 0:
				print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

	def gnn_route(self, G, departure_point, arrival_point):
		"""
		Compute route using GNN-enhanced edge weights

		Parameters:
			G: NetworkX graph
			departure_point: Departure point (shapely.geometry.Point)
			arrival_point: Arrival point (shapely.geometry.Point)

		Returns:
			tuple: (route: LineString, total_distance: float)
		"""
		# Convert graph to PyTorch Geometric format
		data = self.convert_nx_to_pytorch_geometric(G)
		data = data.to(self.device)

		# If model not created yet, create it
		if self.model is None:
			input_dim = data.x.size(1)
			self.create_gnn_model(input_dim)
			print("Warning: Using untrained GNN model. Consider training first.")

		# Evaluate model to get edge weights
		self.model.eval()
		with torch.no_grad():
			_, pred_weights = self.model(data.x, data.edge_index, data.edge_attr)

		# Create a new graph with GNN-predicted weights
		G_gnn = G.copy()

		# Update edge weights based on GNN predictions
		for i, (u, v) in enumerate(G_gnn.edges()):
			# Find corresponding edge in PyTorch Geometric data
			src_idx = data.node_mapping[u]
			dst_idx = data.node_mapping[v]

			# Find this edge in edge_index
			for e_idx in range(data.edge_index.size(1)):
				if (data.edge_index[0, e_idx] == src_idx and
						data.edge_index[1, e_idx] == dst_idx):
					# Update weight
					G_gnn[u][v]['weight'] = float(pred_weights[e_idx].item())
					break

		# Use existing A* implementation with GNN-enhanced weights
		astar = Astar(G_gnn)
		route = astar.compute_route(departure_point, arrival_point)

		# Compute total distance
		total_distance_nm = 0.0
		coords = list(route.coords)
		for i in range(len(coords) - 1):
			lon1, lat1 = coords[i]
			lon2, lat2 = coords[i + 1]
			total_distance_nm += self.haversine(lon1, lat1, lon2, lat2)

		print(f"GNN-based route distance: {total_distance_nm} nautical miles")
		return route, total_distance_nm

	def collect_training_data(self, num_routes=10):
		"""
		Collect training data by generating multiple routes using traditional methods

		Returns:
			list: List of optimal paths (node sequences)
		"""
		# This is a placeholder - in a real implementation, you would:
		# 1. Generate multiple departure/arrival points
		# 2. Compute optimal routes using existing methods
		# 3. Return the node sequences for training

		optimal_paths = []

		# Example implementation:
		G = self.load_graph()  # Load your graph

		# Generate random points within the port boundary
		from shapely.geometry import Point
		import random

		# Get bounds of port boundary
		minx, miny, maxx, maxy = self.port_boundary.bounds

		for _ in range(num_routes):
			# Generate random points
			random_dep = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
			random_arr = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))

			# Ensure points are within boundary
			if not self.port_boundary.contains(random_dep) or not self.port_boundary.contains(random_arr):
				continue

			# Compute route using traditional A*
			astar = Astar(G)
			route = astar.compute_route(random_dep, random_arr)

			# Extract node sequence from route
			node_sequence = []
			for coord in route.coords:
				# Find closest node in graph
				closest_node = min(G.nodes(), key=lambda n: ((n[0] - coord[0]) ** 2 + (n[1] - coord[1]) ** 2))
				node_sequence.append(closest_node)

			optimal_paths.append(node_sequence)

		return optimal_paths
