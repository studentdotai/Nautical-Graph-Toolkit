#!/usr/bin/env python3
"""
pathfinding_lite.py

A lightweight, backend-agnostic module for A* pathfinding on a NetworkX graph.
This module is designed to be self-contained and focused on core routing logic.
"""

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Type, Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely
from shapely.geometry import Point, LineString, shape
from shapely.ops import unary_union

from ..utils.geometry_utils import Bearing

try:
    import rustworkx as rx
    _HAS_RUSTWORKX = True
except ImportError:
    rx = None
    _HAS_RUSTWORKX = False

_NO_PATH_EXC = (nx.NetworkXNoPath,)
if _HAS_RUSTWORKX:
    _NO_PATH_EXC = _NO_PATH_EXC + (rx.NoPathFound,)

logger = logging.getLogger(__name__)


def _build_edge_tree(graph: nx.Graph) -> Tuple[list, list, Any]:
    """Build and return (edges_list, edge_geoms, STRtree_or_None) for a graph."""
    edges_list = list(graph.edges(data=True))
    edge_geoms = []
    for u, v, data in edges_list:
        geom_dict = data.get('geom')
        if geom_dict is not None:
            try:
                edge_geoms.append(shape(geom_dict))
            except Exception:
                edge_geoms.append(LineString([u, v]))
        else:
            edge_geoms.append(LineString([u, v]))
    tree = shapely.STRtree(edge_geoms) if edge_geoms else None
    return edges_list, edge_geoms, tree


class Astar:
    """
    A self-contained implementation of the A* pathfinding algorithm.

    This class operates on a NetworkX graph where nodes are coordinate tuples
    and edges have a 'weight' attribute.
    """

    def __init__(self, graph: nx.Graph, min_cost_factor: float = 1.0):
        """
        Initializes the A* pathfinder.

        Args:
            graph (nx.Graph): The NetworkX graph to perform pathfinding on.
            min_cost_factor: Scale factor for heuristic admissibility. When bonus_factor
                can be < 1.0 (e.g., 0.3 for fairway edges), the haversine heuristic
                must be scaled down to remain admissible. Set to MIN_BONUS_FACTOR.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("Input must be a valid NetworkX graph.")
        self.graph = graph
        self.min_cost_factor = min_cost_factor

    def _heuristic(self, node1: Tuple[float, float], node2: Tuple[float, float]) -> float:
        """
        Calculates the haversine great-circle distance heuristic for A* path planning.
        Returns straight-line distance in nautical miles scaled by min_cost_factor.
        Admissible: heuristic ≤ actual edge cost for all valid graph edges.

        Args:
            node1 (Tuple[float, float]): The first node (lon, lat).
            node2 (Tuple[float, float]): The second node (lon, lat).

        Returns:
            float: The scaled great-circle distance in nautical miles.
        """
        lon1, lat1 = node1[0], node1[1]
        lon2, lat2 = node2[0], node2[1]
        R = 3440.065  # Earth radius in nautical miles
        dlon = math.radians(lon2 - lon1)
        dlat = math.radians(lat2 - lat1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return self.min_cost_factor * R * 2 * math.asin(math.sqrt(a))

    def find_nearest_node(self, point: Point) -> Optional[Tuple[float, float]]:
        """
        Finds the node in the graph that is closest to the given Shapely Point.

        This method iterates through all nodes to find the one with the minimum
        Euclidean distance to the input point.

        Args:
            point (Point): The geographic point to find the nearest node to.

        Returns:
            Optional[Tuple[float, float]]: The coordinate tuple of the nearest node,
                                           or None if the graph is empty.
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Cannot find nearest node in an empty graph.")
            return None

        # Using a generator expression with min() is more memory-efficient
        # than building a list of all nodes and distances.
        try:
            # The key for the min function is a lambda that calculates the distance
            # from the input point to each node in the graph.
            nearest_node = min(
                self.graph.nodes,
                key=lambda node: point.distance(Point(node))
            )
            return nearest_node
        except Exception as e:
            logger.error(f"An error occurred while finding the nearest node: {e}")
            return None

    def _get_edge_tree(self):
        """Lazily build and cache (edges_list, edge_geoms, STRtree) for all graph edges."""
        if not hasattr(self, '_cached_edge_tree'):
            self._cached_edge_tree = _build_edge_tree(self.graph)
        return self._cached_edge_tree

    def compute_route(self, start_point: Point, end_point: Point,
                      weight_key: str = 'adjusted_weight') -> Optional[LineString]:
        """
        Computes the shortest route between a start and end point using the A* algorithm.

        It first finds the closest graph nodes to the geographic start/end points,
        then computes the path between those nodes.

        Args:
            start_point (Point): The starting geographic point.
            end_point (Point): The destination geographic point.
            weight_key (str): The edge attribute to use for pathfinding cost.
                              Defaults to 'adjusted_weight' for vessel-specific routing.

        Returns:
            Optional[LineString]: A LineString representing the computed route,
                                  including the original start and end points.
                                  Returns None if no path can be found.
        """
        logger.info("Computing A* route...")

        start_node = self.find_nearest_node(start_point)
        end_node = self.find_nearest_node(end_point)

        if start_node is None or end_node is None:
            logger.error("Could not find a nearest node for the start or end point.")
            return None

        logger.info(f"Mapped start point to graph node: {start_node}")
        logger.info(f"Mapped end point to graph node: {end_node}")

        try:
            path = nx.astar_path(
                self.graph,
                start_node,
                end_node,
                heuristic=self._heuristic,
                weight=weight_key
            )
            # Prepend the original start point and append the original end point
            full_path_coords = [start_point.coords[0]] + path + [end_point.coords[0]]
            route_linestring = LineString(full_path_coords)
            logger.info(f"Successfully computed route with {len(path)} nodes.")
            return route_linestring
        except nx.NetworkXNoPath:
            logger.warning(f"No path found in the graph between {start_node} and {end_node}.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during A* pathfinding: {e}")
            return None


class AstarImproved(Astar):
    """
    Extends the standard A* algorithm with more advanced, domain-specific heuristics
    and cost functions for maritime navigation.

    This class introduces a "pilot quantity" heuristic that favors straighter paths
    and a customizable sailing cost function.
    """

    def __init__(self, graph: nx.Graph, min_cost_factor: float = 1.0):
        """
        Initializes the improved A* pathfinder.

        Args:
            graph (nx.Graph): The NetworkX graph to perform pathfinding on.
                              Nodes are expected to be coordinate tuples.
            min_cost_factor: Scale factor for heuristic admissibility (passed to parent).
        """
        super().__init__(graph, min_cost_factor=min_cost_factor)

    def _pilot_quantity(self, start_node: Tuple[float, float], target_node: Tuple[float, float],
                        current_node: Tuple[float, float]) -> float:
        """
        Calculates the "pilot quantity" heuristic, which penalizes deviations from
        the straight-line path between the start and target nodes.

        A lower value indicates a path closer to the direct line.

        Args:
            start_node (Tuple[float, float]): The ultimate starting node of the path.
            target_node (Tuple[float, float]): The ultimate target node of the path.
            current_node (Tuple[float, float]): The current node being evaluated.

        Returns:
            float: A heuristic value, typically between 3 and 4.
        """
        sx, sy = start_node
        tx, ty = target_node
        cx, cy = current_node

        # Vector from start to target
        st_vec = (tx - sx, ty - sy)
        # Vector from current to target
        ct_vec = (tx - cx, ty - cy)

        norm_st = math.sqrt(st_vec[0] ** 2 + st_vec[1] ** 2)
        norm_ct = math.sqrt(ct_vec[0] ** 2 + ct_vec[1] ** 2)

        # Avoid division by zero if a vector has zero length
        if norm_st == 0 or norm_ct == 0:
            return 4.0  # Maximum penalty

        # Calculate sine of the angle using the magnitude of the cross product
        cross_product = abs(st_vec[0] * ct_vec[1] - st_vec[1] * ct_vec[0])
        sin_theta = cross_product / (norm_st * norm_ct)

        # The pilot quantity is defined as 4 minus the sine of the angle.
        # A straight path (sin_theta=0) results in a value of 4.
        # A path perpendicular (sin_theta=1) results in a value of 3.
        return 4.0 - sin_theta

    def _improved_heuristic(self, current_node: Tuple[float, float], target_node: Tuple[float, float],
                            start_node: Tuple[float, float]) -> float:
        """
        An improved heuristic that combines Euclidean distance with the pilot quantity.
        This encourages the pathfinding algorithm to prefer nodes that lie closer
        to the direct line of sight to the target.
        """
        distance = self._heuristic(current_node, target_node)
        pq = self._pilot_quantity(start_node, target_node, current_node)
        # The heuristic is inverted (1/pq) to favor straighter paths (higher pq value).
        return distance * (1 / pq if pq > 0 else float('inf'))

    def compute_route_improved(self, start_point: Point, end_point: Point,
                               weight_key: str = 'adjusted_weight') -> Optional[LineString]:
        """
        Computes a route using the improved A* heuristic.

        Args:
            start_point (Point): The starting geographic point.
            end_point (Point): The destination geographic point.
            weight_key (str): The edge attribute to use for pathfinding cost.
                              Defaults to 'adjusted_weight' for vessel-specific routing.

        Returns:
            Optional[LineString]: A LineString representing the computed route,
                                  or None if no path is found.
        """
        logger.info("Computing route with improved A* heuristic...")

        start_node = self.find_nearest_node(start_point)
        end_node = self.find_nearest_node(end_point)

        if start_node is None or end_node is None:
            logger.error("Could not find a nearest node for the start or end point.")
            return None

        try:
            path = nx.astar_path(
                self.graph,
                source=start_node,
                target=end_node,
                heuristic=lambda u, v: self._improved_heuristic(u, v, start_node),
                weight=weight_key
            )
            full_path_coords = [start_point.coords[0]] + path + [end_point.coords[0]]
            route_linestring = LineString(full_path_coords)
            logger.info(f"Successfully computed improved route with {len(path)} nodes.")
            return route_linestring
        except nx.NetworkXNoPath:
            logger.warning(f"No path found in the graph between {start_node} and {end_node}.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during improved A* pathfinding: {e}")
            return None


class AstarMaritime(AstarImproved):
    """
    Two-Pass Corridor Routing for maritime A* pathfinding.

    Combines the speed of A* (Pass 1 scout) with the precision of Dijkstra
    (Pass 2 optimizer) constrained to a spatial corridor.  TSS lanes are
    forcefully included in the subgraph to prevent the "blind buffer" trap —
    the scenario where the optimal deep-water shipping lane lies outside the
    initial rough-path buffer and is therefore invisible to Pass 2.

    Pass 1 (Scout):
        Fast A* on the full graph to identify a rough course.
    Corridor:
        Shapely buffer (``corridor_buffer_nm`` NM) around the rough path,
        with edge-centric filtering via STRtree spatial index.
        All TSS edges within the extended route bounding box are added
        regardless of their distance from the Pass-1 line.
    Pass 2 (Optimizer):
        Dijkstra on the restricted subgraph — mathematically optimal within
        the corridor and guaranteed to evaluate TSS lanes.

    Falls back to the Pass-1 result if Dijkstra cannot find a path inside
    the corridor (e.g. after aggressive buffer trimming).

    After ``compute_route_maritime()`` returns, call ``get_maritime_metrics()``
    to retrieve detailed timing, corridor stats, and weight-factor counts.
    """

    M_PER_DEG: float = 111320.0  # metres per degree of latitude at equator
    MIN_COS: float = 0.5         # cos(60°) — conservative lower bound for lng scaling
    NM_TO_M: float = 1852.0      # 1 nautical mile in metres

    def __init__(
        self,
        graph: nx.Graph,
        min_cost_factor: float = 1.0,
        corridor_buffer_nm: float = 5.0,
        include_tss: bool = True,
        tss_bbox_extend_factor: float = 0.5,
    ):
        """
        Args:
            graph: The NetworkX graph to perform pathfinding on.
            min_cost_factor: Scale factor for heuristic admissibility (passed to parent).
            corridor_buffer_nm: Buffer radius around the Pass-1 path in nautical miles.
                                Converted to decimal degrees using latitude-corrected
                                formula matching ``geometry_utils.Buffer``.
                                Defaults to 5 NM.
            include_tss: If True, all TSS edges within the route bounding box
                         are added to the corridor subgraph, ensuring shipping
                         lanes are always reachable by Pass 2.  TSS edges are
                         those with ``ft_trafic`` set (RECTRC, FAIRWY, etc.) or
                         edges from TSS lane layers (e.g. TSSLPT) that have
                         ``ft_orient`` set and strong route preference
                         (``bonus_factor < 1.0``).
            tss_bbox_extend_factor: Fraction of the route diagonal used to expand the
                                    bounding box when searching for TSS nodes.
                                    0.5 → extend by 50 % of the route diagonal on each
                                    side.  Captures lanes that lie slightly off-course
                                    without including distant irrelevant lanes.
        """
        super().__init__(graph, min_cost_factor=min_cost_factor)
        self.corridor_buffer_nm = corridor_buffer_nm
        self.include_tss = include_tss
        self.tss_bbox_extend_factor = tss_bbox_extend_factor

        # Populated by compute_route_maritime(), accessed via get_maritime_metrics()
        self._maritime_metrics: Optional[dict] = None
        self._pass2_path_nodes: Optional[List[Tuple[float, float]]] = None

        # Debug cache: populated during compute_route_maritime()
        self._debug_pass1_path: Optional[List[Tuple[float, float]]] = None
        self._debug_corridor_polygon: Optional[Any] = None
        self._debug_corridor_edges: Optional[List[Tuple]] = None
        self._debug_corridor_nodes: Optional[List[Tuple[float, float]]] = None
        self._debug_tss_nodes: Optional[set] = None
        self._debug_pass2_path: Optional[List[Tuple[float, float]]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _buffer_degrees_at_lat(nm: float, center_lat: float) -> float:
        """
        Convert nautical miles to decimal degrees, corrected for latitude.

        Uses the same formula as ``geometry_utils.Buffer.apply_buffer_fast_gdf``:
        ``buffer_m / (M_PER_DEG * max(cos(radians(lat)), MIN_COS))``
        """
        buffer_m = nm * AstarMaritime.NM_TO_M
        cos_lat = max(math.cos(math.radians(center_lat)), AstarMaritime.MIN_COS)
        return buffer_m / (AstarMaritime.M_PER_DEG * cos_lat)

    def _build_corridor_polygon(self, path: List[Tuple[float, float]]) -> Any:
        """
        Create a buffered corridor polygon from a node path.

        The buffer radius is latitude-corrected at the midpoint latitude of the
        full path.  This avoids the ~2× undersize error in the east-west direction
        that a naive ``nm / 60`` conversion produces at high latitudes.
        """
        line = LineString(path)
        lats = [node[1] for node in path]
        center_lat = (min(lats) + max(lats)) / 2.0
        buffer_deg = self._buffer_degrees_at_lat(self.corridor_buffer_nm, center_lat)
        return line.buffer(buffer_deg)

    def _build_rx_graph(self, *, refresh: bool = True) -> Tuple[Any, Dict[Tuple[float, float], int]]:
        """Lazily build and cache a rustworkx PyGraph mirroring self.graph.

        Args:
            refresh: If True (default), rebuild unconditionally. If False,
                     perform a lightweight (node_count, edge_count) check
                     and only rebuild when counts diverge.
        """
        if not refresh and hasattr(self, '_cached_rx_graph'):
            rxg, node_to_idx = self._cached_rx_graph
            if (rxg.num_nodes() == self.graph.number_of_nodes()
                    and rxg.num_edges() == self.graph.number_of_edges()):
                return self._cached_rx_graph
            logger.debug("rustworkx cache stale (counts differ), rebuilding.")
        rxg = rx.PyGraph()
        node_to_idx: Dict[Tuple[float, float], int] = {}
        for node in self.graph.nodes():
            idx = rxg.add_node(node)
            node_to_idx[node] = idx
        for u, v, data in self.graph.edges(data=True):
            rxg.add_edge(node_to_idx[u], node_to_idx[v], data)
        self._cached_rx_graph = (rxg, node_to_idx)
        logger.info(
            f"Built rustworkx graph: {rxg.num_nodes()} nodes, "
            f"{rxg.num_edges()} edges."
        )
        return self._cached_rx_graph

    def _pass1_astar_rx(
        self,
        start_node: Tuple[float, float],
        end_node: Tuple[float, float],
        weight_key: str,
        *,
        refresh: bool = True,
    ) -> List[Tuple[float, float]]:
        """A* on the rustworkx graph. Returns path as list of coord tuples."""
        rxg, node_to_idx = self._build_rx_graph(refresh=refresh)
        idx_to_node = {v: k for k, v in node_to_idx.items()}
        start_idx = node_to_idx[start_node]
        target_coord = end_node
        path_indices = rx.astar_shortest_path(
            rxg,
            start_idx,
            goal_fn=lambda n: n == target_coord,
            edge_cost_fn=lambda e: float(e.get(weight_key, 1.0)),
            estimate_cost_fn=lambda n: self._heuristic(n, target_coord),
        )
        return [idx_to_node[i] for i in path_indices]

    def _get_tss_nodes_in_bbox(
        self,
        path: List[Tuple[float, float]],
    ) -> set:
        """
        Return all nodes belonging to TSS edges within an extended bounding box
        computed from ALL nodes of the given path (not just start/end).

        The bounding box is expanded by ``tss_bbox_extend_factor`` × the path
        diagonal on every side.  This captures TSS lanes that run parallel or
        slightly off the actual corridor without pulling in distant lanes.

        TSS edges are identified by the presence of the ``ft_trafic`` attribute
        or, for TSS lane layers that carry ORIENT but not TRAFIC in S-57
        (e.g. TSSLPT), by the combination of ``ft_orient`` being set and a
        strong route preference (``bonus_factor < 1.0``).
        """
        lons = [node[0] for node in path]
        lats = [node[1] for node in path]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        diagonal = math.sqrt((max_lon - min_lon) ** 2 + (max_lat - min_lat) ** 2)
        margin = diagonal * self.tss_bbox_extend_factor

        bbox = (
            min_lon - margin,
            min_lat - margin,
            max_lon + margin,
            max_lat + margin,
        )

        tss_nodes: set = set()
        for u, v, data in self.graph.edges(data=True):
            is_tss = data.get('ft_trafic') is not None
            if not is_tss and data.get('ft_orient') is not None:
                bf = data.get('bonus_factor', 1.0)
                if isinstance(bf, (int, float)) and bf < 1.0:
                    is_tss = True
            if is_tss:
                for node in (u, v):
                    lon, lat = node
                    if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]:
                        tss_nodes.add(node)

        logger.debug(f"Found {len(tss_nodes)} TSS nodes in route bounding box.")
        return tss_nodes

    def _extract_subgraph(
        self,
        corridor_polygon: Any,
        pass1_path: List[Tuple[float, float]],
        start_node: Tuple[float, float],
        end_node: Tuple[float, float],
    ) -> Tuple[nx.Graph, dict]:
        """
        Build the restricted subgraph for Pass 2 using edge-centric filtering.

        Uses ``shapely.STRtree`` on edge geometries for O(N log N) vectorized
        intersection testing (same pattern as ``weights.py`` STRtree enrichment).
        Edges whose geometry intersects the corridor polygon are included — this
        correctly captures edges that cross the corridor even when neither
        endpoint falls inside.

        Returns:
            Tuple of (subgraph, corridor_stats).
        """
        edges_list, edge_geoms, tree = self._get_edge_tree()

        # --- STRtree spatial index query ---
        if tree is None:
            corridor_edge_indices = []
        else:
            corridor_edge_indices = tree.query(corridor_polygon, predicate='intersects')

        # --- Collect nodes from intersecting edges ---
        corridor_nodes: set = set()
        for idx in corridor_edge_indices:
            u, v, _ = edges_list[idx]
            corridor_nodes.add(u)
            corridor_nodes.add(v)

        # --- Cache corridor edges for debug export ---
        self._debug_corridor_edges = [
            (edges_list[idx][0], edges_list[idx][1], edges_list[idx][2], edge_geoms[idx])
            for idx in corridor_edge_indices
        ]
        self._debug_corridor_nodes = list(corridor_nodes)

        # --- TSS enrichment (uses full pass1_path for bbox) ---
        tss_added = 0
        if self.include_tss:
            tss_nodes = self._get_tss_nodes_in_bbox(pass1_path)
            tss_added = len(tss_nodes - corridor_nodes)
            corridor_nodes.update(tss_nodes)
            self._debug_tss_nodes = tss_nodes

        # --- Guarantee start/end connectivity ---
        corridor_nodes.add(start_node)
        corridor_nodes.add(end_node)

        subgraph = self.graph.subgraph(corridor_nodes).copy()

        corridor_stats = {
            'full_graph_nodes': self.graph.number_of_nodes(),
            'full_graph_edges': self.graph.number_of_edges(),
            'subgraph_nodes': subgraph.number_of_nodes(),
            'subgraph_edges': subgraph.number_of_edges(),
            'tss_nodes_added': tss_added,
        }

        logger.info(
            f"Corridor subgraph: {corridor_stats['subgraph_nodes']} nodes, "
            f"{corridor_stats['subgraph_edges']} edges "
            f"(full graph: {corridor_stats['full_graph_nodes']} nodes, "
            f"{corridor_stats['full_graph_edges']} edges; "
            f"TSS enrichment: +{tss_added} nodes)."
        )
        return subgraph, corridor_stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_maritime_metrics(self) -> Optional[dict]:
        """Return metrics from the most recent ``compute_route_maritime()`` call."""
        return self._maritime_metrics

    def compute_route_maritime(
        self,
        start_point: Point,
        end_point: Point,
        weight_key: str = 'adjusted_weight',
        *,
        scout_path: Optional[List[Tuple[float, float]]] = None,
        pass1_backend: str = 'rustworkx',
        pass1_refresh: bool = True,
    ) -> Optional[LineString]:
        """
        Compute a maritime route using the Two-Pass Corridor method.

        Pass 1 (Scout): A* on the full graph to define the rough corridor.
        Pass 2 (Optimizer): Dijkstra on the restricted corridor subgraph for
        the mathematically optimal route within that search space.

        Falls back to the Pass-1 A* result if Dijkstra cannot find a path
        inside the corridor subgraph (logged as a warning).

        After this method returns, call ``get_maritime_metrics()`` to retrieve
        detailed timing, corridor stats, and weight-factor counts.

        Args:
            start_point: Starting geographic point.
            end_point: Destination geographic point.
            weight_key: Edge attribute used as routing cost.
                        Defaults to ``'adjusted_weight'``.
            scout_path: Optional pre-computed path for corridor construction.
                        When provided, Pass 1 (A* scout) is skipped and this
                        path is used directly to build the corridor. Pass 2
                        (Dijkstra) then optimises end-to-end within that
                        corridor. Defaults to None (standard A* scout).
            pass1_backend: Backend for Pass 1 A* search. ``'rustworkx'`` (default)
                           for fast Rust-based search, or ``'networkx'`` as
                           fallback.  Ignored when ``scout_path`` is provided.
            pass1_refresh: If True (default), rebuild the rustworkx graph
                           unconditionally before Pass 1. If False, reuse the
                           cached graph when node/edge counts match. Ignored
                           when ``pass1_backend`` is ``'networkx'`` or when
                           ``scout_path`` is provided.

        Returns:
            LineString of the computed route, or None if Pass 1 finds no path.
        """
        logger.info(
            f"Computing maritime two-pass route "
            f"(corridor buffer: {self.corridor_buffer_nm} NM, "
            f"TSS enrichment: {self.include_tss})..."
        )
        self._maritime_metrics = None
        self._pass2_path_nodes = None
        self._debug_pass1_path = None
        self._debug_corridor_polygon = None
        self._debug_corridor_edges = None
        self._debug_corridor_nodes = None
        self._debug_tss_nodes = None
        self._debug_pass2_path = None

        start_node = self.find_nearest_node(start_point)
        end_node = self.find_nearest_node(end_point)

        if start_node is None or end_node is None:
            logger.error("Could not find a nearest node for start or end point.")
            return None

        # ── Pass 1: A* scout on the full graph ───────────────────────
        if scout_path is not None:
            logger.info(f"Pass 1: Using injected scout path ({len(scout_path)} nodes).")
            pass1_path = scout_path
            t_pass1 = 0.0
        else:
            logger.info(f"Pass 1: Running A* scout ({pass1_backend})...")
            t0 = time.perf_counter()
            try:
                if pass1_backend == 'rustworkx' and _HAS_RUSTWORKX:
                    pass1_path = self._pass1_astar_rx(
                        start_node, end_node, weight_key,
                        refresh=pass1_refresh,
                    )
                else:
                    pass1_path = nx.astar_path(
                        self.graph,
                        start_node,
                        end_node,
                        heuristic=self._heuristic,
                        weight=weight_key,
                    )
            except _NO_PATH_EXC:
                logger.warning("Pass 1: No path found in graph.")
                return None
            t_pass1 = time.perf_counter() - t0
            logger.info(f"Pass 1 complete: {len(pass1_path)} nodes in {t_pass1:.3f}s.")
        self._debug_pass1_path = pass1_path

        # ── Corridor construction ─────────────────────────────────────
        t0 = time.perf_counter()
        corridor_polygon = self._build_corridor_polygon(pass1_path)
        self._debug_corridor_polygon = corridor_polygon
        subgraph, corridor_stats = self._extract_subgraph(
            corridor_polygon, pass1_path, start_node, end_node,
        )
        t_corridor = time.perf_counter() - t0
        logger.info(f"Corridor built in {t_corridor:.3f}s.")

        # ── Pass 2: Dijkstra on the corridor subgraph ─────────────────
        logger.info("Pass 2: Running Dijkstra on corridor subgraph...")
        t0 = time.perf_counter()
        pass_used = "pass2"
        try:
            pass2_path = nx.dijkstra_path(
                subgraph,
                start_node,
                end_node,
                weight=weight_key,
            )
            final_path = pass2_path
        except nx.NetworkXNoPath:
            logger.warning(
                "Pass 2: No path in corridor subgraph — "
                "falling back to Pass-1 A* result."
            )
            final_path = pass1_path
            pass_used = "pass1_fallback"
        t_pass2 = time.perf_counter() - t0
        logger.info(f"Pass 2 complete: {len(final_path)} nodes in {t_pass2:.3f}s.")
        self._debug_pass2_path = final_path

        # ── Build output LineString ───────────────────────────────────
        full_path_coords = [start_point.coords[0]] + final_path + [end_point.coords[0]]
        route_ls = LineString(full_path_coords)

        # ── Compute pass distances for comparison ─────────────────────
        pass1_ls = LineString([start_point.coords[0]] + pass1_path + [end_point.coords[0]])
        pass1_dist = Route._calculate_route_distance(pass1_ls)
        pass2_dist = Route._calculate_route_distance(route_ls)

        # ── Collect per-edge factor counts along the final path ───────
        accumulated_weight = 0.0
        blocking_count = 0
        penalty_count = 0
        bonus_count = 0
        for i in range(len(final_path) - 1):
            u, v = final_path[i], final_path[i + 1]
            if self.graph.has_edge(u, v):
                ed = self.graph[u][v]
                accumulated_weight += ed.get(weight_key, ed.get('weight', 0.0))
                if ed.get('blocking_factor', 0) >= 1000:
                    blocking_count += 1
                if ed.get('penalty_factor', 1.0) > 1.0:
                    penalty_count += 1
                if ed.get('bonus_factor', 1.0) < 1.0:
                    bonus_count += 1

        # ── Store metrics on self ─────────────────────────────────────
        self._maritime_metrics = {
            'pass1_distance_nm': round(pass1_dist, 4),
            'pass2_distance_nm': round(pass2_dist, 4),
            'total_distance_nm': round(pass2_dist, 4),
            'computation_time_s': {
                'pass1': round(t_pass1, 4),
                'corridor_build': round(t_corridor, 4),
                'pass2': round(t_pass2, 4),
                'total': round(t_pass1 + t_corridor + t_pass2, 4),
            },
            'accumulated_weight': round(accumulated_weight, 4),
            'blocking_count': blocking_count,
            'penalty_count': penalty_count,
            'bonus_count': bonus_count,
            'corridor_stats': corridor_stats,
            'pass_used': pass_used,
        }
        self._pass2_path_nodes = final_path

        logger.info(
            f"Maritime route: {pass2_dist:.2f} NM "
            f"(pass1: {pass1_dist:.2f} NM, delta: {pass1_dist - pass2_dist:+.2f} NM), "
            f"pass_used={pass_used}"
        )
        return route_ls


class AstarMaritimeSmooth(AstarMaritime):
    """
    Three-Pass Maritime Routing with String-Pulling post-processing.

    Passes 1-2 (A* scout + Dijkstra optimizer) inherited from AstarMaritime.
    Pass 3 applies a greedy line-of-sight string-pulling algorithm to shortcut
    the Dijkstra path, replacing zig-zag segments with straight lines that
    avoid obstacle edges.

    Obstacle edges within the smoothing buffer are edges where
    blocking_factor > 1 (hard-blocked or partially blocked).

    Only Dijkstra path nodes are used as waypoints — no external corners
    or candidate nodes are introduced.
    """

    def __init__(
        self,
        graph: nx.Graph,
        min_cost_factor: float = 1.0,
        corridor_buffer_nm: float = 5.0,
        include_tss: bool = True,
        tss_bbox_extend_factor: float = 0.5,
        sp_buffer_nm: Optional[float] = None,
        use_land_grid: bool = False,
        channel_layers: Optional[List[str]] = None,
        data_factory: Optional[Any] = None,
        graph_path: Optional[Union[str, Path]] = None,
        graph_name: Optional[str] = None,
        enc_names: Optional[List[str]] = None,
    ):
        super().__init__(
            graph,
            min_cost_factor=min_cost_factor,
            corridor_buffer_nm=corridor_buffer_nm,
            include_tss=include_tss,
            tss_bbox_extend_factor=tss_bbox_extend_factor,
        )
        self.sp_buffer_nm = sp_buffer_nm if sp_buffer_nm is not None else 2.0
        self._data_factory = data_factory
        self._graph_path = Path(graph_path) if graph_path else None
        self._graph_name = graph_name
        self._enc_names = enc_names
        self._shortcut_metadata: Optional[List[dict]] = None

        # Resolved channel layers: use provided list, or TSS defaults when include_tss=True
        if channel_layers is not None:
            self._channel_layer_names = channel_layers
        elif include_tss:
            self._channel_layer_names = ['fairwy', 'rectrc', 'tsslpt', 'dwrtpt', 'rcrtcl']
        else:
            self._channel_layer_names = []

        # Lazy-loaded geometries (populated during Pass 3)
        self._land_geom: Optional[Any] = None
        self._channel_geoms: Optional[List[Any]] = None
        self._mask_loaded: bool = False
        self._use_land_grid = use_land_grid

        # Debug cache: populated during Pass 3
        self._debug_obstacle_geoms: Optional[List[Any]] = None
        self._debug_obstacle_nodes: Optional[set] = None
        self._debug_obstacle_edges_data: Optional[List[Tuple]] = None
        self._debug_buffer_polygon: Optional[Any] = None
        self._debug_smoothed_path: Optional[List[Tuple[float, float]]] = None
        self._debug_navigability_mask: Optional[Any] = None

    # ------------------------------------------------------------------
    # Obstacle space construction
    # ------------------------------------------------------------------

    def _build_obstacle_space(
        self,
        dijkstra_path: List[Tuple[float, float]],
    ) -> Tuple[Optional[Any], List[Any], Any]:
        """
        Build obstacle geometries from non-preferred edges within the smoothing buffer.

        1. Buffer around Dijkstra path (sp_buffer_nm NM).
        2. Query all graph edges intersecting the buffer.
        3. Classify each edge:
           - Obstacle: blocking_factor > 1 (hard-blocked or partially blocked).
           - Preferred: everything else (penalized, neutral, or bonus edges).
        4. Build STRtree from obstacle geometries for fast intersection tests.

        Returns:
            (obstacle_tree, obstacle_geoms, buffer_polygon)
        """
        lats = [n[1] for n in dijkstra_path]
        center_lat = (min(lats) + max(lats)) / 2.0
        buf_deg = AstarMaritime._buffer_degrees_at_lat(self.sp_buffer_nm, center_lat)
        line = LineString(dijkstra_path)
        buf_poly = line.buffer(buf_deg)

        edges_list, edge_geoms, tree = self._get_edge_tree()
        if tree is None:
            intersecting = []
        else:
            intersecting = tree.query(buf_poly, predicate='intersects')

        obstacle_geoms: List[Any] = []
        obstacle_nodes: set = set()
        obstacle_edges_raw: List[Tuple] = []

        for idx in intersecting:
            u, v, data = edges_list[idx]
            blk = data.get('blocking_factor', 0)

            if blk <= 1:
                continue  # not blocked — preferred / penalized / neutral

            obstacle_geoms.append(edge_geoms[idx])
            obstacle_edges_raw.append((u, v, data))
            obstacle_nodes.add(u)
            obstacle_nodes.add(v)

        # Cache for debug export
        self._debug_obstacle_geoms = obstacle_geoms
        self._debug_obstacle_nodes = obstacle_nodes
        self._debug_obstacle_edges_data = obstacle_edges_raw

        obstacle_tree = shapely.STRtree(obstacle_geoms) if obstacle_geoms else None
        logger.info(
            f"Obstacle space: {len(obstacle_geoms)} obstacle edges, "
            f"{len(obstacle_nodes)} obstacle nodes "
            f"(buffer {self.sp_buffer_nm:.1f} NM)."
        )
        return obstacle_tree, obstacle_geoms, buf_poly

    def _load_mask_geometries(self) -> None:
        """
        Lazy-load land grid and channel geometries from data sources.

        Called once during Pass 3 before navigability mask construction.

        Land grid: loaded from graph GPKG 'land_area' layer (GeoPackage) or
        from the grid schema in PostGIS.  Proceeds without subtraction if not found.

        Channel layers: loaded from ENC data factory using self._channel_layer_names
        (defaults to TSS route layers when include_tss=True).
        """
        if self._mask_loaded:
            return
        self._mask_loaded = True
        self._channel_geoms = []

        # --- Load land grid ---
        if self._use_land_grid:
            land_geom = self._load_land_grid()
            if land_geom is not None:
                self._land_geom = land_geom
            else:
                logger.warning(
                    "use_land_grid=True but no land grid found — "
                    "proceeding without land subtraction. "
                    "Ensure apply_static_weights_* was run (produces land_grid), "
                    "or pass graph_path / data_factory with a land_area layer."
                )

        # --- Load channel layers ---
        if self._channel_layer_names and self._data_factory is not None:
            for layer_name in self._channel_layer_names:
                try:
                    gdf = self._data_factory.get_layer(
                        layer_name, filter_by_enc=self._enc_names)
                    if not gdf.empty:
                        self._channel_geoms.append(gdf.geometry.union_all())
                        logger.debug(f"Loaded channel layer '{layer_name}'")
                except Exception as e:
                    logger.debug(f"Channel layer '{layer_name}' not available: {e}")
            if self._channel_geoms:
                logger.info(
                    f"Loaded {len(self._channel_geoms)} channel layer(s) "
                    f"for navigability mask."
                )

    def _load_land_grid(self) -> Optional[Any]:
        """
        Load land grid geometry from the available data source.

        GPKG: tries 'land_area' then 'land_grid' layers in the graph file.
        PostGIS: queries {graph_name}_land_grid from the grid schema, falls back
        to LNDARE from the ENC schema.
        Returns None if no land geometry is found.
        """
        # Try GeoPackage: 'land_area' / 'land_grid' layer in graph file
        if self._graph_path is not None and self._graph_path.exists():
            try:
                import fiona
                layers = fiona.listlayers(str(self._graph_path))
                for land_layer_name in ('land_area', 'land_grid'):
                    if land_layer_name in layers:
                        land_gdf = gpd.read_file(
                            str(self._graph_path),
                            layer=land_layer_name, engine='fiona')
                        if not land_gdf.empty:
                            geom = land_gdf.geometry.union_all()
                            logger.info(
                                f"Loaded land grid from GPKG layer "
                                f"'{land_layer_name}' ({geom.geom_type})."
                            )
                            return geom
            except FileNotFoundError:
                logger.debug(f"GPKG file not found: {self._graph_path}")
            except Exception as e:
                logger.warning(f"GPKG land grid lookup failed: {e}")

        # Try PostGIS: {graph_name}_land_grid in grid schema
        if self._data_factory is not None and self._graph_name is not None:
            try:
                manager = self._data_factory.manager
                if hasattr(manager, 'engine'):
                    land_table = f"{self._graph_name}_land_grid"
                    grid_schema = 'grid'
                    try:
                        land_gdf = gpd.read_postgis(
                            f'SELECT geometry FROM "{grid_schema}"."{land_table}"',
                            manager.engine, geom_col='geometry')
                        if not land_gdf.empty:
                            geom = land_gdf.geometry.union_all()
                            logger.info(
                                f"Loaded land grid from PostGIS: "
                                f"{grid_schema}.{land_table} ({geom.geom_type})."
                            )
                            return geom
                    except Exception as e:
                        logger.debug(f"PostGIS land_grid table not found: {e}")

                    # Fallback: LNDARE from ENC schema
                    if self._enc_names:
                        land_gdf = self._data_factory.get_layer(
                            'lndare', filter_by_enc=self._enc_names)
                        if not land_gdf.empty:
                            geom = land_gdf.geometry.union_all()
                            logger.info(
                                f"Loaded land geometry from LNDARE layer "
                                f"({geom.geom_type})."
                            )
                            return geom
            except Exception as e:
                logger.debug(f"PostGIS land grid lookup failed: {e}")

        return None

    def _build_navigability_mask(self, buf_poly: Any) -> Any:
        """
        Construct a navigability mask for string-pulling containment.

        Starts from the SP buffer (spatial limit for shortcuts), expands with
        preferred navigation channel geometries, then subtracts land areas.

        Returns buf_poly unchanged if no land_geom or channel_geoms are available
        (fully backward-compatible).
        """
        self._load_mask_geometries()

        if self._land_geom is None and not self._channel_geoms:
            return buf_poly

        mask = buf_poly

        for geom in (self._channel_geoms or []):
            if geom is not None and not geom.is_empty:
                mask = mask.union(geom)

        if self._land_geom is not None and not self._land_geom.is_empty:
            mask = mask.difference(self._land_geom)

        # union/difference can produce GeometryCollection with non-polygonal
        # artifacts (LineString, Point) — extract polygonal parts only.
        if mask.geom_type not in ('Polygon', 'MultiPolygon'):
            parts = [g for g in mask.geoms
                     if g.geom_type in ('Polygon', 'MultiPolygon')]
            if parts:
                mask = unary_union(parts)

        # Keep only polygonal components connected to the original buffer.
        # Union with channels can add disconnected fragments far from the
        # corridor — they are not navigable in this context.
        if mask.geom_type == 'MultiPolygon':
            connected = [p for p in mask.geoms if buf_poly.intersects(p)]
            if connected:
                mask = unary_union(connected)

        logger.info(
            f"Navigability mask: land_subtraction={self._land_geom is not None}, "
            f"channels={len(self._channel_geoms or [])}, "
            f"result_type={mask.geom_type}."
        )
        return mask

    @staticmethod
    def _line_intersects_obstacles(p1: tuple, p2: tuple, obstacle_tree: Optional[Any]) -> bool:
        """Check if a straight line p1 -> p2 intersects any obstacle geometry."""
        if obstacle_tree is None:
            return False
        line = LineString([p1, p2])
        indices = obstacle_tree.query(line, predicate='intersects')
        return len(indices) > 0

    # ------------------------------------------------------------------
    # Core String-Pulling algorithm
    # ------------------------------------------------------------------

    def _string_pull(
        self,
        dijkstra_path: List[Tuple[float, float]],
        obstacle_tree: Optional[Any],
        buf_poly: Optional[Any] = None,
    ) -> List[Tuple[float, float]]:
        """
        Greedy line-of-sight string-pulling on the Dijkstra path.

        Starting from the first node, greedily finds the furthest reachable
        node along the path with a clear line of sight (no obstacle
        intersection).  That node becomes the next anchor, and the process
        repeats until the destination is reached.

        Two constraints are enforced for each shortcut candidate:
          1. No intersection with obstacle geometries.
          2. The shortcut segment must remain inside the smoothing buffer
             corridor (``buf_poly``), preventing shortcuts that swing outside
             the buffered region around the Dijkstra path.

        Only Dijkstra path nodes are considered — no external corners or
        candidate nodes are introduced.
        """
        if len(dijkstra_path) <= 2:
            return list(dijkstra_path)

        result = [dijkstra_path[0]]
        i = 0

        while i < len(dijkstra_path) - 1:
            furthest = i + 1
            for j in range(len(dijkstra_path) - 1, i + 1, -1):
                seg = LineString([dijkstra_path[i], dijkstra_path[j]])
                if not self._line_intersects_obstacles(
                    dijkstra_path[i], dijkstra_path[j], obstacle_tree
                ):
                    if buf_poly is not None and not buf_poly.covers(seg):
                        continue
                    furthest = j
                    break
            result.append(dijkstra_path[furthest])
            i = furthest

        if result[-1] != dijkstra_path[-1]:
            result.append(dijkstra_path[-1])

        logger.info(
            f"String-Pulling: {len(dijkstra_path)} input nodes -> "
            f"{len(result)} output nodes."
        )
        return result

    # ------------------------------------------------------------------
    # Shortcut metadata aggregation
    # ------------------------------------------------------------------

    def _aggregate_shortcut_metadata(
        self,
        smoothed_path: List[Tuple[float, float]],
    ) -> List[dict]:
        """
        For each segment in the smoothed path, find intersecting graph
        edges and aggregate their metadata.
        """
        edges_list, _, tree = self._get_edge_tree()
        if tree is None:
            return []

        metadata = []
        for k in range(len(smoothed_path) - 1):
            start = smoothed_path[k]
            end = smoothed_path[k + 1]
            seg_geom = LineString([start, end])

            indices = tree.query(seg_geom, predicate='intersects')

            edge_ids = []
            weights = []
            for idx in indices:
                data = edges_list[idx][2]
                eid = data.get('id')
                if eid is not None:
                    edge_ids.append(eid)
                w = data.get('adjusted_weight', data.get('weight', 0.0))
                if w is not None:
                    weights.append(w)

            metadata.append({
                'segment_index': k,
                'start': start,
                'end': end,
                'distance_nm': round(Route._calculate_route_distance(LineString([start, end])), 4),
                'intersecting_edge_ids': edge_ids,
                'adjusted_weight_min': round(min(weights), 4) if weights else None,
                'adjusted_weight_max': round(max(weights), 4) if weights else None,
                'adjusted_weight_avg': round(sum(weights) / len(weights), 4) if weights else None,
                'is_shortcut': not self.graph.has_edge(start, end),
            })

        return metadata

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_shortcut_metadata(self) -> Optional[List[dict]]:
        """Return shortcut metadata from the most recent routing call."""
        return self._shortcut_metadata

    def compute_route_maritime_smooth(
        self,
        start_point: Point,
        end_point: Point,
        weight_key: str = 'adjusted_weight',
        *,
        scout_path: Optional[List[Tuple[float, float]]] = None,
        pass1_backend: str = 'rustworkx',
        pass1_refresh: bool = True,
    ) -> Optional[LineString]:
        """
        Three-pass maritime route: A* + Dijkstra + String-Pulling.

        Passes 1 and 2 are delegated to ``AstarMaritime``.
        Pass 3 applies String-Pulling on the Dijkstra result.

        Args:
            start_point: Starting geographic point.
            end_point: Destination geographic point.
            weight_key: Edge attribute used as routing cost.
            scout_path: Optional pre-computed path for corridor construction.
                        Forwarded to ``compute_route_maritime()``.
            pass1_backend: Backend for Pass 1 A* search. ``'rustworkx'`` (default)
                           or ``'networkx'``. Forwarded to ``compute_route_maritime()``.
            pass1_refresh: Forwarded to ``compute_route_maritime()``.
        """
        logger.info("Computing three-pass maritime route with String-Pulling...")
        self._shortcut_metadata = None
        self._debug_obstacle_geoms = None
        self._debug_obstacle_nodes = None
        self._debug_obstacle_edges_data = None
        self._debug_buffer_polygon = None
        self._debug_smoothed_path = None
        self._debug_navigability_mask = None

        # Passes 1 + 2
        route_ls = super().compute_route_maritime(
            start_point, end_point, weight_key=weight_key,
            scout_path=scout_path, pass1_backend=pass1_backend,
            pass1_refresh=pass1_refresh,
        )
        if route_ls is None:
            return None

        dijkstra_path = self._pass2_path_nodes
        if dijkstra_path is None or len(dijkstra_path) <= 2:
            logger.info("Dijkstra path too short for String-Pulling, returning as-is.")
            return route_ls

        # Pass 3: Build obstacle space and string-pull
        try:
            t0 = time.perf_counter()
            obstacle_tree, obstacle_geoms, buf_poly = self._build_obstacle_space(dijkstra_path)
            self._debug_buffer_polygon = buf_poly

            navigability_mask = self._build_navigability_mask(buf_poly)
            self._debug_navigability_mask = navigability_mask

            smoothed_path = self._string_pull(dijkstra_path, obstacle_tree, buf_poly=navigability_mask)
            self._debug_smoothed_path = smoothed_path
            t_sp = time.perf_counter() - t0

            # Aggregate shortcut metadata
            self._shortcut_metadata = self._aggregate_shortcut_metadata(smoothed_path)

            # Override stored path so Route.detailed_route() uses smoothed path
            self._pass2_path_nodes = smoothed_path

            # Build output LineString
            full_coords = [start_point.coords[0]] + smoothed_path + [end_point.coords[0]]
            route_ls = LineString(full_coords)

            # Update metrics
            if self._maritime_metrics is not None:
                self._maritime_metrics['computation_time_s']['string_pull'] = round(t_sp, 4)
                self._maritime_metrics['computation_time_s']['total'] = round(
                    self._maritime_metrics['computation_time_s']['total'] + t_sp, 4
                )
                self._maritime_metrics['sp_original_nodes'] = len(dijkstra_path)
                self._maritime_metrics['sp_smoothed_nodes'] = len(smoothed_path)
                self._maritime_metrics['sp_reduction_pct'] = round(
                    (1 - len(smoothed_path) / max(len(dijkstra_path), 1)) * 100, 2
                )
                self._maritime_metrics['pass_used'] = 'pass3_string_pull'

                sp_dist = Route._calculate_route_distance(route_ls)
                self._maritime_metrics['total_distance_nm'] = round(sp_dist, 4)
                self._maritime_metrics['sp_distance_nm'] = round(sp_dist, 4)
                logger.info(
                    f"String-Pulling complete: {len(dijkstra_path)} -> {len(smoothed_path)} nodes "
                    f"({self._maritime_metrics['sp_reduction_pct']:.1f}% reduction), "
                    f"{sp_dist:.2f} NM in {t_sp:.3f}s."
                )
        except Exception as e:
            logger.warning(
                f"Pass 3 (String-Pulling) failed: {e}. "
                f"Falling back to Pass 1+2 result."
            )

        return route_ls

    # ------------------------------------------------------------------
    # Debug export
    # ------------------------------------------------------------------

    def export_debug_gpkg(self, output_path: Union[str, Path]) -> bool:
        """
        Export all intermediate objects from the 3-pass workflow to a GeoPackage.

        Must be called AFTER ``compute_route_maritime_smooth()`` has completed.
        Each intermediate object is written as its own layer for QGIS inspection.

        Layers produced:
          pass1_scout_path    – A* scout route (Pass 1)
          pass2_dijkstra_path – Dijkstra route (Pass 2, before smoothing)
          pass3_smoothed_path – String-Pulled route (Pass 3)
          corridor_polygon    – Buffered corridor polygon
          corridor_edges      – All edges inside the corridor
          corridor_nodes      – All nodes inside the corridor
          tss_nodes           – TSS enrichment nodes
          obstacle_edges      – Non-preferred / blocking / penalized edges
          obstacle_nodes      – Nodes adjacent to obstacle edges
          smoothing_buffer    – String-pulling buffer polygon
          shortcut_segments   – Shortcutted segments with weight metadata
          metrics             – Timing and corridor statistics

        Args:
            output_path: Path to the output ``.gpkg`` file. Parent directories
                are created automatically. Overwrites existing files.

        Returns:
            True if at least one layer was written successfully.
        """
        output_path = Path(output_path)

        if self._maritime_metrics is None:
            logger.warning("No routing data to export. Run compute_route_maritime_smooth() first.")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _edge_geom(u, v, data):
            geom_dict = data.get('geom')
            if geom_dict is not None:
                try:
                    return shape(geom_dict)
                except Exception:
                    return LineString([u, v])
            return LineString([u, v])

        layers: List[Tuple[str, gpd.GeoDataFrame]] = []

        # --- Pass paths ---
        for label, cache_attr, pass_num in [
            ('pass1_scout_path', '_debug_pass1_path', 1),
            ('pass2_dijkstra_path', '_debug_pass2_path', 2),
            ('pass3_smoothed_path', '_debug_smoothed_path', 3),
        ]:
            path = getattr(self, cache_attr, None)
            if path is not None and len(path) >= 2:
                geom = LineString(path)
                dist = Route._calculate_route_distance(geom)
                gdf = gpd.GeoDataFrame(
                    [{'pass': pass_num, 'node_count': len(path),
                      'distance_nm': round(dist, 4)}],
                    geometry=[geom], crs='EPSG:4326',
                )
                layers.append((label, gdf))

        # --- Corridor polygon ---
        if self._debug_corridor_polygon is not None:
            poly = self._debug_corridor_polygon
            props = {'buffer_nm': self.corridor_buffer_nm, 'area_sq_deg': round(poly.area, 8)}
            mm = self._maritime_metrics
            if mm and 'corridor_stats' in mm:
                props.update(mm['corridor_stats'])
            layers.append(('corridor_polygon', gpd.GeoDataFrame(
                [props], geometry=[poly], crs='EPSG:4326')))

        # --- Corridor edges ---
        if self._debug_corridor_edges:
            records = []
            geoms = []
            for u, v, data, geom in self._debug_corridor_edges:
                records.append({
                    'u_lon': u[0], 'u_lat': u[1], 'v_lon': v[0], 'v_lat': v[1],
                    'adjusted_weight': data.get('adjusted_weight'),
                    'blocking_factor': data.get('blocking_factor'),
                    'bonus_factor': data.get('bonus_factor'),
                })
                geoms.append(geom)
            layers.append(('corridor_edges', gpd.GeoDataFrame(
                records, geometry=geoms, crs='EPSG:4326')))

        # --- Corridor nodes ---
        if self._debug_corridor_nodes:
            layers.append(('corridor_nodes', gpd.GeoDataFrame(
                [{'lon': n[0], 'lat': n[1]} for n in self._debug_corridor_nodes],
                geometry=[Point(n) for n in self._debug_corridor_nodes],
                crs='EPSG:4326')))

        # --- TSS nodes ---
        if self._debug_tss_nodes:
            nodes = list(self._debug_tss_nodes)
            layers.append(('tss_nodes', gpd.GeoDataFrame(
                [{'lon': n[0], 'lat': n[1], 'is_tss': True} for n in nodes],
                geometry=[Point(n) for n in nodes], crs='EPSG:4326')))

        # --- Obstacle edges ---
        if self._debug_obstacle_edges_data:
            records = []
            geoms = []
            for u, v, data in self._debug_obstacle_edges_data:
                records.append({
                    'u_lon': u[0], 'u_lat': u[1], 'v_lon': v[0], 'v_lat': v[1],
                    'adjusted_weight': data.get('adjusted_weight', 1.0),
                    'blocking_factor': data.get('blocking_factor', 0),
                    'penalty_factor': data.get('penalty_factor', 1.0),
                })
                geoms.append(_edge_geom(u, v, data))
            layers.append(('obstacle_edges', gpd.GeoDataFrame(
                records, geometry=geoms, crs='EPSG:4326')))

        # --- Obstacle nodes ---
        if self._debug_obstacle_nodes:
            nodes = list(self._debug_obstacle_nodes)
            layers.append(('obstacle_nodes', gpd.GeoDataFrame(
                [{'lon': n[0], 'lat': n[1]} for n in nodes],
                geometry=[Point(n) for n in nodes], crs='EPSG:4326')))

        # --- Smoothing buffer ---
        if self._debug_buffer_polygon is not None:
            bpoly = self._debug_buffer_polygon
            layers.append(('smoothing_buffer', gpd.GeoDataFrame(
                [{'buffer_nm': self.sp_buffer_nm, 'area_sq_deg': round(bpoly.area, 8)}],
                geometry=[bpoly], crs='EPSG:4326')))

        # --- Navigability mask ---
        if self._debug_navigability_mask is not None:
            mpoly = self._debug_navigability_mask
            layers.append(('navigability_mask', gpd.GeoDataFrame(
                [{'has_land_subtraction': self._land_geom is not None,
                  'num_channel_geoms': len(self._channel_geoms or []),
                  'area_sq_deg': round(mpoly.area, 8)}],
                geometry=[mpoly], crs='EPSG:4326')))

        # --- Shortcut segments ---
        if self._shortcut_metadata:
            records = []
            geoms = []
            for m in self._shortcut_metadata:
                start, end = m['start'], m['end']
                records.append({
                    'segment_index': m['segment_index'],
                    'distance_nm': m['distance_nm'],
                    'is_shortcut': m['is_shortcut'],
                    'adjusted_weight_min': m.get('adjusted_weight_min'),
                    'adjusted_weight_max': m.get('adjusted_weight_max'),
                    'adjusted_weight_avg': m.get('adjusted_weight_avg'),
                })
                geoms.append(LineString([start, end]))
            layers.append(('shortcut_segments', gpd.GeoDataFrame(
                records, geometry=geoms, crs='EPSG:4326')))

        # --- Metrics ---
        if self._maritime_metrics is not None:
            flat = {}
            mm = self._maritime_metrics
            for k, v in mm.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        flat[f"{k}_{sk}" if k != 'corridor_stats' else f"corridor_{sk}"] = sv
                else:
                    flat[k] = v
            layers.append(('metrics', gpd.GeoDataFrame(
                [flat], geometry=[Point(0, 0)], crs='EPSG:4326')))

        # --- Write all layers ---
        if not layers:
            logger.warning("No debug layers to export.")
            return False

        first = True
        written = 0
        for layer_name, gdf in layers:
            if len(gdf) == 0:
                continue
            try:
                if first:
                    gdf.to_file(str(output_path), layer=layer_name, driver='GPKG',
                                mode='w', engine='pyogrio')
                    first = False
                else:
                    gdf.to_file(str(output_path), layer=layer_name, driver='GPKG',
                                mode='a', engine='pyogrio')
                logger.info(f"Debug GPKG: layer '{layer_name}' — {len(gdf)} features.")
                written += 1
            except Exception as e:
                logger.warning(
                    f"Debug GPKG: failed to write layer '{layer_name}': {e}"
                )

        logger.info(f"Debug GPKG exported: {output_path} ({written}/{len(layers)} layers).")
        return written > 0


class Route:
    """
    A backend-agnostic class for computing routes on a pre-constructed NetworkX graph.
    This class is designed to work with any data source (PostGIS, GPKG, etc.)
    by operating on a standard graph object.
    """

    def __init__(self, graph: nx.Graph, data_manager: Any,
                 node_id_map: Optional[Dict[int, Tuple[float, float]]] = None):
        """
        Initializes the Route computer.

        Args:
            graph (nx.Graph): The NetworkX graph to perform routing on.
            data_manager (Any): An instance of a data manager (e.g., PostGISManager,
                                GPKGManager) used for saving and loading routes.
            node_id_map: Optional mapping of integer node IDs to (lon, lat) tuples.
                         Build it with ``GraphUtils.node_id_map()``.  Required only
                         when waypoints are specified as integer IDs in ``forced_route()``.
        """
        if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
            raise ValueError("A valid, non-empty NetworkX graph is required.")
        self.graph = graph
        if not hasattr(data_manager, 'save_route') or not hasattr(data_manager, 'load_route'):
            raise TypeError("The provided data_manager must have 'save_route' and 'load_route' methods.")
        self.manager = data_manager
        self._last_pathfinder: Optional[Any] = None
        self._node_id_map: Dict[int, Tuple[float, float]] = node_id_map or {}

    @staticmethod
    def _calculate_route_distance(route: LineString) -> float:
        """
        Calculates the total haversine distance of a route in nautical miles.

        Args:
            route (LineString): The route geometry.

        Returns:
            float: The total distance in nautical miles.
        """
        total_distance_nm = 0.0
        coords = list(route.coords)
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]

            # Haversine calculation
            R = 3440.065  # Earth radius in nautical miles
            dlon = math.radians(lon2 - lon1)
            dlat = math.radians(lat2 - lat1)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            total_distance_nm += R * c

        return total_distance_nm

    @staticmethod
    def _haversine_nm(p1: tuple, p2: tuple) -> float:
        """Haversine distance between two (lon, lat) tuples in nautical miles."""
        R = 3440.065
        dlon = math.radians(p2[0] - p1[0])
        dlat = math.radians(p2[1] - p1[1])
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(p1[1])) * math.cos(math.radians(p2[1]))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def _bearing_deg(p1: tuple, p2: tuple) -> float:
        """Initial bearing from p1 to p2 in degrees [0, 360)."""
        dlon = math.radians(p2[0] - p1[0])
        y = math.sin(dlon) * math.cos(math.radians(p2[1]))
        x = (math.cos(math.radians(p1[1])) * math.sin(math.radians(p2[1]))
             - math.sin(math.radians(p1[1])) * math.cos(math.radians(p2[1])) * math.cos(dlon))
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def _has_line_of_sight(self, p1: tuple, p2: tuple) -> bool:
        """
        Checks if a straight line between two points intersects blocking geometries.
        Lazily builds and caches an STRtree of blocking edges (blocking_factor > 1)
        for efficiency.
        """
        # Lazily build the STRtree of blocking geometries
        if not hasattr(self, '_blocking_geoms_tree'):
            blocking_geoms = []
            for u, v, data in self.graph.edges(data=True):
                if data.get('blocking_factor', 0) > 1:
                    geom_dict = data.get('geom')
                    if geom_dict:
                        try:
                            blocking_geoms.append(shape(geom_dict))
                        except Exception:
                            blocking_geoms.append(LineString([u, v]))
                    else:
                        blocking_geoms.append(LineString([u, v]))

            if not blocking_geoms:
                self._blocking_geoms_tree = None
            else:
                self._blocking_geoms_tree = shapely.STRtree(blocking_geoms)
                logger.info(f"Built STRtree with {len(blocking_geoms)} blocking geometries for LoS checks.")

        if self._blocking_geoms_tree is None:
            return True

        line = LineString([p1, p2])
        intersecting_indices = self._blocking_geoms_tree.query(line, predicate='intersects')

        if len(intersecting_indices) > 0:
            logger.debug(f"LoS check failed for segment {p1} -> {p2}")
            return False

        return True

    # ------------------------------------------------------------------
    # Fillet smoothing helpers
    # ------------------------------------------------------------------

    _MIN_FILLET_DEFLECTION = 0.087  # ~5 deg minimum bend for fillet construction

    @staticmethod
    def _resolve_fillet_radius(ft_buffer_zone_dist: float) -> float:
        """Map buffer zone distance to turning radius in nautical miles."""
        if ft_buffer_zone_dist == 3.0:
            return 1.0   # pilot zone / port approach
        elif ft_buffer_zone_dist in (4.0, 12.0):
            return 2.0   # coastal approach
        else:
            return 4.0   # open water (0.0 or missing)

    def _build_nearest_edge_tree(self):
        """Lazily build an STRtree over all graph edges for nearest-edge lookup."""
        if hasattr(self, '_all_edge_tree'):
            return

        # Reuse pathfinder's cached edge tree when available (avoids ~50s
        # duplicate build on large graphs).  All Astar subclasses share the
        # same _get_edge_tree() / _cached_edge_tree mechanism.
        if (self._last_pathfinder is not None
                and hasattr(self._last_pathfinder, '_cached_edge_tree')):
            edges_list, edge_geoms, tree = self._last_pathfinder._cached_edge_tree
            self._all_edge_geoms = edge_geoms
            self._all_edge_data = [data for _, _, data in edges_list]
            self._all_edge_tree = tree
            logger.info(f"Reused pathfinder STRtree with {len(edge_geoms)} edges.")
            return

        if not hasattr(self, '_cached_edge_tree'):
            self._cached_edge_tree = _build_edge_tree(self.graph)
        edges_list, edge_geoms, tree = self._cached_edge_tree
        self._all_edge_geoms = edge_geoms
        self._all_edge_data = [data for _, _, data in edges_list]
        self._all_edge_tree = tree
        logger.info(f"Built nearest-edge STRtree with {len(edge_geoms)} edges.")

    def _find_nearest_edge_data(self, vertex: tuple) -> Optional[dict]:
        """Find the data dict of the nearest graph edge to a vertex."""
        self._build_nearest_edge_tree()
        if self._all_edge_tree is None:
            return {}
        idx = self._all_edge_tree.nearest(Point(vertex))
        return self._all_edge_data[idx]

    @staticmethod
    def _compute_fillet(p1: tuple, p2: tuple, p3: tuple, radius_deg: float):
        """
        Compute a circular arc fillet at vertex p2 between segments p1-p2 and p2-p3.

        Args:
            p1, p2, p3: Coordinate tuples (lon, lat).
            radius_deg: Fillet radius in decimal degrees (already lat-corrected).

        Returns:
            Tuple of (tangent_a, arc_points, tangent_b, actual_radius_deg),
            or None if the angle is too shallow or geometry is degenerate.
        """
        # Vectors from p2 toward p1 and p3
        v_in = (p1[0] - p2[0], p1[1] - p2[1])
        v_out = (p3[0] - p2[0], p3[1] - p2[1])

        len_in = math.sqrt(v_in[0] ** 2 + v_in[1] ** 2)
        len_out = math.sqrt(v_out[0] ** 2 + v_out[1] ** 2)

        if len_in < 1e-12 or len_out < 1e-12:
            return None

        n_in = (v_in[0] / len_in, v_in[1] / len_in)
        n_out = (v_out[0] / len_out, v_out[1] / len_out)

        # Dot product gives cos of the deflection angle
        dot = n_in[0] * n_out[0] + n_in[1] * n_out[1]
        dot = max(-1.0, min(1.0, dot))

        # Half of the turning angle at p2
        half_angle = math.acos(dot) / 2.0

        # Deflection = how much the path bends at this vertex.
        # deflection = pi - 2*half_angle.  Skip if too gentle (< ~5 deg).
        # Also skip degenerate U-turns (half_angle → 0, tan → 0, div-by-zero).
        deflection = math.pi - 2 * half_angle
        if deflection < Route._MIN_FILLET_DEFLECTION or half_angle < 0.01:
            return None

        # Tangent distance along each segment
        t = radius_deg / math.tan(half_angle)

        # Clamp to 45% of the shorter adjacent segment
        max_t = min(len_in, len_out) * 0.45
        if t > max_t:
            t = max_t
            radius_deg = t * math.tan(half_angle)

        if radius_deg < 1e-12:
            return None

        # Tangent points on the two segments
        t_a = (p2[0] + n_in[0] * t, p2[1] + n_in[1] * t)
        t_b = (p2[0] + n_out[0] * t, p2[1] + n_out[1] * t)

        # Arc center: bisector direction from p2
        bisect = (n_in[0] + n_out[0], n_in[1] + n_out[1])
        bisect_len = math.sqrt(bisect[0] ** 2 + bisect[1] ** 2)
        if bisect_len < 1e-12:
            return None
        bisect_norm = (bisect[0] / bisect_len, bisect[1] / bisect_len)

        # Distance from p2 to arc center along bisector
        d_center = radius_deg / math.sin(half_angle)
        center = (p2[0] + bisect_norm[0] * d_center,
                  p2[1] + bisect_norm[1] * d_center)

        # Angles from center to tangent points
        angle_a = math.atan2(t_a[1] - center[1], t_a[0] - center[0])
        angle_b = math.atan2(t_b[1] - center[1], t_b[0] - center[0])

        # Determine arc sweep direction (shortest path)
        delta = angle_b - angle_a
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi

        # Discretize arc
        abs_delta_deg = abs(math.degrees(delta))
        n_points = max(4, int(abs_delta_deg / 5))
        n_points = min(n_points, 36)

        arc_points = []
        for j in range(1, n_points):
            frac = j / n_points
            a = angle_a + delta * frac
            px = center[0] + radius_deg * math.cos(a)
            py = center[1] + radius_deg * math.sin(a)
            arc_points.append((px, py))

        return t_a, arc_points, t_b, radius_deg

    def _check_arc_safety(self, arc_points: list, tangent_a: tuple, tangent_b: tuple) -> bool:
        """Verify a fillet arc doesn't cross blocking geometries."""
        # Check consecutive arc segments
        all_points = [tangent_a] + arc_points + [tangent_b]
        for i in range(len(all_points) - 1):
            if not self._has_line_of_sight(all_points[i], all_points[i + 1]):
                return False

        # Also check direct chord and midpoint cross-check for large arcs
        mid_idx = len(all_points) // 2
        if not self._has_line_of_sight(tangent_a, tangent_b):
            return False
        if not self._has_line_of_sight(tangent_a, all_points[mid_idx]):
            return False

        return True

    def apply_fillet_smoothing(
        self,
        route_geom: LineString,
        merge_threshold_deg: float = 1.0,
        arc_threshold_deg: float = 3.0,
    ) -> Tuple[LineString, List[dict], List[dict]]:
        """
        Apply bearing-merge simplification followed by circular arc fillets
        with zone-based turning radii.

        Pipeline:
          1. Break the input LineString into segments with bearing and distance.
          2. Merge consecutive segments whose bearing differs by at most
             ``merge_threshold_deg`` (single forward pass, deterministic).
          3. At each merged-segment junction where the bearing change exceeds
             ``arc_threshold_deg``, construct a circular fillet arc whose radius
             is determined by the nearest edge's buffer zone distance:
               - 3.0 (inside 3NM) -> 1 NM radius (pilot zone)
               - 4.0 or 12.0 (3-12NM) -> 2 NM radius (coastal approach)
               - 0.0 or missing (>12NM) -> 4 NM radius (open water)
             Each arc is validated against blocking edges in the graph.

        Args:
            route_geom: Input route as a LineString.
            merge_threshold_deg: Max bearing difference (degrees) to merge two
                consecutive segments into one (default 1.0).
            arc_threshold_deg: Min bearing difference (degrees) at a junction to
                trigger fillet arc construction (default 3.0).

        Returns:
            Tuple of (smoothed LineString, fillet_metadata list, segments list).
            Segments are dicts with 'type' ('leg' or 'arc') and 'coords'.
            Legs also have 'distance_nm' and 'bearing_deg'.
            Arcs also have 'radius_nm', 'turn_angle_deg', and 'direction'.
        """
        coords = list(route_geom.coords)
        if len(coords) < 3:
            return route_geom, [], []

        # ── Phase A: Break into segments with bearing and distance ──
        _MIN_SEG_NM = 0.0001  # skip degenerate near-zero-length segments
        segments_raw = []
        for i in range(len(coords) - 1):
            start, end = coords[i], coords[i + 1]
            dist = self._haversine_nm(start, end)
            if dist < _MIN_SEG_NM:
                continue
            segments_raw.append({
                'start': start,
                'end': end,
                'bearing_deg': self._bearing_deg(start, end),
                'distance_nm': dist,
            })

        if not segments_raw:
            return route_geom, [], []

        # ── Phase B: Merge consecutive segments by bearing similarity ──
        merged = [dict(segments_raw[0])]
        for seg in segments_raw[1:]:
            diff = Bearing.angular_difference_scalar(merged[-1]['bearing_deg'],
                                                     seg['bearing_deg'])
            if diff <= merge_threshold_deg:
                merged[-1]['end'] = seg['end']
                merged[-1]['bearing_deg'] = self._bearing_deg(
                    merged[-1]['start'], merged[-1]['end'])
                merged[-1]['distance_nm'] = self._haversine_nm(
                    merged[-1]['start'], merged[-1]['end'])
            else:
                merged.append(dict(seg))

        logger.info(
            f"Bearing merge: {len(coords)} coords -> {len(segments_raw)} raw -> "
            f"{len(merged)} merged segments (threshold={merge_threshold_deg}°)."
        )

        if len(merged) < 2:
            # Essentially straight route — emit one leg, no fillets
            seg = merged[0]
            segments_out = [{
                'type': 'leg',
                'coords': [seg['start'], seg['end']],
                'distance_nm': round(seg['distance_nm'], 4),
                'bearing_deg': round(seg['bearing_deg'], 2),
            }]
            return LineString([seg['start'], seg['end']]), [], segments_out

        # ── Phase C: Apply fillet arcs at junctions ──

        # Pre-compute junction info: bearing diff, turn direction, radius
        junctions = []
        for k in range(len(merged) - 1):
            bearing_diff = Bearing.angular_difference_scalar(
                merged[k]['bearing_deg'], merged[k + 1]['bearing_deg'])
            p2 = merged[k]['end']  # junction point
            edge_data = self._find_nearest_edge_data(p2)
            buf_dist = float(edge_data.get('ft_buffer_zone_dist') or 0.0)
            radius_nm = self._resolve_fillet_radius(buf_dist)
            radius_deg = AstarMaritime._buffer_degrees_at_lat(radius_nm, p2[1])

            # Turn direction via cross product of in/out vectors
            v_in = (merged[k]['start'][0] - p2[0], merged[k]['start'][1] - p2[1])
            v_out = (merged[k + 1]['end'][0] - p2[0], merged[k + 1]['end'][1] - p2[1])
            cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
            direction = 'starboard' if cross > 0 else 'port'

            junctions.append({
                'k': k,
                'p1': merged[k]['start'],
                'p2': p2,
                'p3': merged[k + 1]['end'],
                'bearing_diff': bearing_diff,
                'radius_nm': radius_nm,
                'radius_deg': radius_deg,
                'buf_zone_dist': buf_dist,
                'nearest_edge_data': edge_data,
                'direction': direction,
                'half_angle': None,  # filled during fillet computation
                't_in': 0.0,
                't_out': 0.0,
            })

        # Simplified overlap check between consecutive fillets
        for j in range(len(junctions)):
            jn = junctions[j]
            p1, p2, p3 = jn['p1'], jn['p2'], jn['p3']
            v_in = (p1[0] - p2[0], p1[1] - p2[1])
            v_out = (p3[0] - p2[0], p3[1] - p2[1])
            len_in = math.sqrt(v_in[0] ** 2 + v_in[1] ** 2)
            len_out = math.sqrt(v_out[0] ** 2 + v_out[1] ** 2)
            if len_in < 1e-12 or len_out < 1e-12:
                jn['_skip'] = True
                continue

            n_in = (v_in[0] / len_in, v_in[1] / len_in)
            n_out = (v_out[0] / len_out, v_out[1] / len_out)
            dot = max(-1.0, min(1.0, n_in[0] * n_out[0] + n_in[1] * n_out[1]))
            half_angle = math.acos(dot) / 2.0
            jn['half_angle'] = half_angle

            deflection = math.pi - 2 * half_angle
            if deflection < Route._MIN_FILLET_DEFLECTION or half_angle < 0.01:
                jn['_skip'] = True
                continue

            t = jn['radius_deg'] / math.tan(half_angle)
            jn['t_in'] = t
            jn['t_out'] = t

            # Overlap: if previous junction has an outgoing tangent that
            # collides with this junction's incoming tangent
            if j > 0 and not junctions[j - 1].get('_skip'):
                prev_jn = junctions[j - 1]
                # Intervening segment length in degrees (approximate)
                seg_len = math.sqrt(
                    (jn['p2'][0] - prev_jn['p2'][0]) ** 2
                    + (jn['p2'][1] - prev_jn['p2'][1]) ** 2)
                if jn['t_in'] + prev_jn['t_out'] > seg_len * 0.9:
                    shrink = (seg_len * 0.9) / (jn['t_in'] + prev_jn['t_out'])
                    jn['radius_deg'] *= shrink
                    prev_jn['radius_deg'] *= shrink
                    jn['t_in'] *= shrink
                    jn['t_out'] *= shrink
                    prev_jn['t_in'] *= shrink
                    prev_jn['t_out'] *= shrink

        # Build output geometry and metadata
        fillet_metadata = []
        segments = []
        output_coords = [merged[0]['start']]
        prev_leg_start = merged[0]['start']
        pending_tangent_b = None

        for j in range(len(junctions)):
            jn = junctions[j]
            k = jn['k']

            # Flush previous fillet's tangent_b
            if pending_tangent_b is not None:
                output_coords.append(pending_tangent_b)
                pending_tangent_b = None

            # Close the straight leg from prev_leg_start to this junction
            leg_end = jn['p2']
            segments.append({
                'type': 'leg',
                'coords': [prev_leg_start, leg_end],
                'distance_nm': round(self._haversine_nm(prev_leg_start, leg_end), 4),
                'bearing_deg': round(self._bearing_deg(prev_leg_start, leg_end), 2),
            })

            if jn.get('_skip') or jn['bearing_diff'] <= arc_threshold_deg:
                output_coords.append(jn['p2'])
                if jn['bearing_diff'] <= arc_threshold_deg:
                    reason = f"below arc threshold ({jn['bearing_diff']:.1f}°)"
                else:
                    reason = 'degenerate angle'
                fillet_metadata.append({
                    'vertex_coords': jn['p2'],
                    'fillet_applied': False,
                    'reason': reason,
                    'turning_radius_nm': jn['radius_nm'],
                    'buf_zone_dist': jn['buf_zone_dist'],
                    'nearest_edge_data': jn['nearest_edge_data'],
                })
                prev_leg_start = jn['p2']
                continue

            # Attempt fillet construction
            result = self._compute_fillet(
                jn['p1'], jn['p2'], jn['p3'], jn['radius_deg'])

            if result is None:
                output_coords.append(jn['p2'])
                fillet_metadata.append({
                    'vertex_coords': jn['p2'],
                    'fillet_applied': False,
                    'reason': 'geometry degenerate',
                    'turning_radius_nm': jn['radius_nm'],
                    'buf_zone_dist': jn['buf_zone_dist'],
                    'nearest_edge_data': jn['nearest_edge_data'],
                })
                prev_leg_start = jn['p2']
                continue

            t_a, arc_pts, t_b, actual_r = result

            if not self._check_arc_safety(arc_pts, t_a, t_b):
                logger.debug(f"Fillet at {jn['p2']} rejected by blocking check.")
                output_coords.append(jn['p2'])
                fillet_metadata.append({
                    'vertex_coords': jn['p2'],
                    'fillet_applied': False,
                    'reason': 'blocking intersection',
                    'turning_radius_nm': jn['radius_nm'],
                    'buf_zone_dist': jn['buf_zone_dist'],
                    'nearest_edge_data': jn['nearest_edge_data'],
                })
                prev_leg_start = jn['p2']
                continue

            # Fillet accepted — replace last leg with leg ending at t_a
            segments[-1] = {
                'type': 'leg',
                'coords': [prev_leg_start, t_a],
                'distance_nm': round(self._haversine_nm(prev_leg_start, t_a), 4),
                'bearing_deg': round(self._bearing_deg(prev_leg_start, t_a), 2),
            }

            output_coords.append(t_a)
            output_coords.extend(arc_pts)
            pending_tangent_b = t_b

            actual_nm = actual_r / max(
                AstarMaritime._buffer_degrees_at_lat(1.0, jn['p2'][1]), 1e-12)
            turn_angle = math.degrees(
                math.pi - 2 * (jn.get('half_angle') or 0))

            segments.append({
                'type': 'arc',
                'coords': [t_a] + arc_pts + [t_b],
                'radius_nm': round(actual_nm, 4),
                'turn_angle_deg': round(turn_angle, 2),
                'direction': jn['direction'],
            })

            prev_leg_start = t_b

            fillet_metadata.append({
                'vertex_coords': jn['p2'],
                'fillet_applied': True,
                'turning_radius_nm': round(actual_nm, 4),
                'turn_angle_degrees': round(turn_angle, 2),
                'buf_zone_dist': jn['buf_zone_dist'],
                'nearest_edge_data': jn['nearest_edge_data'],
            })

        # Flush last deferred tangent_b
        if pending_tangent_b is not None:
            output_coords.append(pending_tangent_b)

        output_coords.append(merged[-1]['end'])

        # Final leg from prev_leg_start to route end
        segments.append({
            'type': 'leg',
            'coords': [prev_leg_start, merged[-1]['end']],
            'distance_nm': round(
                self._haversine_nm(prev_leg_start, merged[-1]['end']), 4),
            'bearing_deg': round(
                self._bearing_deg(prev_leg_start, merged[-1]['end']), 2),
        })

        applied = sum(1 for f in fillet_metadata if f.get('fillet_applied'))
        rejected = len(fillet_metadata) - applied
        n_legs = sum(1 for s in segments if s['type'] == 'leg')
        n_arcs = sum(1 for s in segments if s['type'] == 'arc')
        logger.info(
            f"Fillet smoothing: {applied}/{len(fillet_metadata)} fillets applied, "
            f"{rejected} rejected; {len(coords)} raw -> {len(merged)} merged -> "
            f"{len(output_coords)} output points ({n_legs} legs, {n_arcs} arcs)."
        )

        return LineString(output_coords), fillet_metadata, segments

    def _resolve_waypoint(self, wp: Union[int, Point, Tuple[float, float]],
                          helper: 'Astar') -> Tuple[float, float]:
        """
        Convert a waypoint in any supported format to a graph node tuple.

        Args:
            wp: Integer node ID, Shapely Point, or (lon, lat) coordinate tuple.
            helper: An Astar instance whose ``find_nearest_node`` is used for Points.

        Returns:
            (lon, lat) tuple that exists in ``self.graph``.

        Raises:
            KeyError: If an integer ID is not in ``node_id_map``.
            ValueError: If a coordinate tuple is not a graph node.
        """
        if isinstance(wp, int):
            node = self._node_id_map.get(wp)
            if node is None:
                raise KeyError(
                    f"Node ID {wp} not found in node_id_map. "
                    "Pass node_id_map=GraphUtils.node_id_map(...) to Route.__init__."
                )
            return node
        elif isinstance(wp, Point):
            return helper.find_nearest_node(wp)
        else:  # (lon, lat) tuple
            if wp not in self.graph:
                raise ValueError(f"Node {wp} not found in graph.")
            return wp

    def base_route(self, departure_point: Point, arrival_point: Point,
                   astar_impl: Type['Astar'] = Astar,
                   weight_key: str = 'adjusted_weight',
                   min_cost_factor: float = 1.0,
                   **pathfinder_kwargs) -> Optional[Tuple[LineString, float]]:
        """
        Computes a route using the specified A* implementation and calculates its distance.

        Args:
            departure_point (Point): The starting geographic point.
            arrival_point (Point): The destination geographic point.
            astar_impl (Type[Astar]): The A* class to use for pathfinding
                (e.g., Astar, AstarImproved, or AstarMaritime).
            weight_key (str): The edge attribute to use for pathfinding cost.
                Defaults to 'adjusted_weight'.
            min_cost_factor: Scale factor for A* heuristic admissibility (default: 1.0).
            **pathfinder_kwargs: Extra keyword arguments forwarded to the pathfinder
                constructor (e.g., ``corridor_buffer_nm``, ``include_tss`` for
                ``AstarMaritime``).

        Returns:
            Optional[Tuple[LineString, float]]: A tuple containing the route LineString
                                                and its total distance in nautical miles,
                                                or None if no route is found.
        """
        logger.info(f"Computing base route with {astar_impl.__name__}...")

        # Instantiate the chosen A* pathfinder
        # Strip keys that belong to Route/compute methods, not the pathfinder constructor
        _ctor_kwargs = {k: v for k, v in pathfinder_kwargs.items()
                        if k not in ('apply_smoothing', 'merge_threshold_deg', 'arc_threshold_deg',
                                     'scout_path', 'pass1_backend', 'pass1_refresh')}
        _scout_path = pathfinder_kwargs.get('scout_path')
        _pass1_backend = pathfinder_kwargs.get('pass1_backend', 'rustworkx')
        _pass1_refresh = pathfinder_kwargs.get('pass1_refresh', True)
        pathfinder = astar_impl(self.graph, min_cost_factor=min_cost_factor,
                                **_ctor_kwargs)
        self._last_pathfinder = pathfinder

        # Compute the route using the appropriate method.
        # AstarMaritimeSmooth must be checked before AstarMaritime (subclass).
        if isinstance(pathfinder, AstarMaritimeSmooth):
            route_geom = pathfinder.compute_route_maritime_smooth(
                departure_point, arrival_point, weight_key=weight_key,
                scout_path=_scout_path, pass1_backend=_pass1_backend,
                pass1_refresh=_pass1_refresh)
        elif isinstance(pathfinder, AstarMaritime):
            route_geom = pathfinder.compute_route_maritime(
                departure_point, arrival_point, weight_key=weight_key,
                scout_path=_scout_path, pass1_backend=_pass1_backend,
                pass1_refresh=_pass1_refresh)
        elif isinstance(pathfinder, AstarImproved):
            route_geom = pathfinder.compute_route_improved(departure_point, arrival_point, weight_key=weight_key)
        else:
            route_geom = pathfinder.compute_route(departure_point, arrival_point, weight_key=weight_key)

        if route_geom is None:
            logger.warning("Route computation failed. No path found.")
            return None

        # Calculate the total distance of the computed route
        total_distance = self._calculate_route_distance(route_geom)

        logger.info(f"Route computed successfully. Total distance: {total_distance:.2f} nautical miles.")
        return route_geom, total_distance

    def save_route(self, route_geom: LineString, route_name: str, overwrite: bool = False) -> bool:
        """
        Saves a route to the data source using the provided data manager.

        Args:
            route_geom (LineString): The route geometry to save.
            route_name (str): The name for the route.
            overwrite (bool): If True, overwrite an existing route with the same name.

        Returns:
            bool: True if the route was saved successfully, False otherwise.
        """
        logger.info(f"Saving route '{route_name}' to data source...")
        try:
            self.manager.save_route(
                route_geom=route_geom,
                route_name=route_name,
                overwrite=overwrite
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save route '{route_name}': {e}")
            return False

    def load_route(self, route_name: str) -> Optional[LineString]:
        """
        Loads a route from the data source using the provided data manager.

        Args:
            route_name (str): The name of the route to load.

        Returns:
            Optional[LineString]: The loaded route geometry, or None if not found.
        """
        logger.info(f"Loading route '{route_name}' from data source...")
        try:
            route_geom = self.manager.load_route(route_name)
            return route_geom
        except Exception as e:
            logger.error(f"Failed to load route '{route_name}': {e}")
            return None

    def detailed_route(self, departure_point: Point, arrival_point: Point,
                      astar_impl: Type['Astar'] = Astar,
                      weight_key: str = 'adjusted_weight',
                      collect_edge_stats: bool = True,
                      min_cost_factor: float = 1.0,
                      apply_smoothing: bool = False,
                      merge_threshold_deg: float = 1.0,
                      arc_threshold_deg: float = 3.0,
                      debug_export_path: Optional[Union[str, Path]] = None,
                      **pathfinder_kwargs) -> Optional[dict]:
        """
        Computes a route and collects detailed edge statistics for each segment.
        Optionally applies bearing-merge simplification and circular arc fillet
        smoothing with zone-based turning radii.

        This is a convenience wrapper around ``forced_route()`` with no waypoints.

        When ``astar_impl=AstarMaritime``, the returned dict also contains a
        ``maritime_metrics`` key with timing, corridor stats, and weight-factor
        counts, and ``summary_stats`` is populated from those metrics.

        When ``apply_smoothing=True``, the returned dict also contains a
        ``fillet_metadata`` key with per-vertex turning radius, zone assignment,
        and inherited edge data, and a ``smoothed_segments`` key with a list of
        leg/arc dicts for maritime visualization and export.

        Args:
            departure_point (Point): The starting geographic point.
            arrival_point (Point): The destination geographic point.
            astar_impl (Type[Astar]): The A* class to use for pathfinding.
            weight_key (str): The edge attribute to use for pathfinding cost.
                Defaults to 'adjusted_weight'.
            collect_edge_stats (bool): Whether to collect detailed edge statistics.
            min_cost_factor: Scale factor for A* heuristic admissibility (default: 1.0).
            apply_smoothing (bool): If True, applies bearing-merge simplification
                and circular arc fillets with zone-based turning radii.
            merge_threshold_deg (float): Max bearing difference (degrees) to merge
                consecutive segments (default 1.0).
            arc_threshold_deg (float): Min bearing difference (degrees) at a junction
                to trigger fillet arc construction (default 3.0).
            **pathfinder_kwargs: Extra keyword arguments forwarded to the pathfinder
                constructor (e.g., ``corridor_buffer_nm``, ``include_tss`` for
                ``AstarMaritime``).  ``pass1_backend`` is extracted and forwarded
                to ``compute_route_maritime_smooth()`` / ``compute_route_maritime()``.

        Returns:
            Optional[dict]: A dictionary containing:
                - 'route_geometry': LineString of the route
                - 'total_distance_nm': Total distance in nautical miles
                - 'num_edges': Number of edges in the route
                - 'edge_details': List of dicts with per-edge statistics (if collect_edge_stats=True)
                - 'summary_stats': Populated for AstarMaritime routes
                - 'maritime_metrics': (AstarMaritime only) timing, corridor, factor counts
                - 'waypoint_nodes': [departure_node, arrival_node]
                - 'segment_info': empty list

            Returns None if no route is found.

        Example:
            route_computer = Route(graph, manager)
            detailed_info = route_computer.detailed_route(
                Point(-122.4, 37.8),
                Point(-122.0, 37.6)
            )

            if detailed_info:
                print(f"Route distance: {detailed_info['total_distance_nm']:.2f} nm")
                print(f"Number of segments: {detailed_info['num_edges']}")

                # Export edge details to CSV
                df = pd.DataFrame(detailed_info['edge_details'])
                df.to_csv('route_analysis.csv', index=False)
        """
        return self.forced_route(
            departure_point=departure_point,
            arrival_point=arrival_point,
            waypoints=[],
            astar_impl=astar_impl,
            weight_key=weight_key,
            collect_edge_stats=collect_edge_stats,
            min_cost_factor=min_cost_factor,
            apply_smoothing=apply_smoothing,
            merge_threshold_deg=merge_threshold_deg,
            arc_threshold_deg=arc_threshold_deg,
            debug_export_path=debug_export_path,
            **pathfinder_kwargs,
        )

    def save_route_to_file(self, route_name: str, output_path: Union[str, 'Path'],
                          output_format: str = 'auto', layer_name: str = 'route',
                          route_properties: Optional[dict] = None) -> bool:
        """
        Loads a route from the data manager and exports it to a file (GeoPackage or GeoJSON).

        This method is useful for exporting routes stored in PostGIS to portable file formats,
        or for converting routes from one format to another.

        Args:
            route_name (str): The name of the route to load from the data manager.
            output_path (Union[str, Path]): The output file path.
            output_format (str): The output format ('gpkg', 'geojson', or 'auto' to infer from extension).
                                Defaults to 'auto'.
            layer_name (str): The layer name for GeoPackage output. Defaults to 'route'.
            route_properties (Optional[dict]): Additional properties to attach to the route feature.
                                              If None, default properties (route_name, distance) are added.

        Returns:
            bool: True if the route was successfully exported, False otherwise.

        Example:
            # Export PostGIS route to GeoJSON
            route_computer = Route(graph, postgis_manager)
            route_computer.save_route_to_file('my_route', 'output/my_route.geojson')

            # Export to GeoPackage with custom properties
            route_computer.save_route_to_file(
                'my_route',
                'output/routes.gpkg',
                layer_name='maritime_routes',
                route_properties={'vessel_type': 'cargo', 'draft': 12.5}
            )
        """
        output_path = Path(output_path)

        # Load the route from the data manager
        logger.info(f"Loading route '{route_name}' for export to file...")
        route_geom = self.load_route(route_name)

        if route_geom is None:
            logger.error(f"Cannot export route '{route_name}' - route not found or failed to load.")
            return False

        # Determine output format
        if output_format == 'auto':
            ext = output_path.suffix.lower()
            if ext == '.gpkg':
                output_format = 'gpkg'
            elif ext in ['.geojson', '.json']:
                output_format = 'geojson'
            else:
                logger.error(f"Cannot auto-detect format from extension '{ext}'. "
                           "Please specify output_format explicitly ('gpkg' or 'geojson').")
                return False

        # Validate format
        if output_format not in ['gpkg', 'geojson']:
            logger.error(f"Unsupported output format '{output_format}'. Use 'gpkg' or 'geojson'.")
            return False

        # Calculate route distance
        route_distance = self._calculate_route_distance(route_geom)

        # Prepare properties
        if route_properties is None:
            properties = {
                'route_name': route_name,
                'distance_nm': round(route_distance, 2)
            }
        else:
            properties = route_properties.copy()
            # Ensure route_name and distance are always included
            properties['route_name'] = route_name
            properties['distance_nm'] = round(route_distance, 2)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            [properties],
            geometry=[route_geom],
            crs='EPSG:4326'
        )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to file
        try:
            if output_format == 'gpkg':
                gdf.to_file(output_path, driver='GPKG', layer=layer_name)
                logger.info(f"Route '{route_name}' successfully exported to GeoPackage: {output_path} (layer: {layer_name})")
            elif output_format == 'geojson':
                gdf.to_file(output_path, driver='GeoJSON')
                logger.info(f"Route '{route_name}' successfully exported to GeoJSON: {output_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to export route '{route_name}' to {output_format.upper()}: {e}")
            return False

    def save_detailed_route_to_file(self, detailed_route_info: dict, output_path: Union[str, Path],
                                    output_format: str = 'auto', layer_name: str = 'route_edges',
                                    include_summary: bool = True) -> bool:
        """
        Saves detailed route with all edge statistics to a file (CSV, GeoJSON, or GeoPackage).

        This exports the complete edge-by-edge analysis including all weight factors,
        directional attributes, safety margins, and feature information.

        Args:
            detailed_route_info (dict): The dictionary returned by detailed_route() method
            output_path (Union[str, Path]): The output file path
            output_format (str): Output format ('csv', 'geojson', 'gpkg', or 'auto' to infer)
            layer_name (str): Layer name for GeoPackage output (default: 'route_edges')
            include_summary (bool): If True, also export summary stats as a separate layer/file

        Returns:
            bool: True if export was successful, False otherwise

        Example:
            route_computer = Route(graph, manager)
            detailed_info = route_computer.detailed_route(start, end)

            # Export to CSV with all columns
            route_computer.save_detailed_route_to_file(detailed_info, 'route_analysis.csv')

            # Export to GeoPackage with geometries and summary
            route_computer.save_detailed_route_to_file(
                detailed_info,
                'route_analysis.gpkg',
                include_summary=True
            )
        """
        output_path = Path(output_path)

        if not detailed_route_info or not detailed_route_info.get('edge_details'):
            logger.error("No edge details to export")
            return False

        # Determine output format
        if output_format == 'auto':
            ext = output_path.suffix.lower()
            if ext == '.csv':
                output_format = 'csv'
            elif ext == '.gpkg':
                output_format = 'gpkg'
            elif ext in ['.geojson', '.json']:
                output_format = 'geojson'
            else:
                logger.error(f"Cannot auto-detect format from extension '{ext}'")
                return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            edge_details = detailed_route_info['edge_details']

            if output_format == 'csv':
                # Export edge details to CSV
                df = pd.DataFrame(edge_details)
                df.to_csv(output_path, index=False)
                logger.info(f"Exported {len(df)} edge records to CSV: {output_path}")

            elif output_format in ['gpkg', 'geojson']:
                # Create geometries for each edge segment
                geometries = []
                for edge in edge_details:
                    source = (edge['source_lon'], edge['source_lat'])
                    target = (edge['target_lon'], edge['target_lat'])
                    geometries.append(LineString([source, target]))

                # Create GeoDataFrame with all columns
                gdf = gpd.GeoDataFrame(
                    edge_details,
                    geometry=geometries,
                    crs='EPSG:4326'
                )

                # Include the (possibly smoothed) route geometry as an extra feature
                route_geom = detailed_route_info.get('route_geometry')
                if route_geom is not None:
                    route_props = {
                        'feature_type': 'route',
                        'distance_nm': detailed_route_info.get('total_distance_nm', 0),
                        'num_edges': detailed_route_info.get('num_edges', 0),
                    }
                    mm = detailed_route_info.get('maritime_metrics')
                    if mm:
                        route_props['pass_used'] = mm.get('pass_used', '')
                    route_gdf = gpd.GeoDataFrame(
                        [route_props], geometry=[route_geom], crs='EPSG:4326'
                    )

                if output_format == 'gpkg':
                    gdf.to_file(output_path, driver='GPKG', layer=layer_name)
                    logger.info(f"Exported {len(gdf)} edge segments to GeoPackage layer '{layer_name}': {output_path}")
                    if route_geom is not None:
                        route_gdf.to_file(output_path, driver='GPKG', layer='route')
                        logger.info(f"Exported route geometry to GeoPackage layer 'route'.")

                else:  # geojson — smoothed route gets main filename, edges get _segments
                    smoothed_segments = detailed_route_info.get('smoothed_segments')
                    if smoothed_segments:
                        # Main file: smoothed route (clean visualization)
                        seg_records = []
                        seg_geoms = []
                        for seg_idx, seg in enumerate(smoothed_segments):
                            coords = seg['coords']
                            geom = LineString(coords)
                            record = {'seg_index': seg_idx, 'type': seg['type']}
                            if seg['type'] == 'leg':
                                record['distance_nm'] = seg.get('distance_nm', 0.0)
                                record['bearing_deg'] = seg.get('bearing_deg', 0.0)
                            elif seg['type'] == 'arc':
                                record['radius_nm'] = seg.get('radius_nm', 0.0)
                                record['turn_angle_deg'] = seg.get('turn_angle_deg', 0.0)
                                record['direction'] = seg.get('direction', '')
                            seg_records.append(record)
                            seg_geoms.append(geom)

                        seg_gdf = gpd.GeoDataFrame(seg_records, geometry=seg_geoms, crs='EPSG:4326')
                        seg_gdf.to_file(output_path, driver='GeoJSON')
                        n_legs = sum(1 for s in smoothed_segments if s['type'] == 'leg')
                        n_arcs = sum(1 for s in smoothed_segments if s['type'] == 'arc')
                        logger.info(f"Exported {len(seg_gdf)} smoothed segments "
                                    f"({n_legs} legs, {n_arcs} arcs) to {output_path}")

                        # Companion file: edge details (_segments suffix)
                        edge_path = output_path.with_name(
                            output_path.stem + '_segments.geojson'
                        )
                        gdf.to_file(edge_path, driver='GeoJSON')
                        logger.info(f"Exported {len(gdf)} edge features to GeoJSON: {edge_path}")
                    else:
                        gdf.to_file(output_path, driver='GeoJSON')
                        logger.info(f"Exported {len(gdf)} edge features to GeoJSON: {output_path}")

                # Export smoothed segments as GPKG layer (separate from GeoJSON logic)
                if output_format == 'gpkg':
                    smoothed_segments = detailed_route_info.get('smoothed_segments')
                    if smoothed_segments:
                        seg_records = []
                        seg_geoms = []
                        for seg_idx, seg in enumerate(smoothed_segments):
                            coords = seg['coords']
                            geom = LineString(coords)
                            record = {'seg_index': seg_idx, 'type': seg['type']}
                            if seg['type'] == 'leg':
                                record['distance_nm'] = seg.get('distance_nm', 0.0)
                                record['bearing_deg'] = seg.get('bearing_deg', 0.0)
                            elif seg['type'] == 'arc':
                                record['radius_nm'] = seg.get('radius_nm', 0.0)
                                record['turn_angle_deg'] = seg.get('turn_angle_deg', 0.0)
                                record['direction'] = seg.get('direction', '')
                            seg_records.append(record)
                            seg_geoms.append(geom)

                        seg_gdf = gpd.GeoDataFrame(seg_records, geometry=seg_geoms, crs='EPSG:4326')
                        seg_gdf.to_file(output_path, driver='GPKG', layer='smoothed_segments')
                        n_legs = sum(1 for s in smoothed_segments if s['type'] == 'leg')
                        n_arcs = sum(1 for s in smoothed_segments if s['type'] == 'arc')
                        logger.info(f"Exported {len(seg_gdf)} smoothed segments "
                                    f"({n_legs} legs, {n_arcs} arcs) to GeoPackage layer 'smoothed_segments'.")

            return True

        except Exception as e:
            logger.error(f"Failed to export detailed route: {e}")
            return False

    def forced_route(
        self,
        departure_point: Point,
        arrival_point: Point,
        waypoints: List[Union[int, Point, Tuple[float, float]]],
        astar_impl: Type['Astar'] = Astar,
        weight_key: str = 'adjusted_weight',
        collect_edge_stats: bool = True,
        min_cost_factor: float = 1.0,
        apply_smoothing: bool = False,
        merge_threshold_deg: float = 1.0,
        arc_threshold_deg: float = 3.0,
        debug_export_path: Optional[Union[str, Path]] = None,
        **pathfinder_kwargs,
    ) -> Optional[dict]:
        """
        Compute a route that must pass through a declared sequence of waypoints.

        Instead of a single departure->arrival A* search, the route is split into
        consecutive segments: departure -> wp[0] -> wp[1] -> ... -> arrival.  Each
        segment is solved independently with A*.  This lets you pin the route to
        specific nodes in fairway corridors or TSS lanes and analyse each section
        in isolation.

        Args:
            departure_point: Starting geographic point (Shapely Point).
            arrival_point: Destination geographic point (Shapely Point).
            waypoints: Ordered list of intermediate nodes.  Each element can be:
                - ``int``  -- integer node ID (requires ``node_id_map`` on init)
                - ``Tuple[float, float]`` -- (lon, lat) node coordinate tuple
                - ``Point`` -- Shapely Point (snapped to nearest graph node)
            astar_impl: A* class to use (``Astar`` or ``AstarImproved``).
            weight_key: Edge attribute used as routing cost.
            collect_edge_stats: Whether to populate ``edge_details`` in output.
            min_cost_factor: Heuristic scale factor for admissibility.
            apply_smoothing (bool): If True, applies bearing-merge simplification
                and circular arc fillets with zone-based turning radii.
            merge_threshold_deg (float): Max bearing difference (degrees) to merge
                consecutive segments (default 1.0).
            arc_threshold_deg (float): Min bearing difference (degrees) at a junction
                to trigger fillet arc construction (default 3.0).
            debug_export_path: If provided and ``astar_impl=AstarMaritimeSmooth``, exports
                a per-segment debug GeoPackage for QGIS inspection.
            **pathfinder_kwargs: Extra keyword arguments forwarded to the pathfinder
                constructor (e.g., ``corridor_buffer_nm``, ``include_tss``, ``sp_buffer_nm``).

        Returns:
            dict with keys:
                ``route_geometry``   -- LineString of the full combined route
                ``total_distance_nm`` -- total haversine distance in nautical miles
                ``num_edges``        -- total number of edges across all segments
                ``edge_details``     -- list of per-edge dicts (each has
                                       ``segment_index`` and ``edge_index``)
                ``summary_stats``    -- empty dict (reserved for future use)
                ``waypoint_nodes``   -- resolved (lon, lat) node sequence
                ``segment_info``     -- per-segment breakdown list, each entry:
                                       ``segment_index``, ``from_node``,
                                       ``to_node``, ``num_edges``, ``distance_nm``
                ``fillet_metadata``  -- (when apply_smoothing=True) per-vertex fillet info
                ``maritime_metrics``  -- (AstarMaritime/AstarMaritimeSmooth only) aggregated metrics
                ``shortcut_metadata``  -- (AstarMaritimeSmooth only) per-segment shortcut info
            Returns ``None`` if any segment has no path.
        """
        logger.info(
            f"Computing forced route with {len(waypoints)} waypoint(s) "
            f"using {astar_impl.__name__}..."
        )

        _ctor_kwargs = {k: v for k, v in pathfinder_kwargs.items()
                        if k not in ('apply_smoothing', 'merge_threshold_deg', 'arc_threshold_deg',
                                     'scout_path', 'pass1_backend', 'pass1_refresh')}
        pathfinder = astar_impl(self.graph, min_cost_factor=min_cost_factor,
                                **_ctor_kwargs)
        self._last_pathfinder = pathfinder
        _pass1_backend = pathfinder_kwargs.get('pass1_backend', 'rustworkx')
        _pass1_refresh = pathfinder_kwargs.get('pass1_refresh', True)

        # --- Resolve all points to graph nodes ---
        dep_node = pathfinder.find_nearest_node(departure_point)
        arr_node = pathfinder.find_nearest_node(arrival_point)
        if dep_node is None or arr_node is None:
            logger.error("Could not find nearest node for departure or arrival point.")
            return None

        try:
            wp_nodes = [self._resolve_waypoint(wp, pathfinder) for wp in waypoints]
        except (KeyError, ValueError) as exc:
            logger.error(f"Waypoint resolution failed: {exc}")
            return None

        node_sequence: List[Tuple[float, float]] = [dep_node] + wp_nodes + [arr_node]
        logger.info(f"Node sequence: {len(node_sequence)} nodes "
                    f"({len(node_sequence) - 1} segment(s))")

        # --- Phase A: Build multi-waypoint scout path ---
        scout_path: Optional[List[Tuple[float, float]]] = None
        scout_segments: List[List[Tuple[float, float]]] = []
        if len(node_sequence) > 2:
            for seg_idx in range(len(node_sequence) - 1):
                src, tgt = node_sequence[seg_idx], node_sequence[seg_idx + 1]
                try:
                    seg = nx.astar_path(
                        self.graph, src, tgt,
                        heuristic=pathfinder._heuristic, weight=weight_key,
                    )
                    scout_segments.append(seg)
                except nx.NetworkXNoPath:
                    logger.warning(f"No path for scout segment {seg_idx}: {src} -> {tgt}")
                    return None

            scout_path = scout_segments[0][:]
            for seg in scout_segments[1:]:
                scout_path.extend(seg[1:])

            logger.info(f"Scout path through {len(waypoints)} waypoint(s): "
                        f"{len(scout_path)} nodes.")

        # --- Phase B: Route ---
        dep_pt, arr_pt = Point(*dep_node), Point(*arr_node)
        is_maritime = isinstance(pathfinder, (AstarMaritimeSmooth, AstarMaritime))

        if is_maritime:
            # Maritime: single end-to-end call with scout_path for corridor
            if isinstance(pathfinder, AstarMaritimeSmooth):
                route_geom = pathfinder.compute_route_maritime_smooth(
                    dep_pt, arr_pt, weight_key=weight_key, scout_path=scout_path,
                    pass1_backend=_pass1_backend, pass1_refresh=_pass1_refresh)
            else:
                route_geom = pathfinder.compute_route_maritime(
                    dep_pt, arr_pt, weight_key=weight_key, scout_path=scout_path,
                    pass1_backend=_pass1_backend, pass1_refresh=_pass1_refresh)

            if route_geom is None:
                logger.warning("No route found for forced route.")
                return None

            if isinstance(pathfinder, AstarMaritimeSmooth):
                final_path = pathfinder._pass2_path_nodes if pathfinder._pass2_path_nodes is not None else list(route_geom.coords)[1:-1]
                maritime_metrics = pathfinder.get_maritime_metrics()
                shortcut_metadata = pathfinder.get_shortcut_metadata()
            else:
                final_path = pathfinder._pass2_path_nodes if pathfinder._pass2_path_nodes is not None else list(route_geom.coords)[1:-1]
                maritime_metrics = pathfinder.get_maritime_metrics()
                shortcut_metadata = None
        else:
            # Non-maritime: use scout_segments directly (per-segment A* routing)
            if not scout_segments:
                # No waypoints -- single end-to-end route
                if isinstance(pathfinder, AstarImproved):
                    route_geom = pathfinder.compute_route_improved(dep_pt, arr_pt, weight_key=weight_key)
                else:
                    route_geom = pathfinder.compute_route(dep_pt, arr_pt, weight_key=weight_key)
                if route_geom is None:
                    return None
                scout_segments = [list(route_geom.coords)[1:-1]]

            # Merge scout segments into final path
            final_path = scout_segments[0][:]
            for seg in scout_segments[1:]:
                final_path.extend(seg[1:])
            route_geom = LineString(final_path)
            maritime_metrics, shortcut_metadata = None, None

        # --- Fillet smoothing ---
        fillet_metadata = []
        smoothed_segments = []
        if apply_smoothing:
            logger.info("Applying fillet smoothing to the forced route...")
            route_geom, fillet_metadata, smoothed_segments = self.apply_fillet_smoothing(
                route_geom,
                merge_threshold_deg=merge_threshold_deg,
                arc_threshold_deg=arc_threshold_deg,
            )
            final_path = list(route_geom.coords)[1:-1]

        total_distance = self._calculate_route_distance(route_geom)

        # --- Scout-based segment info ---
        segment_info = []
        if scout_segments:
            for seg_idx, seg in enumerate(scout_segments):
                seg_dist = self._calculate_route_distance(LineString(seg))
                segment_info.append({
                    'segment_index': seg_idx,
                    'from_node': node_sequence[seg_idx],
                    'to_node': node_sequence[seg_idx + 1],
                    'num_edges': len(seg) - 1,
                    'distance_nm': round(seg_dist, 4),
                })

        # --- Phase D: Build result dict ---
        result: dict = {
            'route_geometry': route_geom,
            'total_distance_nm': total_distance,
            'num_edges': len(final_path) - 1,
            'edge_details': [],
            'summary_stats': {},
            'waypoint_nodes': node_sequence,
            'segment_info': segment_info,
        }

        if scout_path is not None:
            result['scout_path'] = scout_path
        if fillet_metadata:
            result['fillet_metadata'] = fillet_metadata
        if smoothed_segments:
            result['smoothed_segments'] = smoothed_segments

        if maritime_metrics is not None:
            result['maritime_metrics'] = maritime_metrics
            result['summary_stats'] = {
                'blocking_count': maritime_metrics['blocking_count'],
                'penalty_count': maritime_metrics['penalty_count'],
                'bonus_count': maritime_metrics['bonus_count'],
                'accumulated_weight': maritime_metrics['accumulated_weight'],
                'pass_used': maritime_metrics['pass_used'],
                'pass1_distance_nm': maritime_metrics.get('pass1_distance_nm'),
                'pass2_distance_nm': maritime_metrics.get('pass2_distance_nm'),
            }
        if shortcut_metadata:
            result['shortcut_metadata'] = shortcut_metadata

        # --- Debug export ---
        if debug_export_path is not None and isinstance(pathfinder, AstarMaritimeSmooth):
            pathfinder.export_debug_gpkg(debug_export_path)

        if not collect_edge_stats or len(final_path) < 2:
            return result

        # --- Collect edge statistics ---
        # Prefer paths with real graph edges.  scout_segments (multi-waypoint)
        # are first choice.  For single-segment AstarMaritimeSmooth routes,
        # string-pulling replaces the Dijkstra path with shortcuts, so use
        # the preserved Dijkstra path from _debug_pass2_path instead.
        if scout_segments:
            stats_paths = scout_segments
        elif (isinstance(pathfinder, AstarMaritimeSmooth)
              and pathfinder._debug_pass2_path is not None):
            stats_paths = [pathfinder._debug_pass2_path]
        else:
            stats_paths = [final_path]
        total_stats_edges = sum(len(p) - 1 for p in stats_paths)
        logger.info(f"Collecting statistics for {total_stats_edges} edges "
                     f"across {len(stats_paths)} segment(s)...")

        edge_details = []
        global_edge_idx = 0
        for seg_idx, seg_path in enumerate(stats_paths):
            for i in range(len(seg_path) - 1):
                source_node = seg_path[i]
                target_node = seg_path[i + 1]

                if not self.graph.has_edge(source_node, target_node):
                    if isinstance(self._last_pathfinder, AstarMaritimeSmooth):
                        logger.debug(
                            f"Shortcut segment {source_node} -> {target_node} "
                            f"(expected for string-pulling)"
                        )
                    else:
                        logger.warning(
                            f"Edge {source_node} -> {target_node} not found in graph"
                        )
                    continue

                edge_data = self.graph[source_node][target_node]
                seg_geom = LineString([source_node, target_node])
                seg_dist = self._calculate_route_distance(seg_geom)

                edge_stats = edge_data.copy()
                edge_stats.update({
                    'edge_index': global_edge_idx,
                    'segment_index': seg_idx,
                    'source_lon': source_node[0],
                    'source_lat': source_node[1],
                    'target_lon': target_node[0],
                    'target_lat': target_node[1],
                    'segment_distance_nm': round(seg_dist, 4),
                    'edge_id': edge_data.get('id', f"{source_node} -> {target_node}"),
                })
                edge_details.append(edge_stats)
                global_edge_idx += 1

        result['edge_details'] = edge_details
        logger.info(f"Collected statistics for {len(edge_details)} edges.")
        return result