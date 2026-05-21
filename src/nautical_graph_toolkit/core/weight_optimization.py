"""
Weight optimization and fine-tuning utilities for maritime navigation graphs.

This module contains:
- GraphWeightOptimizer: Stateless ML pipeline tools (PyTorch export/import, validation)
- FineTuning: Stateful tools for recalculating/updating edge weights in the database
"""
import logging
from pathlib import Path
from typing import Union, List, Dict, Any

import networkx as nx
import numpy as np
import pandas as pd
from sqlalchemy import text

from .s57_data import ENCDataFactory
from .graph import BaseGraph, GraphConfigManager, PerformanceMetrics

logger = logging.getLogger(__name__)


class GraphWeightOptimizer:
    """
    ML pipeline utilities for maritime graph weight optimization.

    Stateless tools for PyTorch training workflows: data export/import,
    vessel parameter encoding, historical route loading, and validation.

    These methods have zero dependency on WeightsOpen state (factory,
    classifier, config, layer_registry). They operate purely on graph
    data passed as arguments.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize GraphWeightOptimizer.

        Args:
            config: Optional graph config dict. Weight system constants are loaded
                    from config['weight_settings']['constants'] if available,
                    otherwise hardcoded defaults are used.
        """
        constants = {}
        if config:
            constants = config.get('weight_settings', {}).get('constants', {})
        self.BLOCKING_THRESHOLD = constants.get('blocking_threshold', 999.0)
        self.DEFAULT_MAX_PENALTY = constants.get('max_penalty', 50.0)
        self.MIN_BONUS_FACTOR = constants.get('min_bonus_factor', 0.5)

    def validate_against_weights(
        self,
        graph_open: nx.Graph,
        graph_closed: nx.Graph,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Verify WeightsOpen produces identical final results to Weights.

        Compares aggregated weight factors to ensure WeightsOpen's individual
        layer tracking doesn't affect final routing weights. This validation
        confirms that the new layer-tracking architecture maintains backward
        compatibility.

        Fields Compared:
        - wt_static_blocking (MAX aggregation validation)
        - wt_static_penalty (MULTIPLY aggregation validation)
        - wt_static_bonus (MULTIPLY aggregation validation)
        - blocking_factor (includes dynamic)
        - penalty_factor (includes dynamic)
        - bonus_factor (includes dynamic)
        - adjusted_weight (final routing weight)

        Args:
            graph_open: Graph processed with WeightsOpen methods
            graph_closed: Graph processed with Weights methods (ground truth)
            tolerance: Numerical tolerance for floating-point comparison

        Returns:
            Dict with validation results:
            {
                'match': bool,  # True if all fields match within tolerance
                'total_edges': int,
                'mismatches': [
                    {
                        'edge': (u, v),
                        'field': 'blocking_factor',
                        'open_value': 999.0,
                        'closed_value': 999.0,
                        'diff': 0.0
                    },
                    ...
                ],
                'stats': {
                    'blocking_match_rate': 0.995,
                    'penalty_match_rate': 0.998,
                    'bonus_match_rate': 1.0,
                    'adjusted_weight_match_rate': 0.997
                }
            }

        Example:
            # Compare outputs
            weights_closed = Weights(factory)
            weights_open = WeightsOpen(factory)

            graph1 = weights_closed.apply_static_weights(graph.copy(), ...)
            graph2 = weights_open.apply_static_weights_open(graph.copy(), ...)

            results = weights_open.validate_against_weights(graph2, graph1)
            assert results['match'], f"Validation failed: {results['stats']}"
        """
        fields_to_compare = [
            'wt_static_blocking',
            'wt_static_penalty',
            'wt_static_bonus',
            'blocking_factor',
            'penalty_factor',
            'bonus_factor',
            'adjusted_weight'
        ]

        mismatches = []
        total_edges = graph_open.number_of_edges()
        field_match_counts = {field: 0 for field in fields_to_compare}

        # Compare each edge
        for u, v in graph_open.edges():
            # Check if edge exists in both graphs
            if not graph_closed.has_edge(u, v):
                logger.warning(f"Edge ({u}, {v}) exists in open but not in closed graph")
                continue

            data_open = graph_open[u][v]
            data_closed = graph_closed[u][v]

            # Compare each field
            for field in fields_to_compare:
                val_open = data_open.get(field, 1.0)
                val_closed = data_closed.get(field, 1.0)
                diff = abs(val_open - val_closed)

                if diff > tolerance:
                    mismatches.append({
                        'edge': (u, v),
                        'field': field,
                        'open_value': val_open,
                        'closed_value': val_closed,
                        'diff': diff
                    })
                else:
                    field_match_counts[field] += 1

        # Calculate match statistics
        stats = {
            f'{field}_match_rate': field_match_counts[field] / total_edges
            for field in fields_to_compare
        }

        match = len(mismatches) == 0

        # Log results
        logger.info(f"=== WeightsOpen Validation ===")
        logger.info(f"Validation: {'PASS' if match else 'FAIL'}")
        logger.info(f"Total edges: {total_edges:,}")
        logger.info(f"Mismatches: {len(mismatches):,}")
        for field, rate in stats.items():
            logger.info(f"  {field}: {rate*100:.2f}% match")

        if not match and len(mismatches) > 0:
            logger.warning(f"First 5 mismatches:")
            for mismatch in mismatches[:5]:
                logger.warning(f"  Edge {mismatch['edge']}: {mismatch['field']} "
                             f"open={mismatch['open_value']:.6f} vs closed={mismatch['closed_value']:.6f} "
                             f"(diff={mismatch['diff']:.6e})")

        return {
            'match': match,
            'total_edges': total_edges,
            'mismatches': mismatches,
            'stats': stats
        }

    def export_for_pytorch(
        self,
        graph: nx.Graph,
        output_format: str = 'dataframe'
    ) -> Union[pd.DataFrame, Dict[str, Any], Dict]:
        """
        Export layer weights in ML-friendly format for PyTorch training.

        # TODO: WeightsOpen no longer writes wt_static_sources to graph edges.
        # Migrate this method to read from flat columns (wt_{name}/wt_{name}_n)
        # instead of the JSON nested dict.

        This method extracts individual layer weights from the graph's dual storage
        (wt_static_sources nested dict + wt_layer_* flat columns) and formats them for
        machine learning pipelines. Supports three output formats optimized for
        different ML workflows.

        Args:
            graph: NetworkX graph with wt_static_sources and wt_layer_* edge attributes
            output_format: Output format - 'dataframe', 'tensors', or 'dict'
                - 'dataframe': Pandas DataFrame with one row per edge, columns for each layer
                - 'tensors': PyTorch tensors dict ready for model input
                - 'dict': Raw nested wt_static_sources structure per edge

        Returns:
            dataframe format:
                pd.DataFrame with columns:
                - edge_id: "{u}_{v}" string identifier
                - u, v: node IDs
                - base_weight: original distance/weight
                - wt_layer_<name>: individual layer weights (one column per layer)
                - wt_dynamic_<name>: dynamic weight components (if present)
                - blocking_factor: aggregated blocking factor
                - penalty_factor: aggregated penalty factor
                - bonus_factor: aggregated bonus factor
                - adjusted_weight: final routing weight
                - geometry: edge geometry (if present)

            tensors format:
                Dict with PyTorch-ready tensors:
                - edge_features: torch.Tensor [n_edges, n_layers] - static layer weights
                - base_weights: torch.Tensor [n_edges] - original edge weights
                - dynamic_features: torch.Tensor [n_edges, n_dynamic] - dynamic weights (if present)
                - edge_ids: List[(u, v)] - edge identifiers
                - layer_names: List[str] - layer names corresponding to feature columns
                - dynamic_feature_names: List[str] - dynamic feature names
                - geometries: List[geometry] - edge geometries (if present)

            dict format:
                Dict mapping edge_id to:
                - wt_static_sources: nested dict with blocking/penalty/bonus tiers
                - base_weight: original weight
                - factors: dict with blocking/penalty/bonus aggregated factors

        Raises:
            ValueError: If output_format is not recognized
            ImportError: If 'tensors' format requested but torch not available

        Examples:
            >>> # Export to DataFrame for analysis
            >>> df = weights_open.export_for_pytorch(graph, 'dataframe')
            >>> print(df[['edge_id', 'wt_layer_lndare', 'wt_layer_fairwy']].head())

            >>> # Export to PyTorch tensors for training
            >>> tensors = weights_open.export_for_pytorch(graph, 'tensors')
            >>> edge_features = tensors['edge_features']  # [n_edges, n_layers]
            >>> layer_names = tensors['layer_names']       # ['lndare', 'fairwy', ...]

            >>> # Export to dict for inspection
            >>> data = weights_open.export_for_pytorch(graph, 'dict')
            >>> edge_wts = data['1_2']['wt_static_sources']  # Nested dict for edge (1, 2)

        Notes:
            - NaN values in layer columns are filled with 1.0 (neutral weight)
            - Only edges with wt_static_sources attribute are exported
            - Dynamic features are only included if present in graph
            - Tensor format requires PyTorch installation
        """
        if output_format == 'dataframe':
            rows = []
            for u, v, data in graph.edges(data=True):
                # Skip edges without wt_static_sources
                if 'wt_static_sources' not in data:
                    continue

                row = {
                    'edge_id': f"{u}_{v}",
                    'u': u,
                    'v': v,
                    'base_weight': data.get('weight', 1.0)
                }

                # Extract geometry if present
                if 'geom' in data:
                    row['geometry'] = data['geom']

                # Extract static layer weights from flat columns
                for key, value in data.items():
                    if key.startswith('wt_layer_'):
                        row[key] = value

                # Extract dynamic weight components
                for key in ['wt_dynamic_ukc_band', 'wt_dynamic_clearance',
                           'wt_dynamic_hazard', 'wt_dynamic_deep_water']:
                    if key in data:
                        row[key] = data[key]

                # Add aggregated factors
                row['blocking_factor'] = data.get('blocking_factor', 1.0)
                row['penalty_factor'] = data.get('penalty_factor', 1.0)
                row['bonus_factor'] = data.get('bonus_factor', 1.0)
                row['adjusted_weight'] = data.get('adjusted_weight', row['base_weight'])

                rows.append(row)

            df = pd.DataFrame(rows)

            # Fill NaN with 1.0 (neutral weight)
            layer_cols = [c for c in df.columns if c.startswith('wt_layer_') or c.startswith('wt_dynamic_')]
            if layer_cols:
                df[layer_cols] = df[layer_cols].fillna(1.0)

            logger.info(f"Exported {len(df)} edges × {len([c for c in df.columns if c.startswith('wt_layer_')])} layers to DataFrame")
            return df

        elif output_format == 'tensors':
            try:
                import torch
            except ImportError:
                raise ImportError("PyTorch is required for 'tensors' output format. Install with: pip install torch")

            # First get DataFrame
            df = self.export_for_pytorch(graph, 'dataframe')

            # Get static layer columns in sorted order
            layer_cols = sorted([c for c in df.columns if c.startswith('wt_layer_')])
            layer_names = [c.replace('wt_layer_', '') for c in layer_cols]

            # Get dynamic weight columns
            dynamic_cols = [c for c in df.columns if c.startswith('wt_dynamic_')]

            # Convert to tensors
            if layer_cols:
                edge_features = torch.tensor(df[layer_cols].values, dtype=torch.float32)
            else:
                edge_features = torch.zeros((len(df), 0), dtype=torch.float32)

            base_weights = torch.tensor(df['base_weight'].values, dtype=torch.float32)

            # Dynamic features (if present)
            if dynamic_cols:
                dynamic_features = torch.tensor(df[dynamic_cols].values, dtype=torch.float32)
            else:
                dynamic_features = None

            logger.info(f"Exported to tensors: {edge_features.shape[0]} edges × {edge_features.shape[1]} layers")

            return {
                'edge_features': edge_features,
                'base_weights': base_weights,
                'dynamic_features': dynamic_features,
                'edge_ids': list(zip(df['u'], df['v'])),
                'layer_names': layer_names,
                'dynamic_feature_names': dynamic_cols if dynamic_cols else [],
                'geometries': df['geometry'].tolist() if 'geometry' in df else None
            }

        elif output_format == 'dict':
            export = {}
            for u, v, data in graph.edges(data=True):
                # Skip edges without wt_static_sources
                if 'wt_static_sources' not in data:
                    continue

                edge_id = f"{u}_{v}"
                export[edge_id] = {
                    'wt_static_sources': data.get('wt_static_sources', {}),
                    'wt_dynamic_sources': data.get('wt_dynamic_sources', {}),
                    'base_weight': data.get('weight', 1.0),
                    'factors': {
                        'blocking': data.get('blocking_factor', 1.0),
                        'penalty': data.get('penalty_factor', 1.0),
                        'bonus': data.get('bonus_factor', 1.0)
                    }
                }

            logger.info(f"Exported {len(export)} edges to dict format")
            return export

        else:
            raise ValueError(f"Unknown output_format: '{output_format}'. Must be 'dataframe', 'tensors', or 'dict'")

    def encode_vessel_params(self, vessel_params: Dict[str, Any]) -> np.ndarray:
        """
        Encode vessel parameters as feature vector for machine learning.

        Converts vessel parameters dictionary into a fixed-length numerical feature
        vector suitable for PyTorch model input. Includes both continuous features
        (draft, height, beam, length, safety margins) and categorical features
        (vessel type as one-hot encoding).

        Args:
            vessel_params: Dictionary with vessel parameters containing:
                - draft (float): Vessel draft in meters (depth below waterline)
                - height (float): Air draft in meters (height above waterline)
                - beam (float): Vessel width in meters
                - length (float): Vessel length in meters
                - vessel_type (str): One of 'cargo', 'tanker', 'passenger', 'fishing'
                - ukc_safety_margin (float): UKC safety buffer in meters (default: 2.0)
                - ver_clearance_margin (float): Vertical clearance buffer in meters (default: 3.0)
                Note: Old keys (safety_margin, clearance_safety_margin) are still supported.

        Returns:
            np.ndarray: Feature vector of shape (10,) with:
                - [0]: draft (meters)
                - [1]: height (meters)
                - [2]: beam (meters)
                - [3]: length (meters)
                - [4]: ukc_safety_margin (meters)
                - [5]: ver_clearance_margin (meters)
                - [6-9]: vessel_type one-hot [cargo, tanker, passenger, fishing]

        Examples:
            >>> vessel_params = {
            ...     'draft': 7.5, 'height': 30.0, 'beam': 25.0, 'length': 150.0,
            ...     'vessel_type': 'cargo', 'ukc_safety_margin': 2.0,
            ...     'ver_clearance_margin': 3.0
            ... }
            >>> encoded = weights_open.encode_vessel_params(vessel_params)
            >>> print(encoded.shape)
            (10,)
            >>> print(encoded)
            [7.5, 30.0, 25.0, 150.0, 2.0, 3.0, 1, 0, 0, 0]  # cargo type

            >>> # Use in ML training
            >>> import torch
            >>> vessel_tensor = torch.from_numpy(encoded)

        Notes:
            - Default values are used if parameters are missing
            - Unknown vessel types default to 'cargo' one-hot encoding
            - All numerical features should be normalized for ML training
            - Vessel type mapping:
                * cargo: [1, 0, 0, 0]
                * tanker: [0, 1, 0, 0]
                * passenger: [0, 0, 1, 0]
                * fishing: [0, 0, 0, 1]
        """
        # Vessel type one-hot encoding mapping
        vessel_type_map = {
            'cargo': [1, 0, 0, 0],
            'tanker': [0, 1, 0, 0],
            'passenger': [0, 0, 1, 0],
            'fishing': [0, 0, 0, 1]
        }

        # Get vessel type and one-hot encode (default to cargo if unknown)
        vessel_type = vessel_params.get('vessel_type', 'cargo')
        vessel_onehot = vessel_type_map.get(vessel_type, [1, 0, 0, 0])

        # Build feature vector
        feature_vector = np.array([
            vessel_params.get('draft', 5.0),
            vessel_params.get('height', 25.0),
            vessel_params.get('beam', 15.0),
            vessel_params.get('length', 100.0),
            vessel_params.get('ukc_safety_margin',
                vessel_params.get('safety_margin', 2.0)),
            vessel_params.get('ver_clearance_margin',
                vessel_params.get('clearance_safety_margin',
                    vessel_params.get('clearance_safety', 3.0))),
            *vessel_onehot
        ], dtype=np.float32)

        return feature_vector

    def load_historical_routes(
        self,
        routes_file: str,
        format: str = 'csv'
    ) -> List[Dict[str, Any]]:
        """
        Load historical route data for machine learning training.

        Parses historical maritime routes from various file formats to create
        training datasets for PyTorch-based weight optimization. Routes include
        vessel parameters and waypoint sequences that represent actual navigation
        paths taken by vessels.

        Args:
            routes_file: Path to routes file (CSV, GeoPackage, or AIS data)
            format: File format - 'csv', 'gpkg', 'geojson', or 'ais'
                - 'csv': Simple CSV with route_id, vessel params, waypoints
                - 'gpkg': GeoPackage with routes table and LineString geometries
                - 'geojson': GeoJSON FeatureCollection with route LineStrings
                - 'ais': AIS data parser (placeholder for future implementation)

        Returns:
            List of route dictionaries, each containing:
                - route_id (str): Unique route identifier
                - vessel_params (Dict): Vessel parameters dict
                    * vessel_type, draft, height, beam, length, safety_margin, etc.
                - waypoints (List[Tuple]): List of (lon, lat) coordinates
                - edges (List[Tuple], optional): List of (u, v) edge IDs if graph nodes provided
                - geometry (LineString, optional): Shapely geometry if available
                - metadata (Dict): Additional route metadata (timestamp, source, etc.)

        CSV Format Requirements:
            Columns: route_id, vessel_type, draft, height, start_node, end_node, waypoints
            waypoints format: "lon1,lat1;lon2,lat2;lon3,lat3;..."
            Example row:
                R001,cargo,7.5,30.0,node_123,node_456,"25.5,80.2;25.6,80.3;25.7,80.4"

        GeoPackage Format Requirements:
            Table: 'routes' with columns:
                - route_id (INTEGER or TEXT): Unique route ID
                - vessel_params (TEXT): JSON string with vessel parameters
                - geom (LINESTRING): Route geometry
                - metadata (TEXT, optional): JSON string with additional data

        Raises:
            FileNotFoundError: If routes_file does not exist
            ValueError: If format is not recognized
            NotImplementedError: If 'ais' format is requested (future feature)

        Examples:
            >>> # Load CSV routes
            >>> routes = weights_open.load_historical_routes(
            ...     'data/historical_routes.csv',
            ...     format='csv'
            ... )
            >>> print(f"Loaded {len(routes)} routes")
            >>> print(routes[0]['vessel_params'])
            {'vessel_type': 'cargo', 'draft': 7.5, 'height': 30.0, ...}

            >>> # Load GeoPackage routes
            >>> routes = weights_open.load_historical_routes(
            ...     'data/routes.gpkg',
            ...     format='gpkg'
            ... )
            >>> print(routes[0]['waypoints'])
            [(25.5, 80.2), (25.6, 80.3), (25.7, 80.4)]

            >>> # Use for ML training
            >>> for route in routes:
            ...     vessel_vec = weights_open.encode_vessel_params(route['vessel_params'])
            ...     # Train model with route['waypoints'] as ground truth

        Notes:
            - CSV format is simplest but least flexible
            - GeoPackage format is recommended for complex routes with metadata
            - AIS format will enable direct import of real vessel track data
            - Historical routes can be used to train weights that minimize deviation
              from actual navigation patterns
        """
        import csv

        routes_path = Path(routes_file)
        if not routes_path.exists():
            raise FileNotFoundError(f"Routes file not found: {routes_file}")

        if format == 'csv':
            routes = []
            with open(routes_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse vessel parameters
                    vessel_params = {
                        'vessel_type': row['vessel_type'],
                        'draft': float(row['draft']),
                        'height': float(row.get('height', 25.0)),
                        'beam': float(row.get('beam', 15.0)),
                        'length': float(row.get('length', 100.0)),
                        'ukc_safety_margin': float(row.get('ukc_safety_margin',
                            row.get('safety_margin', 2.0))),
                        'ver_clearance_margin': float(row.get('ver_clearance_margin',
                            row.get('clearance_safety_margin', 3.0)))
                    }

                    # Parse waypoints (format: "lon1,lat1;lon2,lat2;...")
                    waypoint_str = row['waypoints']
                    waypoints = [
                        tuple(map(float, wp.split(',')))
                        for wp in waypoint_str.split(';')
                        if wp.strip()
                    ]

                    routes.append({
                        'route_id': row['route_id'],
                        'vessel_params': vessel_params,
                        'waypoints': waypoints,
                        'start_node': row.get('start_node'),
                        'end_node': row.get('end_node'),
                        'metadata': {
                            'source': row.get('source', 'historical'),
                            'timestamp': row.get('timestamp')
                        }
                    })

            logger.info(f"Loaded {len(routes)} historical routes from CSV: {routes_file}")
            return routes

        elif format == 'gpkg':
            import geopandas as gpd
            import json

            routes_gdf = gpd.read_file(routes_file, layer='routes')
            routes = []

            for idx, row in routes_gdf.iterrows():
                # Parse vessel parameters from JSON
                vessel_params = json.loads(row['vessel_params'])

                # Extract waypoints from LineString geometry
                geom = row['geometry']
                waypoints = list(geom.coords)

                # Parse metadata if present
                metadata = {}
                if 'metadata' in row and pd.notna(row['metadata']):
                    try:
                        metadata = json.loads(row['metadata'])
                    except json.JSONDecodeError:
                        metadata = {'raw': row['metadata']}

                routes.append({
                    'route_id': str(row['route_id']),
                    'vessel_params': vessel_params,
                    'waypoints': waypoints,
                    'geometry': geom,
                    'metadata': metadata
                })

            logger.info(f"Loaded {len(routes)} historical routes from GeoPackage: {routes_file}")
            return routes

        elif format == 'geojson':
            import geopandas as gpd
            import json

            routes_gdf = gpd.read_file(routes_file)
            routes = []

            for idx, row in routes_gdf.iterrows():
                # Try to get vessel params from properties
                properties = row.get('properties', {}) if hasattr(row, 'get') else {}

                # Parse vessel parameters
                if isinstance(properties, str):
                    properties = json.loads(properties)

                vessel_params = properties.get('vessel_params', {})
                if not vessel_params:
                    # Try individual fields
                    vessel_params = {
                        'vessel_type': properties.get('vessel_type', 'cargo'),
                        'draft': float(properties.get('draft', 5.0)),
                        'height': float(properties.get('height', 25.0)),
                        'beam': float(properties.get('beam', 15.0)),
                        'length': float(properties.get('length', 100.0)),
                        'ukc_safety_margin': float(properties.get('ukc_safety_margin',
                            properties.get('safety_margin', 2.0))),
                        'ver_clearance_margin': float(properties.get('ver_clearance_margin',
                            properties.get('clearance_safety_margin', 3.0)))
                    }

                # Extract waypoints from geometry
                geom = row['geometry']
                waypoints = list(geom.coords)

                routes.append({
                    'route_id': properties.get('route_id', f'route_{idx}'),
                    'vessel_params': vessel_params,
                    'waypoints': waypoints,
                    'geometry': geom,
                    'metadata': properties
                })

            logger.info(f"Loaded {len(routes)} historical routes from GeoJSON: {routes_file}")
            return routes

        elif format == 'ais':
            # Future: AIS data parser
            # Would parse AIS messages (NMEA format) and reconstruct vessel tracks
            raise NotImplementedError(
                "AIS format loader not yet implemented. "
                "Future enhancement will support direct AIS data parsing. "
                "For now, please convert AIS tracks to CSV or GeoPackage format."
            )

        else:
            raise ValueError(
                f"Unknown format: '{format}'. "
                f"Supported formats: 'csv', 'gpkg', 'geojson', 'ais' (coming soon)"
            )

    def import_learned_weights(
        self,
        graph: nx.Graph,
        learned_weights: Union[pd.DataFrame, Dict[str, float], np.ndarray],
        mode: str = 'replace'
    ) -> nx.Graph:
        """
        Apply PyTorch-learned weights back to graph.

        Takes weights learned from machine learning training and applies them to
        the graph's individual layer weights (wt_layer_* columns and wt_static_sources
        nested dict). After updating individual layer weights, re-aggregates the
        three tiers (blocking/penalty/bonus) to compute final routing weights.

        Args:
            graph: NetworkX graph with existing wt_layer_* columns and wt_static_sources
            learned_weights: Learned layer weights in one of three formats:
                - pd.DataFrame: columns 'layer_name', 'learned_weight'
                - Dict[str, float]: {layer_name: learned_weight}
                - np.ndarray: array of weights (must match layer order from export)
            mode: How to apply learned weights - 'replace', 'blend', or 'additive'
                - 'replace': Overwrite original layer weights completely
                - 'blend': Average of original and learned (0.5 × original + 0.5 × learned)
                - 'additive': Multiply learned adjustment to original (original × learned)

        Returns:
            nx.Graph: Updated graph with:
                - Modified wt_layer_* columns
                - Updated wt_static_sources nested dict
                - Re-aggregated wt_static_blocking/penalty/bonus
                - Recalculated adjusted_weight

        Raises:
            ValueError: If learned_weights type is not recognized or mode is invalid
            KeyError: If layer names in learned_weights don't match graph layers

        Examples:
            >>> # After PyTorch training, extract learned layer importance
            >>> learned_importance = {
            ...     'lndare': 1200.0,  # Learned to increase land blocking
            ...     'fairwy': 0.4,     # Learned to prefer fairways more
            ...     'tsslpt': 3.0      # Learned traffic separation penalty
            ... }

            >>> # Apply learned weights (replace mode)
            >>> graph_optimized = weights_open.import_learned_weights(
            ...     graph, learned_importance, mode='replace'
            ... )

            >>> # Or blend with original weights (conservative)
            >>> graph_blended = weights_open.import_learned_weights(
            ...     graph, learned_importance, mode='blend'
            ... )

            >>> # Or apply as multiplicative adjustment
            >>> graph_adjusted = weights_open.import_learned_weights(
            ...     graph, learned_importance, mode='additive'
            ... )

            >>> # Compare routes
            >>> route_original = calculate_route(graph, start, end, vessel_params)
            >>> route_learned = calculate_route(graph_optimized, start, end, vessel_params)

        Notes:
            - Learned weights should be on the same scale as original layer weights
            - 'replace' mode is most aggressive, use when confident in learned weights
            - 'blend' mode is conservative, good for gradual adaptation
            - 'additive' mode treats learned weights as multipliers/adjustments
            - After import, you may want to recalculate dynamic weights if vessel
              parameters have changed
            - Re-aggregation uses same logic as apply_static_weights_open():
                * blocking: MAX aggregation
                * penalty: MULTIPLY aggregation (capped at DEFAULT_MAX_PENALTY)
                * bonus: MULTIPLY aggregation (floored at MIN_BONUS_FACTOR)
        """
        G = graph.copy()

        # Convert learned_weights to dict format
        if isinstance(learned_weights, pd.DataFrame):
            if 'layer_name' not in learned_weights.columns or 'learned_weight' not in learned_weights.columns:
                raise ValueError("DataFrame must have 'layer_name' and 'learned_weight' columns")
            weights_dict = dict(zip(
                learned_weights['layer_name'],
                learned_weights['learned_weight']
            ))

        elif isinstance(learned_weights, np.ndarray):
            # Infer layer order from first edge with wt_layer_* columns
            sample_edge = None
            for u, v, data in G.edges(data=True):
                if any(k.startswith('wt_layer_') for k in data.keys()):
                    sample_edge = data
                    break

            if sample_edge is None:
                raise ValueError("Graph has no wt_layer_* columns. Run apply_static_weights_open() first.")

            layer_cols = sorted([k for k in sample_edge.keys() if k.startswith('wt_layer_')])
            layer_names = [c.replace('wt_layer_', '') for c in layer_cols]

            if len(learned_weights) != len(layer_names):
                raise ValueError(
                    f"Array length {len(learned_weights)} doesn't match number of layers {len(layer_names)}. "
                    f"Expected order: {layer_names}"
                )

            weights_dict = dict(zip(layer_names, learned_weights))

        elif isinstance(learned_weights, dict):
            weights_dict = learned_weights

        else:
            raise ValueError(
                f"Unknown learned_weights type: {type(learned_weights)}. "
                f"Must be pd.DataFrame, dict, or np.ndarray"
            )

        # Validate mode
        if mode not in ['replace', 'blend', 'additive']:
            raise ValueError(f"Unknown mode: '{mode}'. Must be 'replace', 'blend', or 'additive'")

        logger.info(f"Importing learned weights for {len(weights_dict)} layers (mode={mode})")

        # Track which layers were updated
        updated_edges = 0
        updated_layers = set()

        # Apply learned weights to edges
        for u, v, data in G.edges(data=True):
            edge_updated = False

            for layer, learned_weight in weights_dict.items():
                col_name = f'wt_layer_{layer}'

                if col_name in data:
                    original_weight = data[col_name]

                    # Apply learned weight based on mode
                    if mode == 'replace':
                        new_weight = learned_weight
                    elif mode == 'blend':
                        new_weight = 0.5 * original_weight + 0.5 * learned_weight
                    elif mode == 'additive':
                        new_weight = original_weight * learned_weight

                    G[u][v][col_name] = new_weight
                    updated_layers.add(layer)
                    edge_updated = True

                    # Update wt_static_sources nested dict
                    if 'wt_static_sources' in G[u][v]:
                        wt_static_sources = G[u][v]['wt_static_sources']
                        for tier in ['static_blocking', 'static_penalty', 'static_bonus']:
                            if layer in wt_static_sources.get(tier, {}):
                                wt_static_sources[tier][layer] = new_weight

            if edge_updated:
                updated_edges += 1

        logger.info(f"Updated {updated_edges:,} edges across {len(updated_layers)} layers: {sorted(updated_layers)}")

        # Re-aggregate tiers with learned weights
        logger.info("Re-aggregating tiers with learned weights...")

        for u, v, data in G.edges(data=True):
            if 'wt_static_sources' not in data:
                continue

            wt_static_sources = data['wt_static_sources']

            # Blocking: MAX aggregation
            blocking_weights = list(wt_static_sources.get('static_blocking', {}).values())
            if blocking_weights:
                G[u][v]['wt_static_blocking'] = max(blocking_weights)

            # Penalty: MULTIPLY aggregation (capped)
            penalty_factor = 1.0
            for weight in wt_static_sources.get('static_penalty', {}).values():
                penalty_factor *= weight
            G[u][v]['wt_static_penalty'] = min(penalty_factor, self.DEFAULT_MAX_PENALTY)

            # Bonus: MULTIPLY aggregation (floored)
            bonus_factor = 1.0
            for weight in wt_static_sources.get('static_bonus', {}).values():
                bonus_factor *= weight
            G[u][v]['wt_static_bonus'] = max(bonus_factor, self.MIN_BONUS_FACTOR)

            # Recalculate final factors (combine static + dynamic if present)
            static_blocking = G[u][v].get('wt_static_blocking', 1.0)
            static_penalty = G[u][v].get('wt_static_penalty', 1.0)
            static_bonus = G[u][v].get('wt_static_bonus', 1.0)

            # If dynamic weights exist, combine them
            wt_dynamic_sources = data.get('wt_dynamic_sources', {})
            dynamic_blocking_weights = list(wt_dynamic_sources.get('dynamic_blocking', {}).values())
            dynamic_blocking = max(dynamic_blocking_weights) if dynamic_blocking_weights else 1.0

            blocking_factor = max(static_blocking, dynamic_blocking)
            G[u][v]['blocking_factor'] = blocking_factor

            # Penalty: static × dynamic components (capped)
            penalty_factor = static_penalty
            for weight in wt_dynamic_sources.get('dynamic_penalty', {}).values():
                penalty_factor *= weight
            penalty_factor = min(penalty_factor, self.DEFAULT_MAX_PENALTY)
            G[u][v]['penalty_factor'] = penalty_factor

            # Bonus: static × dynamic components (floored)
            bonus_factor = static_bonus
            for weight in wt_dynamic_sources.get('dynamic_bonus', {}).values():
                bonus_factor *= weight
            bonus_factor = max(bonus_factor, self.MIN_BONUS_FACTOR)
            G[u][v]['bonus_factor'] = bonus_factor

            # Recalculate adjusted_weight
            base_weight = data.get('base_weight', data.get('weight', 1.0))
            directional_factor = data.get('directional_factor', data.get('wt_dir', 1.0))
            adjusted_weight = base_weight * blocking_factor * penalty_factor * bonus_factor * directional_factor

            G[u][v]['adjusted_weight'] = adjusted_weight

        logger.info("✓ Learned weights imported and tiers re-aggregated")
        return G


class FineTuning:
    """
    Fine-tuning utilities for graph edge weight adjustments.

    This class provides methods for recalculating and updating edge weights
    based on various factors such as directional differences, traffic patterns,
    and other maritime navigation considerations.
    """

    def __init__(self, data_factory: ENCDataFactory, graph_schema: str = 'graph', config_path: Union[str, Path] = None):
        """
        Initialize the FineTuning class.

        Args:
            data_factory (ENCDataFactory): An initialized factory for accessing ENC data.
            graph_schema (str): Schema name for graph tables (PostGIS) or database path (file-based)
            config_path (Union[str, Path], optional): Path to graph_config.yml. If None, uses default location.
        """
        self.factory = data_factory
        self.graph_schema = graph_schema
        self.performance = PerformanceMetrics()

        # Load configuration
        if config_path is None:
            # Default to data directory
            config_path = Path(__file__).parent.parent / 'data' / 'graph_config.yml'

        self.config_manager = GraphConfigManager(config_path)
        self.config = self.config_manager.data

        # Extract directional weight configuration
        self.dir_config = self.config.get('weight_settings', {}).get('directional_weights', {})

        logger.info(f"FineTuning initialized with schema: {graph_schema}")
        logger.info(f"Directional weights enabled: {self.dir_config.get('enabled', False)}")

    def reapply_directional_weights(self, table_prefix: str = "graph",
                                   batch_size: int = 10000,
                                   commit_interval: int = 50000) -> Dict[str, Any]:
        """
        Recalculate and update wt_dir (directional weight) based on dir_diff.

        This method reads the directional difference (dir_diff) column from the edges table
        and applies weight factors based on the configured angle bands from graph_config.yml.

        Process:
            1. Load directional weight configuration from graph_config.yml
            2. Read edges in batches with dir_diff and dir_trafic values
            3. For each edge, determine the appropriate weight based on:
               - Angular difference (dir_diff) between edge bearing and feature orientation
               - Two-way traffic handling (TRAFIC=4) for reverse direction checking
            4. Update wt_dir column in the database

        Configuration used from graph_config.yml:
            - weight_settings.directional_weights.enabled: Enable/disable processing
            - weight_settings.directional_weights.angle_bands: Weight factors by angle range
            - weight_settings.directional_weights.two_way_traffic: Reverse direction handling
            - weight_settings.directional_weights.apply_to_layers: Layer filter (optional)

        Args:
            table_prefix (str): Prefix for graph tables (default: "graph")
                               Uses {prefix}_edges table
            batch_size (int): Number of edges to process per batch (default: 10000)
            commit_interval (int): Number of edges to commit at once (default: 50000)

        Returns:
            Dict[str, Any]: Processing statistics:
                - 'total_edges': Total number of edges in table
                - 'edges_with_dir_diff': Number of edges with directional data
                - 'edges_updated': Number of edges where wt_dir was updated
                - 'processing_time': Total processing time in seconds
                - 'update_rate': Edges updated per second

        Raises:
            ValueError: If directional weights are disabled in configuration
            ValueError: If required columns (dir_diff, wt_dir) are missing

        Example:
                factory = ENCDataFactory.create_postgis("postgresql://user:pass@localhost/db")
            fine_tuning = FineTuning(factory, graph_schema='graph')

            stats = fine_tuning.reapply_directional_weights(
                 table_prefix='fine_graph_01',
                 batch_size=10000
            )
            logger.info(f"Updated {stats['edges_updated']:,} edges in {stats['processing_time']:.2f}s")
        """
        self.performance.start_timer("reapply_directional_weights_total")

        # Check if directional weights are enabled
        if not self.dir_config.get('enabled', False):
            raise ValueError("Directional weights are disabled in configuration. "
                           "Set weight_settings.directional_weights.enabled to true in graph_config.yml")

        # Validate table prefix
        validated_prefix = BaseGraph._validate_identifier(table_prefix, "table prefix")
        edges_table = f"{validated_prefix}_edges"

        logger.info(f"=== Reapplying Directional Weights ===")
        logger.info(f"Target table: {self.graph_schema}.{edges_table}")
        logger.info(f"Batch size: {batch_size:,}, Commit interval: {commit_interval:,}")

        # Extract angle bands configuration
        angle_bands = self.dir_config.get('angle_bands', [])
        if not angle_bands:
            raise ValueError("No angle_bands defined in configuration")

        # Sort angle bands by max_angle for efficient lookup
        angle_bands_sorted = sorted(angle_bands, key=lambda x: x['max_angle'])

        # Extract two-way traffic configuration
        two_way_config = self.dir_config.get('two_way_traffic', {})
        two_way_enabled = two_way_config.get('enabled', True)
        reverse_threshold = two_way_config.get('reverse_check_threshold', 95)

        logger.info(f"Angle bands configured: {len(angle_bands_sorted)}")
        logger.info(f"Two-way traffic handling: {'enabled' if two_way_enabled else 'disabled'}")
        if two_way_enabled:
            logger.info(f"Reverse check threshold: {reverse_threshold}°")

        try:
            with self.factory.manager.engine.connect() as conn:
                # Build qualified table name using the graph schema (not the ENC data schema)
                if self.graph_schema:
                    edges_qualified = f'"{self.graph_schema}"."{edges_table}"'
                else:
                    edges_qualified = f'"{edges_table}"'

                # Check if required columns exist
                self.performance.start_timer("column_check_time")
                check_cols_sql = text(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    AND column_name IN ('dir_diff', 'dir_trafic', 'wt_dir', 'ft_layer')
                """)

                existing_cols = {row[0] for row in conn.execute(check_cols_sql, {'table_name': edges_table})}

                if 'dir_diff' not in existing_cols:
                    raise ValueError(f"Column 'dir_diff' not found in {edges_table}. "
                                   "Ensure directional weights were calculated during graph enrichment.")
                if 'wt_dir' not in existing_cols:
                    raise ValueError(f"Column 'wt_dir' not found in {edges_table}")

                has_trafic = 'dir_trafic' in existing_cols
                has_layer = 'ft_layer' in existing_cols

                self.performance.end_timer("column_check_time")
                logger.info(f"Required columns present: dir_diff, wt_dir")
                logger.info(f"Optional columns: dir_trafic={has_trafic}, ft_layer={has_layer}")

                # Get total edge count
                self.performance.start_timer("count_edges_time")
                count_sql = text(f"SELECT COUNT(*) FROM {edges_qualified}")
                total_edges = conn.execute(count_sql).scalar()
                self.performance.end_timer("count_edges_time")

                logger.info(f"Total edges in table: {total_edges:,}")

                # Count edges with directional data
                self.performance.start_timer("count_dir_edges_time")
                count_dir_sql = text(f"SELECT COUNT(*) FROM {edges_qualified} WHERE dir_diff IS NOT NULL")
                edges_with_dir = conn.execute(count_dir_sql).scalar()
                self.performance.end_timer("count_dir_edges_time")

                logger.info(f"Edges with directional data: {edges_with_dir:,} ({edges_with_dir/total_edges*100:.1f}%)")

                if edges_with_dir == 0:
                    logger.warning("No edges have directional data (dir_diff). Nothing to update.")
                    return {
                        'total_edges': total_edges,
                        'edges_with_dir_diff': 0,
                        'edges_updated': 0,
                        'processing_time': 0.0,
                        'update_rate': 0.0
                    }

                # Process edges in batches
                self.performance.start_timer("batch_processing_time")
                edges_updated = 0
                batch_num = 0

                # Build SELECT query with optional columns
                select_cols = "id, dir_diff"
                if has_trafic:
                    select_cols += ", dir_trafic"
                if has_layer:
                    select_cols += ", ft_layer"

                # Read edges with directional data
                select_sql = text(f"""
                    SELECT {select_cols}
                    FROM {edges_qualified}
                    WHERE dir_diff IS NOT NULL
                    ORDER BY id
                """)

                # Prepare updates list
                updates = []

                logger.info("Processing edges...")
                result = conn.execute(select_sql)

                for row in result:
                    edge_id = row.id
                    dir_diff = row.dir_diff
                    dir_trafic = row.dir_trafic if has_trafic else None
                    ft_layer = row.ft_layer if has_layer else None

                    # Check layer filter if configured
                    apply_to_layers = self.dir_config.get('apply_to_layers')
                    if apply_to_layers and ft_layer:
                        if ft_layer not in apply_to_layers:
                            continue  # Skip this edge, layer not in filter

                    # Handle two-way traffic (TRAFIC=4)
                    effective_diff = dir_diff
                    if two_way_enabled and dir_trafic == 4 and dir_diff > reverse_threshold:
                        # Check reverse direction (orient + 180)
                        reverse_diff = abs(180 - dir_diff)
                        if reverse_diff < dir_diff:
                            effective_diff = reverse_diff

                    # Find matching angle band
                    wt_dir = 1.0  # Default weight
                    for band in angle_bands_sorted:
                        if effective_diff <= band['max_angle']:
                            wt_dir = band['weight']
                            break

                    # Add to updates
                    updates.append({'edge_id': edge_id, 'wt_dir': wt_dir})

                    # Commit batch if we've reached the interval
                    if len(updates) >= commit_interval:
                        self._execute_batch_update(conn, edges_qualified, updates)
                        edges_updated += len(updates)
                        batch_num += 1
                        logger.info(f"Batch {batch_num}: Updated {edges_updated:,} edges")
                        updates = []

                # Commit remaining updates
                if updates:
                    self._execute_batch_update(conn, edges_qualified, updates)
                    edges_updated += len(updates)
                    batch_num += 1
                    logger.info(f"Batch {batch_num} (final): Updated {edges_updated:,} edges")

                batch_time = self.performance.end_timer("batch_processing_time")

                # Update table statistics
                self.performance.start_timer("analyze_time")
                conn.execute(text(f"ANALYZE {edges_qualified}"))
                conn.commit()
                analyze_time = self.performance.end_timer("analyze_time")
                logger.info(f"Updated table statistics in {analyze_time:.3f}s")

        except Exception as e:
            logger.error(f"Failed to reapply directional weights: {e}")
            raise

        total_time = self.performance.end_timer("reapply_directional_weights_total")

        # Prepare summary
        summary = {
            'total_edges': total_edges,
            'edges_with_dir_diff': edges_with_dir,
            'edges_updated': edges_updated,
            'processing_time': total_time,
            'update_rate': edges_updated / total_time if total_time > 0 else 0.0
        }

        logger.info(f"=== Directional Weight Update Complete ===")
        logger.info(f"Total edges: {total_edges:,}")
        logger.info(f"Edges with directional data: {edges_with_dir:,}")
        logger.info(f"Edges updated: {edges_updated:,}")
        logger.info(f"Processing time: {total_time:.3f}s")
        logger.info(f"Update rate: {summary['update_rate']:,.0f} edges/sec")

        self.performance.log_summary("Directional Weight Reapplication")

        return summary

    def _execute_batch_update(self, conn, edges_qualified: str, updates: List[Dict]) -> None:
        """
        Execute a batch update of wt_dir values using PostgreSQL's efficient UPDATE FROM.

        Args:
            conn: Database connection
            edges_qualified: Qualified table name
            updates: List of dicts with 'edge_id' and 'wt_dir' keys
        """
        if not updates:
            return

        update_sql = text(f"""
            UPDATE {edges_qualified}
            SET wt_dir = :wt_dir
            WHERE id = :id
        """)
        conn.execute(update_sql, [{'id': u['edge_id'], 'wt_dir': u['wt_dir']} for u in updates])
        conn.commit()