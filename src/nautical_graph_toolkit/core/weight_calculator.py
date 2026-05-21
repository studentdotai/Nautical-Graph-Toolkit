#!/usr/bin/env python3
# Copyright (C) 2024-2025 Viktor Kolbasov <contact@studentdotai.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
weight_calculator.py

Stateless weight calculation algorithms for maritime navigation graphs.

This module provides the core weight calculation logic separated from graph management,
enabling:
1. Single source of truth for weight logic
2. Easy testing without graph dependencies
3. Reusability across Weights/WeightsOpen classes
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from ..utils.s57_classification import NavClass
from ..utils.geometry_utils import Buffer, Bearing

logger = logging.getLogger(__name__)


class WeightCalculator:
    """
    Stateless weight calculation algorithms.

    Separated from graph management to enable:
    1. Single source of truth for weight logic
    2. Easy testing without graph dependencies
    3. Reusability across Weights/WeightsOpen

    The calculator is initialized with an S57Classifier instance and provides
    methods for calculating static, dynamic, and directional weights.
    """

    # Legacy tier degradation constants – kept for backward compatibility but
    # no longer used by the simplified single-band buffer logic.
    CAUTION_DEGRADE_FACTOR = 10.0
    SAFE_DEGRADE_FACTOR = 2.0
    SAFE_AMPLIFY_FACTOR = 4.0

    # wt_dir scale: 1.0 = aligned (reward), 2.0 = neutral (open-water baseline), >2.0 = penalty.
    DEFAULT_ANGLE_BANDS = (
        {'max_angle': 30,  'weight': 1.0,   'name': 'aligned'},
        {'max_angle': 60,  'weight': 2.5,   'name': 'small deviation'},
        {'max_angle': 85,  'weight': 10.0,  'name': 'big deviation'},
        {'max_angle': 95,  'weight': 50.0,  'name': 'crossing'},
        {'max_angle': 180, 'weight': 200.0, 'name': 'opposite'},
    )

    def __init__(self, classifier, blocking_threshold: float = 999.0,
                 max_penalty: float = 100.0, min_bonus_factor: float = 0.3,
                 open_water_base_multiplier: float = 2.0,
                 ukc_restricted_penalty: float = 30.0, ukc_shallow_penalty: float = 4.0,
                 ukc_safe_penalty: float = 2.5, deep_water_bonus: float = 1.5,
                 sounding_high_risk: float = 20.0, sounding_moderate_risk: float = 4.0,
                 clearance_restricted_penalty: float = 50.0, anchorage_bonus: float = 1.5,
                 smooth_mode: bool = False, bonus_decay_rate: float = 3.0,
                 penalty_hazard_scale: float = 1.0, sounding_hazard_weight: float = 0.5,
                 step_band_bonus_strength: float = 0.85):
        """
        Initialize the WeightCalculator.

        Args:
            classifier: An instance of S57Classifier for maritime domain knowledge
            blocking_threshold: Value for absolute blocking constraints (default: 999.0)
            max_penalty: Maximum cumulative penalty to prevent explosion (default: 100.0)
            min_bonus_factor: Minimum bonus_factor floor (default: 0.3)
            open_water_base_multiplier: Default cost for undesignated open water (default: 2.0)
            ukc_restricted_penalty: Penalty for 0 < UKC <= safety_margin (default: 30.0)
            ukc_shallow_penalty: Penalty for safety_margin < UKC <= 0.5*draft (default: 4.0)
            ukc_safe_penalty: Penalty for 0.5*draft < UKC <= draft (default: 2.5)
            deep_water_bonus: Divisor bonus for UKC > draft; reduces cost (default: 1.5)
            sounding_high_risk: Penalty for 0 < sounding_ukc <= safety_margin (default: 20.0)
            sounding_moderate_risk: Penalty for safety_margin < sounding_ukc <= draft (default: 4.0)
            clearance_restricted_penalty: Penalty for restricted vertical clearance (default: 50.0)
            anchorage_bonus: Divisor bonus for preferred anchorage category match (default: 1.5)
            smooth_mode: Use continuous exp/log functions instead of step bands (default: False)
            bonus_decay_rate: k in bonus_factor = 1 + exp(-k * preference_score) (default: 3.0)
            penalty_hazard_scale: scalar on hazard_score before log (default: 1.0)
            sounding_hazard_weight: relative weight of sounding vs depth risk (default: 0.5)
            step_band_bonus_strength: strength in bonus = OWB × (1 - preference × strength) (default: 0.85)
        """
        self.classifier = classifier
        self.BLOCKING_THRESHOLD = blocking_threshold
        self.DEFAULT_MAX_PENALTY = max_penalty
        self.MIN_BONUS_FACTOR = min_bonus_factor
        self.OPEN_WATER_BASE_MULTIPLIER = open_water_base_multiplier
        self.UKC_RESTRICTED_PENALTY   = ukc_restricted_penalty
        self.UKC_SHALLOW_PENALTY      = ukc_shallow_penalty
        self.UKC_SAFE_PENALTY         = ukc_safe_penalty
        self.DEEP_WATER_BONUS         = deep_water_bonus
        self.SOUNDING_HIGH_RISK       = sounding_high_risk
        self.SOUNDING_MODERATE_RISK   = sounding_moderate_risk
        self.CLEARANCE_RESTRICTED_PENALTY = clearance_restricted_penalty
        self.ANCHORAGE_BONUS          = anchorage_bonus
        # Smooth mode parameters
        self.smooth_mode              = smooth_mode
        self.bonus_decay_rate         = bonus_decay_rate
        self.penalty_hazard_scale     = penalty_hazard_scale
        self.sounding_hazard_weight   = sounding_hazard_weight
        # Step-band bonus strength
        self.step_band_bonus_strength = step_band_bonus_strength

    @staticmethod
    def _extract_vessel_params(vessel_params: dict) -> tuple:
        """Extract standard vessel parameters from a vessel_params dict.

        Returns:
            Tuple of (draft, vessel_height, safety_margin, clearance_safety, vessel_type)
        """
        draft = vessel_params.get('draft', 5.0)
        vessel_height = vessel_params.get('height', 25.0)
        safety_margin = vessel_params.get('ukc_safety_margin', 2.0)
        clearance_safety = vessel_params.get('ver_clearance_margin', 3.0)
        vessel_type = vessel_params.get('vessel_type', 'cargo')
        return draft, vessel_height, safety_margin, clearance_safety, vessel_type

    @staticmethod
    def validate_vessel_params(
        vessel_params: Dict[str, Any],
        default_vessel: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract and validate vessel parameters, filling defaults from *default_vessel*.

        Returns:
            Dict with keys: vessel_type, draft, vessel_height, base_safety_margin,
            clearance_safety, compliance_zone.

        Raises:
            ValueError: If any value is out of range.
        """
        vessel_type = vessel_params.get('vessel_type', default_vessel['vessel_type'])
        draft = vessel_params.get('draft', default_vessel['draft'])
        vessel_height = vessel_params.get('height', default_vessel['height'])
        base_safety_margin = vessel_params.get('ukc_safety_margin', default_vessel['ukc_safety_margin'])
        clearance_safety = vessel_params.get('ver_clearance_margin', default_vessel['ver_clearance_margin'])

        if draft <= 0:
            raise ValueError(f"Draft must be positive, got {draft}")
        if vessel_height <= 0:
            raise ValueError(f"Vessel height must be positive, got {vessel_height}")
        if base_safety_margin < 0:
            raise ValueError(f"Safety margin must be non-negative, got {base_safety_margin}")
        if base_safety_margin >= 0.5 * draft:
            logging.getLogger(__name__).warning(
                "UKC Band 2 (Shallow) is unreachable: ukc_safety_margin (%.2fm) >= "
                "0.5 x draft (%.2fm). Edges with UKC in (%.2f, %.2f]m will be classified "
                "as Band 3 (Restricted) instead. Behavior is conservative, not an error.",
                base_safety_margin, 0.5 * draft, 0.5 * draft, base_safety_margin,
            )
        if clearance_safety < 0:
            raise ValueError(f"Clearance safety margin must be non-negative, got {clearance_safety}")

        # compliance_zone: per-zone penalty multipliers (≥ 1.0) aligned with distances_nm.
        # None → fall back to pre-computed wt_zone_penalty defaults from config.
        # Single float → broadcast to all zones.
        raw_cz = vessel_params.get('compliance_zone', default_vessel.get('compliance_zone', None))
        if raw_cz is not None:
            if isinstance(raw_cz, (int, float)):
                raw_cz = [float(raw_cz)] * 3  # broadcast scalar; length reconciled at Weights level
            compliance_zone: Optional[List[float]] = [max(1.0, float(v)) for v in raw_cz]
        else:
            compliance_zone = None  # signal: use wt_zone_penalty column (full-compliance defaults)

        return {
            'vessel_type': vessel_type,
            'draft': draft,
            'vessel_height': vessel_height,
            'base_safety_margin': base_safety_margin,
            'clearance_safety': clearance_safety,
            'compliance_zone': compliance_zone,
        }

    @staticmethod
    def validate_env_conditions(
        environmental_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract and validate environmental conditions.

        Returns:
            Dict with keys: weather_factor, visibility_factor, time_of_day.

        Raises:
            ValueError: If any value is out of range.
        """
        if environmental_conditions is None:
            environmental_conditions = {}

        weather_factor = environmental_conditions.get('weather_factor', 1.0)
        visibility_factor = environmental_conditions.get('visibility_factor', 1.0)
        time_of_day = environmental_conditions.get('time_of_day', 'day')

        if weather_factor < 0:
            raise ValueError(f"Weather factor must be non-negative, got {weather_factor}")
        if visibility_factor < 0:
            raise ValueError(f"Visibility factor must be non-negative, got {visibility_factor}")
        if time_of_day not in ('day', 'night'):
            raise ValueError(f"Time of day must be 'day' or 'night', got '{time_of_day}'")

        return {
            'weather_factor': weather_factor,
            'visibility_factor': visibility_factor,
            'time_of_day': time_of_day,
        }

    def encode_depth_bands(self, depth: float, draft: float, safety_margin: float) -> float:
        """
        Encode depth value into 5-band penalty system using UKC (Under Keel Clearance).

        UKC = Water Depth - Vessel Draft

        Band 4 (Grounding):    UKC <= 0                    -> BLOCKING_THRESHOLD (999.0)
        Band 3 (Restricted):   0 < UKC <= safety_margin    -> 10.0
        Band 2 (Shallow):      safety_margin < UKC <= 0.5*draft -> 2.0
        Band 1 (Safe):         0.5*draft < UKC <= draft        -> 1.5
        Band 0 (Deep):         UKC > draft                 -> 1.0

        Args:
            depth: Water depth in meters (from drval1, valsou, etc.)
            draft: Vessel draft in meters
            safety_margin: Safety buffer in meters (additional clearance required)

        Returns:
            float: Penalty factor for pathfinding
        """
        # Calculate Under Keel Clearance (UKC)
        ukc = depth - draft

        if ukc <= 0:
            # Band 4: Grounding - no clearance, impassable
            return self.BLOCKING_THRESHOLD
        elif ukc <= safety_margin:
            # Band 3: Restricted - clearance less than safety margin
            return self.UKC_RESTRICTED_PENALTY
        elif ukc <= 0.5 * draft:
            # Band 2: Shallow - limited UKC
            return self.UKC_SHALLOW_PENALTY
        elif ukc <= draft:
            # Band 1: Transitional - adequate UKC but not deep
            return self.UKC_SAFE_PENALTY
        else:
            # Band 0: Deep - excellent clearance (UKC > draft)
            return 1.0

    def encode_ver_clearance_meters(
        self,
        clearance: float,
        vessel_height: float,
        ver_clearance_margin: float
    ) -> float:
        """
        Encode vertical clearance using precise meter-based thresholds.

        For bridges, cables, and overhead pipelines.

        Args:
            clearance: Vertical clearance in meters (from verclr attribute)
            vessel_height: Maximum vessel height in meters (mast/antenna)
            ver_clearance_margin: Vertical safety buffer in meters

        Returns:
            float: Penalty factor (inf if impassable, DEFAULT_MAX_PENALTY if restricted, 1.0 if safe)
        """
        required_clearance = vessel_height + ver_clearance_margin

        if clearance < vessel_height:
            # Impassable - clearance less than vessel height
            return float('inf')
        elif clearance < required_clearance:
            # Restricted - clearance less than required (vessel + safety)
            return self.DEFAULT_MAX_PENALTY
        else:
            # Safe clearance
            return 1.0

    def calculate_bearing(self, point1: Tuple[float, float],
                       point2: Tuple[float, float]) -> float:
        """
        Calculate forward bearing (azimuth) from point1 to point2 in degrees.

        Uses forward azimuth formula for geodetic coordinates:
        bearing = atan2(sin(Δλ)·cos(φ2), cos(φ1)·sin(φ2) − sin(φ1)·cos(φ2)·cos(Δλ))

        Args:
            point1: (lon, lat) starting point in decimal degrees
            point2: (lon, lat) ending point in decimal degrees

        Returns:
            float: Bearing in degrees (0-360, where 0=North, 90=East)
        """
        return Bearing.bearing_scalar(point1, point2)

    def calculate_angular_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate absolute angular difference between two bearings.

        Handles 360° wrap-around correctly (e.g., difference between 350° and 10° is 20°, not 340°).

        Args:
            angle1: First angle in degrees (0-360)
            angle2: Second angle in degrees (0-360)

        Returns:
            float: Absolute angular difference in degrees (0-180)
        """
        return Bearing.angular_difference_scalar(angle1, angle2)

    def calculate_directional_factor_from_bands(self, dir_diff: float,
                                             angle_bands: List[Dict[str, Any]]) -> float:
        """
        Calculate directional weight factor based on angular difference using configured bands.

        Evaluates angle bands in order (sorted by max_angle) and returns weight
        factor from the first band where dir_diff <= max_angle.

        Args:
            dir_diff: Angular difference in degrees (0-180)
            angle_bands: List of band configurations, each with:
                - max_angle (float): Maximum angle for this band
                - weight (float): Weight factor to apply
                - description (str): Human-readable description

        Returns:
            float: Directional weight factor from matching band (default: 1.0 if no match)
        """
        for band in angle_bands:
            if dir_diff <= band['max_angle']:
                return band['weight']

        # Fallback if no band matches (should not happen if bands are properly configured)
        logger.warning(f"No angle band matched for dir_diff={dir_diff}°, using neutral weight 1.0")
        return 1.0

    def calculate_dynamic_safety_margin(self, base_safety_margin: float,
                                        weather_factor: float = 1.0,
                                        visibility_factor: float = 1.0,
                                        time_of_day: str = 'day') -> float:
        """
        Calculate dynamic safety margin based on environmental conditions.

        Safety margin increases in poor conditions:
        - Poor weather (storms, high seas)
        - Low visibility (fog, rain)
        - Night time navigation

        The formula applies multiplicative factors:
            dynamic_margin = base x weather_factor x visibility_factor x night_factor

        Args:
            base_safety_margin: Base safety margin in meters
            weather_factor: Weather multiplier (1.0 = good, 1.5 = moderate, 2.0 = poor)
            visibility_factor: Visibility multiplier (1.0 = good, 1.5 = reduced, 2.0 = poor)
            time_of_day: 'day' or 'night'

        Returns:
            float: Adjusted safety margin in meters
        """
        dynamic_margin = base_safety_margin
        dynamic_margin *= weather_factor
        dynamic_margin *= visibility_factor

        if time_of_day == 'night':
            dynamic_margin *= 1.2  # 20% increase for night navigation

        logger.debug(
            f"Dynamic safety margin: {base_safety_margin:.2f}m -> {dynamic_margin:.2f}m "
            f"(weather={weather_factor}, visibility={visibility_factor}, time={time_of_day})"
        )

        return dynamic_margin

    def calculate_tier_degradation(
        self,
        nav_class,
        base_factor: float,
        distance_meters: float = 0.0,
        buffer_meters: float = 0.0,
    ) -> Tuple[str, float]:
        """
        Determine tier and weight for a spatially matched edge.

        Simplified single-band logic: if an edge is within the feature's
        spatial reach (intersection or buffer), it receives the tier's weight.
        Buffer only extends the radius of influence — no graduated bands.

        Tier mapping:
        - DANGEROUS → blocking (base_factor)
        - CAUTION   → penalty  (base_factor)
        - SAFE      → bonus    (base_factor)

        Args:
            nav_class: NavClass enum value (DANGEROUS, CAUTION, SAFE)
            base_factor: Base risk multiplier from classifier
            distance_meters: Distance from edge to feature (kept for API compat)
            buffer_meters: Buffer distance for the feature (kept for API compat)

        Returns:
            Tuple[str, float]: (tier_name, weight_factor)
        """
        if nav_class == NavClass.DANGEROUS:
            return ('blocking', base_factor)
        elif nav_class == NavClass.CAUTION:
            return ('penalty', base_factor)
        elif nav_class == NavClass.SAFE:
            return ('bonus', base_factor)
        else:
            return ('penalty', 1.0)

    def apply_static_weights_vectorized(
        self,
        edges_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        layer_name: str,
        classification: Dict[str, Any],
        chunk_size: Optional[int] = None,
        buffer_method: str = 'auto',
        aggr_mode: str = 'max',
    ) -> gpd.GeoDataFrame:
        """
        Vectorized static weight computation for a single S-57 layer.

        Replaces the row-by-row ``_apply_static_weights_logic()`` with a fully
        vectorized pipeline using GeoPandas sjoin + shapely 2.0 array operations
        + pandas groupby aggregation.

        Pipeline
        --------
        1. ``gpd.sjoin(edges, features, predicate="intersects")`` — spatial join
           (builds STRtree on features automatically).
        2. Recover feature geometries from ``index_right`` (sjoin drops them).
        3. ``shapely.distance(edge_geoms, feature_geoms)`` — vectorized distances.
        4. Latitude-adjusted degree→meter conversion (documented approximation).
        5. ``np.select()`` tier assignment — band classification.
        6. pandas groupby + agg — MAX/PROD/MIN aggregations per edge_id.
        7. Merge aggregated results back into edges_gdf via ``loc``.

        Distance approximation
        ----------------------
        ``meters = degrees * 111320 * cos(lat)``

        This is an approximation (~0.3 % error at 60°N). For static weights
        (binary buffer zones) this is fully acceptable. For high-precision use:
        ``joined.to_crs(joined.estimate_utm_crs()).distance(...)``

        Args:
            edges_gdf: Edges GeoDataFrame. Must contain an ``edge_id`` column
                (integer index) and ``wt_static_blocking``, ``wt_static_penalty``,
                ``wt_static_bonus`` columns (initialised by the caller).
            features_gdf: S-57 layer features GeoDataFrame.
            layer_name: S-57 layer acronym (used only for logging).
            classification: Layer classification dict from S57Classifier with keys
                ``nav_class``, ``risk_multiplier``, ``buffer_meters``.
            chunk_size: If set, process ``edges_gdf`` in batches of this many rows
                to limit peak memory usage. ``None`` (default) processes all at once.

        Returns:
            edges_gdf with updated ``wt_static_blocking``, ``wt_static_penalty``,
            and ``wt_static_bonus`` columns.
        """
        nav_class = classification['nav_class']
        base_factor = classification['risk_multiplier']
        buffer_meters = classification['buffer_meters']

        # Skip layers with no practical effect
        if base_factor == 1.0 and nav_class == NavClass.INFORMATIONAL:
            return edges_gdf

        if features_gdf is None or len(features_gdf) == 0:
            return edges_gdf

        # Defensive: ensure contiguous RangeIndex for correct .loc recovery
        if not isinstance(features_gdf.index, pd.RangeIndex):
            features_gdf = features_gdf.reset_index(drop=True)

        # Pre-buffer features when buffer_meters > 0 to match SQL/PostGIS DWithin semantics.
        # 'auto' selects 'fine' for Point/Area (prim≠2) and 'fast' for Line-only (prim=2).
        # 'fine'  → UTM-reprojected geodesically-accurate buffer; no post-filter needed.
        # 'fast'  → per-feature lat-corrected degree buffer; post-filter applied afterwards.
        if buffer_meters > 0:
            effective = Buffer.resolve_method(buffer_method, features_gdf)
            logger.debug(f"    {layer_name}: buffer_method={buffer_method} → effective={effective}")
            if effective == 'fine':
                buf_gdf = Buffer.apply_buffer_fine_gdf(features_gdf, buffer_meters)
            else:
                buf_gdf = Buffer.apply_buffer_fast_gdf(features_gdf, buffer_meters)
            features_buffered = features_gdf.copy()
            features_buffered['geometry'] = buf_gdf.geometry.values
        else:
            features_buffered = features_gdf
            effective = 'fast'  # sentinel — not used when buffer_meters == 0

        total_sjoin_matches = 0

        def _process_chunk(chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            """Apply vectorized static weights for a single chunk of edges."""
            nonlocal total_sjoin_matches
            # ── 1. Spatial join ───────────────────────────────────────────────
            # Rename the index if it conflicts with an existing column name,
            # because gpd.sjoin calls reset_index() internally which would fail.
            if chunk.index.name is not None and chunk.index.name in chunk.columns:
                chunk = chunk.rename_axis(f"_idx_{chunk.index.name}")
            joined = gpd.sjoin(chunk, features_buffered, how='inner', predicate='intersects')
            if joined.empty:
                return chunk

            logger.debug(f"    {layer_name}: sjoin candidates={len(joined)}, distinct_edges={joined.index.nunique()}")

            # ── 2. Recover feature geometries (sjoin drops right-side geom) ──
            feat_geoms = features_gdf.geometry.loc[joined['index_right'].values].values
            edge_geoms = joined.geometry.values

            # ── 3. Vectorized distance in degrees ────────────────────────────
            deg_distances = shapely.distance(edge_geoms, feat_geoms)

            # ── 3a. Distance post-filter (fast mode only) ────────────────────
            # For 'fine' mode the UTM-accurate buffer IS the spatial constraint;
            # any edge that intersects the buffered geometry is genuinely within
            # buffer_meters, so no additional check is needed.
            # For 'fast' mode the sjoin buffer uses per-feature lat correction but
            # Shapely's polygon approximation may include slight overshoots; the
            # post-filter provides the exact per-feature lat-corrected check.
            if buffer_meters > 0 and effective == 'fast':
                feat_centroids = shapely.centroid(feat_geoms)
                feat_lats = shapely.get_y(feat_centroids)
                cos_lats = np.maximum(np.cos(np.radians(feat_lats)), 0.5)
                per_feat_buffer_deg = buffer_meters / (111320.0 * cos_lats)

                within_mask = deg_distances <= per_feat_buffer_deg
                logger.debug(
                    f"    {layer_name}: post-filter kept={within_mask.sum()}/{len(within_mask)}"
                    f" | buffer_deg min={per_feat_buffer_deg.min():.8f}"
                    f" max={per_feat_buffer_deg.max():.8f}"
                    f" | dist min={deg_distances.min():.8f}"
                    f" max={deg_distances.max():.8f}"
                )
                if not within_mask.any():
                    return chunk
                joined = joined[within_mask]
                feat_geoms = feat_geoms[within_mask]
                edge_geoms = edge_geoms[within_mask]
                deg_distances = deg_distances[within_mask]

            total_sjoin_matches += joined.index.nunique()

            # ── 4. Tier assignment (single-band: spatial match → tier weight) ──
            n = len(deg_distances)
            tiers = np.empty(n, dtype=object)
            weights = np.empty(n, dtype=float)

            if nav_class == NavClass.DANGEROUS:
                tiers[:] = 'blocking'
                weights[:] = base_factor
            elif nav_class == NavClass.CAUTION:
                tiers[:] = 'penalty'
                weights[:] = base_factor
            elif nav_class == NavClass.SAFE:
                tiers[:] = 'bonus'
                weights[:] = base_factor
            else:
                tiers[:] = 'penalty'
                weights[:] = 1.0

            # id is the named index after apply_static_weights_gdf sets
            # edges_gdf.index.name = 'id'; fall back to 'edge_id' column for
            # legacy callers that still provide it as a column.
            if joined.index.name in ('id', 'edge_id') or 'edge_id' not in joined.columns:
                edge_ids = joined.index.values
            else:
                edge_ids = joined['edge_id'].values

            # ── 6. Groupby aggregation per edge_id ────────────────────────────
            result_df = pd.DataFrame({
                'edge_id': edge_ids,
                'tier': tiers,
                'weight': weights,
            })

            blocking_df = result_df[result_df['tier'] == 'blocking']
            penalty_df = result_df[result_df['tier'] == 'penalty']
            bonus_df = result_df[result_df['tier'] == 'bonus']

            blocking_agg = blocking_df.groupby('edge_id')['weight'].max() if not blocking_df.empty else pd.Series(dtype=float)
            if aggr_mode == 'exp':
                penalty_agg = penalty_df.groupby('edge_id')['weight'].max() if not penalty_df.empty else pd.Series(dtype=float)
            else:
                penalty_agg = penalty_df.groupby('edge_id')['weight'].max() if not penalty_df.empty else pd.Series(dtype=float)
            bonus_agg = bonus_df.groupby('edge_id')['weight'].max() if not bonus_df.empty else pd.Series(dtype=float)

            blocking_cnt = blocking_df.groupby('edge_id')['weight'].size() if not blocking_df.empty else pd.Series(dtype=int)
            penalty_cnt  = penalty_df.groupby('edge_id')['weight'].size() if not penalty_df.empty else pd.Series(dtype=int)
            bonus_cnt    = bonus_df.groupby('edge_id')['weight'].size() if not bonus_df.empty else pd.Series(dtype=int)

            # ── 7. Merge back via loc (index alignment on edge_id) ────────────
            if not blocking_agg.empty:
                idx = blocking_agg.index
                chunk.loc[idx, 'wt_static_blocking'] = np.maximum(
                    chunk.loc[idx, 'wt_static_blocking'].values,
                    blocking_agg.values
                )

            if not penalty_agg.empty:
                idx = penalty_agg.index
                if aggr_mode == 'exp':
                    chunk.loc[idx, 'wt_static_penalty'] = (
                        chunk.loc[idx, 'wt_static_penalty'].values * penalty_agg.values
                    )
                else:
                    chunk.loc[idx, 'wt_static_penalty'] = np.maximum(
                        chunk.loc[idx, 'wt_static_penalty'].values,
                        penalty_agg.values
                    )

            if not bonus_agg.empty:
                idx = bonus_agg.index
                chunk.loc[idx, 'wt_static_bonus'] = np.maximum(
                    chunk.loc[idx, 'wt_static_bonus'].values,
                    bonus_agg.values
                )

            # ── 8. Update wt_static_sources JSON for affected edges ────────
            if 'wt_static_sources' in chunk.columns:
                def _update_src(current_json, tier_key, layer, value, count):
                    d = json.loads(current_json) if current_json and current_json != '{}' else {}
                    d.setdefault(tier_key, {})[layer] = [round(float(value), 4), int(count)]
                    return json.dumps(d, separators=(',', ':'))

                if not blocking_agg.empty:
                    chunk.loc[blocking_agg.index, 'wt_static_sources'] = [
                        _update_src(chunk.loc[i, 'wt_static_sources'], 'static_blocking', layer_name, v, blocking_cnt[i])
                        for i, v in blocking_agg.items()
                    ]
                if not penalty_agg.empty:
                    chunk.loc[penalty_agg.index, 'wt_static_sources'] = [
                        _update_src(chunk.loc[i, 'wt_static_sources'], 'static_penalty', layer_name, v, penalty_cnt[i])
                        for i, v in penalty_agg.items()
                    ]
                if not bonus_agg.empty:
                    chunk.loc[bonus_agg.index, 'wt_static_sources'] = [
                        _update_src(chunk.loc[i, 'wt_static_sources'], 'static_bonus', layer_name, v, bonus_cnt[i])
                        for i, v in bonus_agg.items()
                    ]

            return chunk

        # ── Chunked or single-pass processing ────────────────────────────────
        if chunk_size is not None:
            n_rows = len(edges_gdf)
            chunks = [edges_gdf.iloc[i:i + chunk_size].copy() for i in range(0, n_rows, chunk_size)]
            processed = pd.concat([_process_chunk(c) for c in chunks])
            # Restore original dtypes and index
            edges_gdf = processed
        else:
            edges_gdf = _process_chunk(edges_gdf.copy())

        return edges_gdf, total_sjoin_matches

    def compute_dynamic_edge(
        self,
        data: Dict[str, Any],
        draft: float,
        vessel_height: float,
        safety_margin: float,
        ver_clearance_margin: float,
        vessel_type: str
    ) -> Dict[str, Any]:
        """
        Compute dynamic weight components for a single edge.

        Returns dict with wt_dynamic_* values and source dicts for blocking/penalty/bonus.

        Args:
            data: Edge attributes dict (must include ft_depth, ft_ver_clearance,
                 ft_sounding, ft_anchorage_category)
            draft: Vessel draft in meters
            vessel_height: Vessel height in meters
            safety_margin: UKC safety margin in meters
            ver_clearance_margin: Vertical clearance safety margin in meters
            vessel_type: Vessel type ('cargo', 'passenger', or other)

        Returns:
            Dict with keys:
                - wt_dynamic_ukc_band, wt_dynamic_clearance, wt_dynamic_hazard,
                  wt_dynamic_deep_water, wt_dynamic_anchorage (float values)
                - dynamic_blocking, dynamic_penalty, dynamic_bonus (source dicts)
                - blocked, penalized, bonus (bool flags)
        """
        result = {
            'wt_dynamic_ukc_band': 1.0,
            'wt_dynamic_clearance': 1.0,
            'wt_dynamic_hazard': 1.0,
            'wt_dynamic_deep_water': 1.0,
            'wt_dynamic_anchorage': 1.0,
            'dynamic_blocking': {},
            'dynamic_penalty': {},
            'dynamic_bonus': {},
            'blocked': False,
            'penalized': False,
            'bonus': False,
        }

        # === TIER 1: DYNAMIC BLOCKING ===
        depth = data.get('ft_depth')
        if depth is not None:
            ukc = depth - draft
            if ukc <= 0:
                result['dynamic_blocking']['ukc_grounding'] = self.BLOCKING_THRESHOLD
                result['wt_dynamic_ukc_band'] = self.BLOCKING_THRESHOLD
                result['blocked'] = True
            else:
                # === TIER 2: UKC PENALTIES ===
                if ukc <= safety_margin:
                    result['dynamic_penalty']['ukc_restricted'] = self.UKC_RESTRICTED_PENALTY
                    result['wt_dynamic_ukc_band'] = self.UKC_RESTRICTED_PENALTY
                    result['penalized'] = True
                elif ukc <= 0.5 * draft:
                    result['dynamic_penalty']['ukc_shallow'] = self.UKC_SHALLOW_PENALTY
                    result['wt_dynamic_ukc_band'] = self.UKC_SHALLOW_PENALTY
                    result['penalized'] = True
                elif ukc <= draft:
                    result['dynamic_penalty']['ukc_safe'] = self.UKC_SAFE_PENALTY
                    result['wt_dynamic_ukc_band'] = self.UKC_SAFE_PENALTY
                    result['penalized'] = True
                elif ukc > draft:
                    # === TIER 3: Deep water bonus ===
                    result['dynamic_bonus']['deep_water'] = self.DEEP_WATER_BONUS
                    result['wt_dynamic_deep_water'] = self.DEEP_WATER_BONUS
                    result['bonus'] = True

        # Clearance penalties
        clearance = data.get('ft_ver_clearance')
        if clearance is not None and clearance >= vessel_height:
            if clearance < vessel_height + ver_clearance_margin:
                result['dynamic_penalty']['clearance_restricted'] = self.CLEARANCE_RESTRICTED_PENALTY
                result['wt_dynamic_clearance'] = self.CLEARANCE_RESTRICTED_PENALTY
                result['penalized'] = True

        # Sounding hazard penalties
        sounding = data.get('ft_sounding')
        if sounding is not None:
            sounding_ukc = sounding - draft
            if sounding_ukc > 0:
                if sounding_ukc <= safety_margin:
                    result['dynamic_penalty']['hazard_high_risk'] = self.SOUNDING_HIGH_RISK
                    result['wt_dynamic_hazard'] = self.SOUNDING_HIGH_RISK
                    result['penalized'] = True
                elif sounding_ukc <= draft:
                    result['dynamic_penalty']['hazard_moderate_risk'] = self.SOUNDING_MODERATE_RISK
                    result['wt_dynamic_hazard'] = self.SOUNDING_MODERATE_RISK
                    result['penalized'] = True

        # Anchorage category bonus
        catach = data.get('ft_anchorage_category')
        if catach:
            preferred = [1, 2] if vessel_type == 'cargo' else [5, 6] if vessel_type == 'passenger' else []
            if any(int(c) in preferred for c in catach if c is not None):
                result['dynamic_bonus']['anchorage'] = self.ANCHORAGE_BONUS
                result['wt_dynamic_anchorage'] = self.ANCHORAGE_BONUS
                result['bonus'] = True

        return result

    def calculate_blocking_factor(self, edge_data: Dict[str, Any],
                               vessel_params: Dict[str, Any]) -> float:
        """
        Calculate Tier 1: Absolute blocking constraints.

        Combines static blocking (from apply_static_weights) with dynamic blocking
        constraints (UKC ≤ 0 grounding risk).

        Args:
            edge_data: Edge attributes dictionary
            vessel_params: Vessel parameters (draft, height, etc.)

        Returns:
            float: Blocking factor (1.0 = passable, BLOCKING_THRESHOLD = effectively impassable)
        """
        draft, _, _, _, _ = self._extract_vessel_params(vessel_params)

        blocking_factor = 1.0

        # STATIC BLOCKING: From apply_static_weights()
        static_blocking = edge_data.get('wt_static_blocking', 1.0)
        blocking_factor = max(blocking_factor, static_blocking)

        # DYNAMIC BLOCKING: UKC grounding risk (vessel-specific)
        depth = edge_data.get('ft_depth')
        if depth is not None:
            ukc = depth - draft
            if ukc <= 0:
                # Grounding risk - absolute blocker
                blocking_factor = max(blocking_factor, self.BLOCKING_THRESHOLD)

        return blocking_factor

    def calculate_penalty_factor(self, edge_data: Dict[str, Any],
                                vessel_params: Dict[str, Any],
                                max_penalty: float = 50.0) -> float:
        """
        Calculate Tier 2: Conditional penalties (cumulative hazards).

        Combines static penalties with dynamic penalties (vessel-specific constraints).

        Args:
            edge_data: Edge attributes dictionary
            vessel_params: Vessel parameters
            max_penalty: Maximum cumulative penalty

        Returns:
            float: Penalty factor (1.0 = no penalty, up to max_penalty)
        """
        if self.smooth_mode:
            return self._calculate_penalty_factor_smooth(edge_data, vessel_params)

        draft, vessel_height, safety_margin, clearance_safety, _ = self._extract_vessel_params(vessel_params)

        penalty_factor = 1.0

        # STATIC PENALTIES: From apply_static_weights()
        static_penalty = edge_data.get('wt_static_penalty', 1.0)
        penalty_factor *= static_penalty

        # DYNAMIC PENALTIES: Vessel-specific constraints

        # === DEPTH PENALTIES (UKC-based) ===
        depth = edge_data.get('ft_depth')
        if depth is not None:
            ukc = depth - draft

            if ukc > 0:  # Not blocking (blocking handled in Tier 1)
                if ukc <= safety_margin:
                    # Restricted: very shallow but passable
                    penalty_factor *= self.UKC_RESTRICTED_PENALTY
                elif ukc <= 0.5 * draft:
                    # Safe: adequate clearance
                    penalty_factor *= self.UKC_SHALLOW_PENALTY
                elif ukc <= draft:
                    # Transitional: good clearance
                    penalty_factor *= self.UKC_SAFE_PENALTY

        # === CLEARANCE PENALTIES ===
        clearance = edge_data.get('ft_ver_clearance')
        if clearance is not None:
            if clearance >= vessel_height:  # Not blocking
                if clearance < vessel_height + clearance_safety:
                    # Restricted clearance
                    penalty_factor *= self.CLEARANCE_RESTRICTED_PENALTY

        # === HAZARD ACCUMULATION ===
        sounding = edge_data.get('ft_sounding')
        if sounding is not None:
            sounding_ukc = sounding - draft
            if sounding_ukc > 0:  # Passable but hazardous
                if sounding_ukc <= safety_margin:
                    # High risk: hazard just above draft
                    penalty_factor *= self.SOUNDING_HIGH_RISK
                elif sounding_ukc <= draft:
                    # Moderate risk: hazard with some clearance
                    penalty_factor *= self.SOUNDING_MODERATE_RISK

        # CAP ACCUMULATION - prevent explosion
        penalty_factor = min(penalty_factor, max_penalty)

        return penalty_factor

    def calculate_bonus_factor(self, edge_data: Dict[str, Any],
                              vessel_params: Dict[str, Any]) -> float:
        """
        Calculate Tier 3: Preference bonuses (safe routes).

        Combines static bonuses with dynamic bonuses (vessel-specific preferences).

        Args:
            edge_data: Edge attributes dictionary
            vessel_params: Vessel parameters

        Returns:
            float: Bonus factor (MIN_BONUS_FACTOR–OPEN_WATER_BASE_MULTIPLIER, where lower = preferred route)
        """
        if self.smooth_mode:
            return self._calculate_bonus_factor_smooth(edge_data, vessel_params)

        draft, _, _, _, vessel_type = self._extract_vessel_params(vessel_params)

        # STATIC BONUSES: preference_intensity ∈ [0.0, 1.0] (1.0 = max preference)
        preference = max(0.0, min(edge_data.get('wt_static_bonus', 0.0), 1.0))
        bonus_factor = self.OPEN_WATER_BASE_MULTIPLIER * (1.0 - preference * self.step_band_bonus_strength)

        # DYNAMIC BONUSES: Vessel-specific preferences — divide to reduce cost for preferred conditions

        # === DEEP WATER BONUS ===
        depth = edge_data.get('ft_depth')
        if depth is not None:
            ukc = depth - draft
            if ukc > draft:
                # Excellent clearance (UKC > draft) — divide to lower cost
                bonus_factor /= self.DEEP_WATER_BONUS

        # === ANCHORAGE CATEGORY BONUS (vessel type matching) ===
        catach = edge_data.get('ft_anchorage_category')
        if catach:
            preferred = [1, 2] if vessel_type == 'cargo' else [5, 6] if vessel_type == 'passenger' else []
            if any(int(c) in preferred for c in catach if c is not None):
                bonus_factor /= self.ANCHORAGE_BONUS

        # Ensure bonus doesn't go below floor
        bonus_factor = max(bonus_factor, self.MIN_BONUS_FACTOR)

        return bonus_factor

    # ------------------------------------------------------------------
    # Smooth weight methods (smooth_mode=True)
    # These expose pre-activation scores for GNN/PyTorch pipelines.
    # ------------------------------------------------------------------

    def _compute_preference_score(self, edge_data: Dict[str, Any],
                                  vessel_params: Dict[str, Any]) -> float:
        """
        Aggregate navigational preference for the bonus tier.

        Higher scores indicate more preferred routes (deep fairways, channels).
        Used as input to the exponential bonus formula in smooth mode.

        Returns:
            float: preference_score ∈ [0, ∞)
        """
        draft, _, _, _, _ = self._extract_vessel_params(vessel_params)

        # Static preference: preference_intensity ∈ [0.0, 1.0]
        # FAIRWY (wt_static_bonus=1.0) → static_pref=1.0
        # Plain open water (wt_static_bonus=0.0) → static_pref=0.0
        static_pref = max(0.0, min(edge_data.get('wt_static_bonus', 0.0), 1.0))

        # Deep water preference: reward UKC excess beyond one draft length
        depth_pref = 0.0
        depth = edge_data.get('ft_depth')
        if depth is not None and draft > 0:
            ukc = depth - draft
            if ukc > draft:
                depth_pref = (ukc - draft) / draft  # normalized excess

        # Directional preference: reward for traffic alignment (smooth mode only).
        # wt_dir=1.0 (aligned) → dir_pref=1.0 | wt_dir=2.0 (neutral) → 0.0 | >2.0 clamped to 0.
        # Penalty side (wt_dir > 2.0) is captured in _compute_hazard_score to avoid double-counting.
        wt_dir = edge_data.get('wt_dir', self.OPEN_WATER_BASE_MULTIPLIER)
        dir_pref = max(0.0, self.OPEN_WATER_BASE_MULTIPLIER - wt_dir)

        return max(0.0, static_pref + depth_pref + dir_pref)

    def _compute_hazard_score(self, edge_data: Dict[str, Any],
                              vessel_params: Dict[str, Any]) -> float:
        """
        Aggregate navigational risk for the penalty tier.

        Higher scores indicate more hazardous routes. The score feeds into
        log(1 + hazard_score) to produce a self-limiting penalty.
        UKC→0 causes ukc_risk→∞, creating a repulsive gradient before blocking.

        Returns:
            float: hazard_score ∈ [0, ∞)
        """
        draft, vessel_height, _, clearance_safety, _ = self._extract_vessel_params(vessel_params)
        eps = 1e-6
        hazard_score = 0.0

        # Static hazard: accumulated from apply_static_weights()
        static_penalty = edge_data.get('wt_static_penalty', 1.0)
        hazard_score += max(0.0, static_penalty - 1.0)

        # UKC risk: continuous gradient for 0 < ukc <= draft
        # Blocking (ukc <= 0) is handled separately in Tier 1
        depth = edge_data.get('ft_depth')
        if depth is not None:
            ukc = depth - draft
            if 0.0 < ukc <= draft:
                hazard_score += max(0.0, draft / max(ukc, eps) - 1.0)

        # Vertical clearance risk: vessel fits but margin is tight
        clearance = edge_data.get('ft_ver_clearance')
        if clearance is not None and vessel_height <= clearance < vessel_height + clearance_safety:
            hazard_score += clearance_safety / max(clearance - vessel_height, eps)

        # Sounding risk: weighted hazard from obstruction/wreck soundings
        sounding = edge_data.get('ft_sounding')
        if sounding is not None:
            sounding_ukc = sounding - draft
            if 0.0 < sounding_ukc <= draft:
                sounding_risk = self.sounding_hazard_weight * max(
                    0.0, draft / max(sounding_ukc, eps) - 1.0
                )
                hazard_score += sounding_risk

        # Directional hazard: penalty for going against traffic flow (smooth mode only).
        # wt_dir=2.0 (neutral) → 0  |  wt_dir=50 (crossing) → 48  |  wt_dir=200 (opposite) → 198.
        # Reward side (wt_dir < 2.0) is captured in _compute_preference_score.
        wt_dir = edge_data.get('wt_dir', self.OPEN_WATER_BASE_MULTIPLIER)
        hazard_score += max(0.0, wt_dir - self.OPEN_WATER_BASE_MULTIPLIER)

        return hazard_score

    def _calculate_bonus_factor_smooth(self, edge_data: Dict[str, Any],
                                       vessel_params: Dict[str, Any]) -> float:
        """
        Smooth Tier 3: bonus_factor = 1 + exp(-k * preference_score).

        Range: (1.0, 2.0]
          - preference_score=0 (open water)  → 1 + exp(0)  = 2.0
          - preference_score→∞ (deep fairway) → 1.0 asymptotically
        """
        pref_score = self._compute_preference_score(edge_data, vessel_params)
        return 1.0 + math.exp(-self.bonus_decay_rate * pref_score)

    def _calculate_penalty_factor_smooth(self, edge_data: Dict[str, Any],
                                         vessel_params: Dict[str, Any]) -> float:
        """
        Smooth Tier 2: penalty_factor = 1 + log(1 + hazard_score * penalty_hazard_scale).

        Range: [1.0, ∞)  — logarithmic growth is naturally self-limiting (no hard cap needed).
          - hazard_score=0 (deep, clear water) → 1 + log(1) = 1.0
          - hazard_score=9                     → ≈ 2.4
          - hazard_score=99                    → ≈ 5.6
        """
        hazard_score = self._compute_hazard_score(edge_data, vessel_params)
        return 1.0 + math.log(1.0 + hazard_score * self.penalty_hazard_scale)

    # ------------------------------------------------------------------
    # PostGIS SQL expression builders (smooth mode)
    # These produce parameterized SQL fragments that mirror the Python
    # smooth scoring methods above.
    # ------------------------------------------------------------------

    def _build_preference_score_sql_expr(self) -> str:
        """
        Return a parameterized SQL fragment computing the preference score.

        Mirrors _compute_preference_score().
        Parameters used in the fragment: :draft
        """
        return (
            "GREATEST(0.0,\n"
            "    LEAST(GREATEST(COALESCE(wt_static_bonus, 0.0), 0.0), 1.0)\n"
            "    + CASE\n"
            "        WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft\n"
            "        THEN (ft_depth - :draft - :draft) / NULLIF(:draft, 0)\n"
            "        ELSE 0.0\n"
            "      END\n"
            "    + GREATEST(0.0, 2.0 - COALESCE(wt_dir, 2.0))\n"
            ")"
        )

    def _build_hazard_score_sql_expr(self) -> str:
        """
        Return a parameterized SQL fragment computing the hazard score.

        Mirrors _compute_hazard_score().
        Parameters used in the fragment:
            :draft, :vessel_height, :clearance_safety, :sounding_hazard_weight
        """
        return (
            "GREATEST(0.0, COALESCE(wt_static_penalty, 1.0) - 1.0)\n"
            "+ CASE\n"
            "    WHEN ft_depth IS NOT NULL\n"
            "         AND (ft_depth - :draft) > 0\n"
            "         AND (ft_depth - :draft) <= :draft\n"
            "    THEN GREATEST(0.0, :draft / GREATEST(ft_depth - :draft, 1e-6) - 1.0)\n"
            "    ELSE 0.0\n"
            "  END\n"
            "+ CASE\n"
            "    WHEN ft_ver_clearance IS NOT NULL\n"
            "         AND ft_ver_clearance >= :vessel_height\n"
            "         AND ft_ver_clearance < :vessel_height + :clearance_safety\n"
            "    THEN :clearance_safety / GREATEST(ft_ver_clearance - :vessel_height, 1e-6)\n"
            "    ELSE 0.0\n"
            "  END\n"
            "+ CASE\n"
            "    WHEN ft_sounding IS NOT NULL\n"
            "         AND (ft_sounding - :draft) > 0\n"
            "         AND (ft_sounding - :draft) <= :draft\n"
            "    THEN :sounding_hazard_weight\n"
            "         * GREATEST(0.0, :draft / GREATEST(ft_sounding - :draft, 1e-6) - 1.0)\n"
            "    ELSE 0.0\n"
            "  END\n"
            "+ GREATEST(0.0, COALESCE(wt_dir, 2.0) - 2.0)"
        )

    def _build_preference_score_gpkg_expr(self) -> str:
        """
        Return a parameterized SQL fragment computing the preference score for SQLite/GeoPackage.

        Mirrors _build_preference_score_sql_expr() but uses MAX() instead of GREATEST()
        (SQLite has no GREATEST function).
        Parameters used in the fragment: :draft
        """
        return (
            "MAX(0.0,\n"
            "    MIN(MAX(COALESCE(wt_static_bonus, 0.0), 0.0), 1.0)\n"
            "    + CASE\n"
            "        WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft\n"
            "        THEN (ft_depth - :draft - :draft) / NULLIF(:draft, 0)\n"
            "        ELSE 0.0\n"
            "      END\n"
            "    + MAX(0.0, 2.0 - COALESCE(wt_dir, 2.0))\n"
            ")"
        )

    def _build_hazard_score_gpkg_expr(self) -> str:
        """
        Return a parameterized SQL fragment computing the hazard score for SQLite/GeoPackage.

        Mirrors _build_hazard_score_sql_expr() but uses MAX() instead of GREATEST()
        (SQLite has no GREATEST function).
        Parameters used in the fragment:
            :draft, :vessel_height, :clearance_safety, :sounding_hazard_weight
        """
        return (
            "MAX(0.0, COALESCE(wt_static_penalty, 1.0) - 1.0)\n"
            "+ CASE\n"
            "    WHEN ft_depth IS NOT NULL\n"
            "         AND (ft_depth - :draft) > 0\n"
            "         AND (ft_depth - :draft) <= :draft\n"
            "    THEN MAX(0.0, :draft / MAX(ft_depth - :draft, 1e-6) - 1.0)\n"
            "    ELSE 0.0\n"
            "  END\n"
            "+ CASE\n"
            "    WHEN ft_ver_clearance IS NOT NULL\n"
            "         AND ft_ver_clearance >= :vessel_height\n"
            "         AND ft_ver_clearance < :vessel_height + :clearance_safety\n"
            "    THEN :clearance_safety / MAX(ft_ver_clearance - :vessel_height, 1e-6)\n"
            "    ELSE 0.0\n"
            "  END\n"
            "+ CASE\n"
            "    WHEN ft_sounding IS NOT NULL\n"
            "         AND (ft_sounding - :draft) > 0\n"
            "         AND (ft_sounding - :draft) <= :draft\n"
            "    THEN :sounding_hazard_weight\n"
            "         * MAX(0.0, :draft / MAX(ft_sounding - :draft, 1e-6) - 1.0)\n"
            "    ELSE 0.0\n"
            "  END\n"
            "+ MAX(0.0, COALESCE(wt_dir, 2.0) - 2.0)"
        )

    def _calculate_smooth_weights_sql(
        self,
        conn,
        table_name: str,
        vessel_params: Dict[str, Any],
        store_scores: bool = True,
        max_penalty: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Execute smooth-mode weight computation against a SQLite/SpatiaLite table.

        Mirrors _calculate_smooth_weights_postgis() but uses sqlite3 cursor
        and SQLite-compatible SQL (MAX instead of GREATEST, ln() instead of LN()).

        Args:
            conn: Active sqlite3 connection (must support commit())
            table_name: Bare table name, e.g. 'edges'
            vessel_params: Dict with at least 'draft'; optionally 'height',
                           'ver_clearance_margin' / 'clearance_safety_margin' /
                           'clearance_safety'
            store_scores: When True, persist preference_score and hazard_score
                          columns alongside the factor columns (default True)

        Returns:
            Dict with keys:
                - 'updated_rows': total row count of the table
                - 'store_scores': the store_scores argument used
        """
        cursor = conn.cursor()

        draft, vessel_height, _, clearance_safety, _ = self._extract_vessel_params(vessel_params)

        # Step A — Optional score columns (via pragma_table_info, matching existing gpkg pattern)
        if store_scores:
            existing = {row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")}
            for col in ('preference_score', 'hazard_score'):
                if col not in existing:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} REAL")
            conn.commit()

        # Step B — Single inline UPDATE (no CTE; pref/hazard expressions inlined)
        pref_expr = self._build_preference_score_gpkg_expr()
        hazard_expr = self._build_hazard_score_gpkg_expr()

        if store_scores:
            score_cols = (
                ",\n"
                f"    preference_score = {pref_expr},\n"
                f"    hazard_score     = {hazard_expr}"
            )
        else:
            score_cols = ""

        update_sql = (
            f"UPDATE {table_name}\n"
            f"SET\n"
            f"    bonus_factor   = MAX(1.0 + EXP(-:bonus_decay_rate * ({pref_expr})), :min_bonus_factor),\n"
            f"    penalty_factor = 1.0 + ln(1.0 + ({hazard_expr}) * :penalty_hazard_scale),\n"
            f"    ukc_meters     = CASE WHEN ft_depth IS NOT NULL THEN ft_depth - :draft ELSE NULL END"
            f"{score_cols}"
        )

        cursor.execute(update_sql, {
            'draft': draft,
            'vessel_height': vessel_height,
            'clearance_safety': clearance_safety,
            'sounding_hazard_weight': self.sounding_hazard_weight,
            'bonus_decay_rate': self.bonus_decay_rate,
            'min_bonus_factor': self.MIN_BONUS_FACTOR,
            'penalty_hazard_scale': self.penalty_hazard_scale,
        })
        conn.commit()

        # Step B.2 — Penalty cap (safety guard)
        cursor.execute(
            f"UPDATE {table_name}\n"
            f"SET penalty_factor = MIN(penalty_factor, :max_penalty)\n"
            f"WHERE penalty_factor > :max_penalty",
            {'max_penalty': max_penalty},
        )
        conn.commit()

        # Step B.3 — Bonus floor (safety guard, mirrors formula MAX)
        cursor.execute(
            f"UPDATE {table_name}\n"
            f"SET bonus_factor = MAX(bonus_factor, :min_bonus_factor)\n"
            f"WHERE bonus_factor < :min_bonus_factor",
            {'min_bonus_factor': self.MIN_BONUS_FACTOR},
        )
        conn.commit()

        # Step C — Static blocking (idempotent; main flow already applied Tier 1,
        # but included here so the method is self-contained when called directly)
        cursor.execute(
            f"UPDATE {table_name}\n"
            f"SET blocking_factor = MAX(blocking_factor, wt_static_blocking)\n"
            f"WHERE wt_static_blocking IS NOT NULL AND wt_static_blocking > 1.0"
        )
        conn.commit()

        # Step C.2 — Null-depth penalty (unsurveyed = critically shallow)
        cursor.execute(
            f"UPDATE {table_name}\n"
            f"SET penalty_factor = MIN(penalty_factor * :ukc_restricted, :max_penalty)\n"
            f"WHERE ft_depth IS NULL",
            {'ukc_restricted': self.UKC_RESTRICTED_PENALTY, 'max_penalty': max_penalty},
        )
        conn.commit()

        # Step D — Populate wt_dynamic_* aggregate columns (smooth mode)
        existing = {row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")}
        for col in ('wt_dynamic_ukc_band', 'wt_dynamic_blocking',
                     'wt_dynamic_penalty', 'wt_dynamic_bonus'):
            if col not in existing:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN [{col}] REAL DEFAULT 1.0")
        conn.commit()

        cursor.execute(
            f"UPDATE {table_name} SET\n"
            f"    wt_dynamic_ukc_band = CASE\n"
            f"        WHEN ft_depth IS NULL THEN :ukc_restricted\n"
            f"        WHEN (ft_depth - :draft) <= 0 THEN :threshold\n"
            f"        WHEN (ft_depth - :draft) > :draft THEN 1.0\n"
            f"        ELSE 1.0 + ln(1.0 + MAX(0.0, :draft / MAX(ft_depth - :draft, 1e-6) - 1.0) * :penalty_hazard_scale)\n"
            f"    END,\n"
            f"    wt_dynamic_blocking = CASE\n"
            f"        WHEN blocking_factor >= :threshold THEN blocking_factor\n"
            f"        ELSE 1.0 END,\n"
            f"    wt_dynamic_penalty  = penalty_factor,\n"
            f"    wt_dynamic_bonus    = bonus_factor",
            {'threshold': self.BLOCKING_THRESHOLD, 'draft': draft,
             'penalty_hazard_scale': self.penalty_hazard_scale,
             'ukc_restricted': self.UKC_RESTRICTED_PENALTY},
        )
        conn.commit()

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]

        return {'updated_rows': total_rows, 'store_scores': store_scores}

    def _calculate_smooth_weights_postgis(
        self,
        conn,
        qualified_table: str,
        vessel_params: Dict[str, Any],
        store_scores: bool = True,
        max_penalty: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Execute smooth-mode weight computation against a PostGIS table.

        Mirrors _calculate_smooth_weights_sql() but uses server-side SQL
        with PostgreSQL functions (GREATEST, LN, EXP).

        Args:
            conn: Active SQLAlchemy connection from engine.begin() context
            qualified_table: Fully-qualified quoted table name,
                             e.g. '"graph"."fine_graph_01_edges"'
            vessel_params: Dict with at least 'draft'; optionally 'height',
                           'ver_clearance_margin' / 'clearance_safety_margin' /
                           'clearance_safety'
            store_scores: When True, persist preference_score and hazard_score
                          columns alongside the factor columns (default True)
            max_penalty: Penalty cap (default 100.0)

        Returns:
            Dict with keys:
                - 'updated_rows': total row count of the table
                - 'store_scores': the store_scores argument used
        """
        from sqlalchemy import text  # local import: DB dependency not at module level

        draft, vessel_height, _, clearance_safety, _ = self._extract_vessel_params(vessel_params)

        # Step A — Optional score columns
        if store_scores:
            for col_ddl in (
                f'ALTER TABLE {qualified_table}'
                f' ADD COLUMN IF NOT EXISTS preference_score DOUBLE PRECISION',
                f'ALTER TABLE {qualified_table}'
                f' ADD COLUMN IF NOT EXISTS hazard_score DOUBLE PRECISION',
            ):
                conn.execute(text(col_ddl))

        # Step B — Single CTE UPDATE: compute both scores and derive factors
        pref_expr = self._build_preference_score_sql_expr()
        hazard_expr = self._build_hazard_score_sql_expr()

        if store_scores:
            score_cols = (
                ",\n"
                "    preference_score = scores.pref,\n"
                "    hazard_score     = scores.hazard"
            )
        else:
            score_cols = ""

        cte_update_sql = text(f"""
WITH scores AS (
    SELECT id,
        {pref_expr} AS pref,
        {hazard_expr} AS hazard
    FROM {qualified_table}
)
UPDATE {qualified_table} e
SET
    bonus_factor   = GREATEST(
                         1.0 + EXP(-:bonus_decay_rate * scores.pref),
                         :min_bonus_factor
                     ),
    penalty_factor = 1.0 + LN(1.0 + scores.hazard * :penalty_hazard_scale){score_cols}
FROM scores
WHERE e.id = scores.id
""")

        conn.execute(cte_update_sql, {
            'draft': draft,
            'vessel_height': vessel_height,
            'clearance_safety': clearance_safety,
            'sounding_hazard_weight': self.sounding_hazard_weight,
            'bonus_decay_rate': self.bonus_decay_rate,
            'min_bonus_factor': self.MIN_BONUS_FACTOR,
            'penalty_hazard_scale': self.penalty_hazard_scale,
        })

        # Step B.5 — Populate ukc_meters for all edges with depth data.
        # The reset in calculate_dynamic_weights_postgis() nullifies ukc_meters for every row.
        # The step-band path restores it as a side-effect of its band/bonus UPDATEs.
        # The smooth CTE UPDATE does not touch ukc_meters, so we do it explicitly here.
        ukc_meters_sql = text(f"""
            UPDATE {qualified_table}
            SET ukc_meters = ft_depth - :draft
            WHERE ft_depth IS NOT NULL
        """)
        conn.execute(ukc_meters_sql, {'draft': draft})

        # Step B.6 — Penalty cap (safety guard)
        conn.execute(text(f"""
            UPDATE {qualified_table}
            SET penalty_factor = LEAST(penalty_factor, :max_penalty)
            WHERE penalty_factor > :max_penalty
        """), {'max_penalty': max_penalty})

        # Step B.7 — Bonus floor (safety guard, mirrors formula GREATEST)
        conn.execute(text(f"""
            UPDATE {qualified_table}
            SET bonus_factor = GREATEST(bonus_factor, :min_bonus_factor)
            WHERE bonus_factor < :min_bonus_factor
        """), {'min_bonus_factor': self.MIN_BONUS_FACTOR})

        # Step C — Static blocking (idempotent; main flow already applied Tier 1,
        # but included here so the method is self-contained when called directly)
        conn.execute(text(f"""
            UPDATE {qualified_table}
            SET blocking_factor = GREATEST(blocking_factor, wt_static_blocking)
            WHERE wt_static_blocking IS NOT NULL AND wt_static_blocking > 1.0
        """))

        # Step C.2 — Null-depth penalty (unsurveyed = critically shallow)
        conn.execute(text(f"""
            UPDATE {qualified_table}
            SET penalty_factor = LEAST(penalty_factor * :ukc_restricted, :max_penalty)
            WHERE ft_depth IS NULL
        """), {'ukc_restricted': self.UKC_RESTRICTED_PENALTY, 'max_penalty': max_penalty})

        # Step D — Populate wt_dynamic_* aggregate columns (smooth mode)
        for col_name in ('wt_dynamic_ukc_band', 'wt_dynamic_blocking',
                         'wt_dynamic_penalty', 'wt_dynamic_bonus'):
            conn.execute(text(
                f'ALTER TABLE {qualified_table}'
                f' ADD COLUMN IF NOT EXISTS {col_name} DOUBLE PRECISION DEFAULT 1.0'
            ))

        conn.execute(text(f"""
            UPDATE {qualified_table} SET
                wt_dynamic_ukc_band = CASE
                    WHEN ft_depth IS NULL THEN :ukc_restricted
                    WHEN (ft_depth - :draft) <= 0 THEN :threshold
                    WHEN (ft_depth - :draft) > :draft THEN 1.0
                    ELSE 1.0 + LN(1.0 + GREATEST(0.0, :draft / GREATEST(ft_depth - :draft, 1e-6) - 1.0) * :penalty_hazard_scale)
                END,
                wt_dynamic_blocking = CASE
                    WHEN blocking_factor >= :threshold THEN blocking_factor
                    ELSE 1.0 END,
                wt_dynamic_penalty  = penalty_factor,
                wt_dynamic_bonus    = bonus_factor
        """), {'threshold': self.BLOCKING_THRESHOLD, 'draft': draft,
               'penalty_hazard_scale': self.penalty_hazard_scale,
               'ukc_restricted': self.UKC_RESTRICTED_PENALTY})

        total_rows = conn.execute(
            text(f"SELECT COUNT(*) FROM {qualified_table}")
        ).scalar()

        return {'updated_rows': total_rows, 'store_scores': store_scores}

    def _calculate_smooth_weights_gdf(
        self,
        gdf: 'gpd.GeoDataFrame',
        vessel_params: Dict[str, Any],
        max_penalty: float = 100.0,
        store_scores: bool = True,
        buffer_zone_distances: Optional[List[float]] = None,
    ) -> 'gpd.GeoDataFrame':
        """
        Vectorised smooth-mode weight computation on a GeoDataFrame.

        Mirrors _calculate_smooth_weights_sql() and
        _calculate_smooth_weights_postgis() using numpy operations.

        The caller is responsible for initialising factor columns to 1.0 and
        applying Tier 1 (UKC grounding) blocking BEFORE calling this method.
        This method computes Tier 2 (penalty) and Tier 3 (bonus) via
        continuous exp/ln formulas, applies guards, and populates the
        wt_dynamic_* aggregate columns.

        Args:
            gdf: Edges GeoDataFrame with blocking_factor, penalty_factor,
                 bonus_factor already initialised.
            vessel_params: Dict with at least 'draft'.
            max_penalty: Penalty cap (default 100.0).
            store_scores: Persist preference_score / hazard_score columns.
            buffer_zone_distances: Zone distance thresholds (e.g. [3.0, 4.0, 12.0])
                aligned with ``compliance_zone`` in *vessel_params*.  Required
                for per-edge compliance_zone mapping; when ``None`` the method
                falls back to ``wt_zone_penalty`` column.

        Returns:
            GeoDataFrame with smooth factors and aggregate columns populated.
        """
        draft, vessel_height, _, clearance_safety, _ = self._extract_vessel_params(vessel_params)
        eps = 1e-6

        # --- helper: safe column access ----------------------------------------
        def _col(name: str, default: float = np.nan) -> pd.Series:
            if name in gdf.columns:
                return gdf[name]
            return pd.Series(default, index=gdf.index)

        ft_depth = _col('ft_depth')
        ft_sounding = _col('ft_sounding')
        ft_ver_clearance = _col('ft_ver_clearance')
        wt_static_bonus = _col('wt_static_bonus', 0.0).fillna(0.0)
        wt_static_penalty = _col('wt_static_penalty', 1.0).fillna(1.0)

        has_depth = ft_depth.notna()
        has_sounding = ft_sounding.notna()
        has_clearance = ft_ver_clearance.notna()
        ukc = ft_depth - draft

        # --- Directional components (mirrors _compute_preference_score / _compute_hazard_score) ---
        wt_dir_vals = _col('wt_dir', self.OPEN_WATER_BASE_MULTIPLIER).fillna(
            self.OPEN_WATER_BASE_MULTIPLIER
        ).values
        dir_pref = np.maximum(0.0, self.OPEN_WATER_BASE_MULTIPLIER - wt_dir_vals)
        dir_hazard = np.maximum(0.0, wt_dir_vals - self.OPEN_WATER_BASE_MULTIPLIER)

        # --- Preference score (mirrors _compute_preference_score) ---------------
        static_pref = np.clip(wt_static_bonus.values, 0.0, 1.0)
        depth_pref = np.where(
            has_depth.values & (ukc.values > draft),
            (ukc.values - draft) / max(draft, eps),
            0.0,
        )
        preference_score = np.maximum(0.0, static_pref + depth_pref + dir_pref)

        # --- Hazard score (mirrors _compute_hazard_score) -----------------------
        # Component 1: static penalty excess
        hazard_static = np.maximum(0.0, wt_static_penalty.values - 1.0)

        # Component 2: UKC risk (0 < ukc <= draft)
        ukc_vals = ukc.values
        ukc_risk_mask = has_depth.values & (ukc_vals > 0) & (ukc_vals <= draft)
        hazard_ukc = np.where(
            ukc_risk_mask,
            np.maximum(0.0, draft / np.maximum(ukc_vals, eps) - 1.0),
            0.0,
        )

        # Component 3: Clearance risk (vessel fits but margin is tight)
        clr_vals = ft_ver_clearance.values
        clr_risk_mask = (
            has_clearance.values
            & (clr_vals >= vessel_height)
            & (clr_vals < vessel_height + clearance_safety)
        )
        hazard_clearance = np.where(
            clr_risk_mask,
            clearance_safety / np.maximum(clr_vals - vessel_height, eps),
            0.0,
        )

        # Component 4: Sounding risk (weighted)
        snd_ukc = ft_sounding.values - draft
        snd_risk_mask = has_sounding.values & (snd_ukc > 0) & (snd_ukc <= draft)
        hazard_sounding = np.where(
            snd_risk_mask,
            self.sounding_hazard_weight * np.maximum(
                0.0, draft / np.maximum(snd_ukc, eps) - 1.0
            ),
            0.0,
        )

        # Component 5: Zone penalty hazard (regulatory/environmental boundaries)
        # When compliance_zone is provided with buffer_zone_distances, build per-edge
        # zone penalties from the compliance values (same mapping as step-band mode).
        # Otherwise fall back to pre-computed wt_zone_penalty column.
        compliance_zone = vessel_params.get('compliance_zone', None)
        if compliance_zone is not None and buffer_zone_distances is not None \
                and 'ft_buffer_zone_dist' in gdf.columns:
            # Per-edge mapping: ft_buffer_zone_dist → compliance_zone value
            dist_to_mult = {0.0: 1.0}
            for dist_nm, mult in zip(buffer_zone_distances, compliance_zone):
                dist_to_mult[float(dist_nm)] = float(mult)
            effective_zone = gdf['ft_buffer_zone_dist'].map(dist_to_mult).fillna(1.0).values
            zone_base = np.maximum(0.0, effective_zone - 1.0)
        else:
            # No compliance override — use pre-computed wt_zone_penalty
            wt_zone_penalty_col = _col('wt_zone_penalty', 1.0).fillna(1.0)
            zone_base = np.maximum(0.0, wt_zone_penalty_col.values - 1.0)
        hazard_zone = zone_base

        hazard_score = hazard_static + hazard_ukc + hazard_clearance + hazard_sounding + dir_hazard + hazard_zone

        # --- Bonus factor: 1 + exp(-k * pref), floored -------------------------
        bonus_factor = np.maximum(
            1.0 + np.exp(-self.bonus_decay_rate * preference_score),
            self.MIN_BONUS_FACTOR,
        )

        # --- Penalty factor: 1 + ln(1 + hazard * scale), capped ----------------
        penalty_factor = np.minimum(
            1.0 + np.log1p(hazard_score * self.penalty_hazard_scale),
            max_penalty,
        )

        # --- Null-depth penalty (unsurveyed = critically shallow) ---------------
        null_depth = ~has_depth.values
        if null_depth.any():
            penalty_factor = np.where(
                null_depth,
                penalty_factor * self.UKC_RESTRICTED_PENALTY,
                penalty_factor,
            )

        # --- Assign to GDF -----------------------------------------------------
        gdf['penalty_factor'] = penalty_factor
        gdf['bonus_factor'] = bonus_factor
        gdf.loc[has_depth, 'ukc_meters'] = ukc[has_depth]

        if store_scores:
            gdf['preference_score'] = preference_score
            gdf['hazard_score'] = hazard_score

        # Static blocking (idempotent — Tier 1 already ran in the caller)
        static_blk = _col('wt_static_blocking', 1.0).fillna(1.0)
        blk_mask = static_blk > 1.0
        if blk_mask.any():
            gdf.loc[blk_mask, 'blocking_factor'] = np.maximum(
                gdf.loc[blk_mask, 'blocking_factor'].values,
                static_blk[blk_mask].values,
            )

        # --- wt_dynamic_* aggregate columns (smooth mode) ----------------------
        # Depth-only smooth UKC band (decoupled from composite penalty)
        depth_only_hazard = np.where(
            has_depth.values & (ukc_vals > 0) & (ukc_vals <= draft),
            np.maximum(0.0, draft / np.maximum(ukc_vals, eps) - 1.0),
            0.0,
        )
        smooth_ukc_band = np.where(
            has_depth.values & (ukc_vals > 0),
            1.0 + np.log1p(depth_only_hazard * self.penalty_hazard_scale),
            1.0,
        )
        grounding = has_depth.values & (ukc_vals <= 0)
        smooth_ukc_band = np.where(grounding, self.BLOCKING_THRESHOLD, smooth_ukc_band)
        null_depth = ~has_depth.values
        smooth_ukc_band = np.where(null_depth, self.UKC_RESTRICTED_PENALTY, smooth_ukc_band)
        gdf['wt_dynamic_ukc_band'] = smooth_ukc_band
        gdf['wt_dynamic_blocking'] = np.where(
            gdf['blocking_factor'].values >= self.BLOCKING_THRESHOLD,
            gdf['blocking_factor'].values,
            1.0,
        )
        gdf['wt_dynamic_penalty'] = penalty_factor
        gdf['wt_dynamic_bonus'] = bonus_factor

        return gdf

