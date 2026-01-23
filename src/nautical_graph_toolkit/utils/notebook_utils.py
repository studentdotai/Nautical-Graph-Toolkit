"""
notebook_utils.py

Utility functions for Jupyter notebook operations in the Nautical Graph Toolkit.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import pandas as pd

# Optional plotly import for visualization
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class BenchmarkLoader:
    """Load and analyze historical benchmark data from notebooks.

    Provides time estimates based on similar previous runs by reading
    benchmark CSV files from the notebook output directory.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize with output directory containing benchmark CSVs.

        Args:
            output_dir: Directory containing benchmark_graph_*.csv files.
                       If None, will auto-detect from current working directory.
        """
        self._output_dir = output_dir
        self._cached_df: Optional[pd.DataFrame] = None

    def get_default_output_dir(self) -> Path:
        """Get default benchmark output directory.

        Detects the appropriate output directory based on context:
        1. If running from notebook (docs/notebooks/): use cwd/output
        2. Otherwise: use project_root/docs/notebooks/output

        Returns:
            Path to the benchmark output directory.
        """
        # Check if we're running from docs/notebooks/
        cwd = Path.cwd()
        if 'docs' in cwd.parts and 'notebooks' in cwd.parts:
            # Running from notebook directory
            return cwd / 'output'

        # Otherwise, try to find project root and navigate to docs/notebooks/output
        current = cwd
        for _ in range(5):  # Search up to 5 levels up
            if (current / 'src' / 'nautical_graph_toolkit').exists():
                return current / 'docs' / 'notebooks' / 'output'
            current = current.parent

        # Fallback to cwd/output
        logger.warning(f"Could not find project root, using {cwd / 'output'}")
        return cwd / 'output'

    def discover_benchmark_files(self) -> List[Path]:
        """Find all benchmark_graph_*.csv files.

        Returns:
            List of paths to benchmark CSV files.
        """
        output_dir = self._output_dir or self.get_default_output_dir()

        if not output_dir.exists():
            logger.warning(f"Benchmark output directory not found: {output_dir}")
            return []

        pattern = 'benchmark_graph_*.csv'
        files = list(output_dir.glob(pattern))

        if not files:
            logger.warning(f"No benchmark files found matching {pattern} in {output_dir}")

        logger.debug(f"Found {len(files)} benchmark files: {files}")
        return files

    def load_benchmark_data(self) -> pd.DataFrame:
        """Load and combine all benchmark CSVs into single DataFrame.

        Handles different schemas across files by using an outer join,
        which preserves all columns and fills missing values with NaN.

        Returns:
            Combined DataFrame with all benchmark data, or empty DataFrame
            if no files found or all files failed to load.
        """
        # Return cached data if available
        if self._cached_df is not None:
            return self._cached_df

        files = self.discover_benchmark_files()

        if not files:
            return pd.DataFrame()

        dataframes = []

        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                logger.debug(f"Loaded {len(df)} records from {file_path.name}")
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue

        if not dataframes:
            logger.warning("No benchmark data could be loaded")
            return pd.DataFrame()

        # Verify required columns exist in at least one dataframe
        required_cols = ['workflow', 'total_pipeline_sec']
        for col in required_cols:
            if not any(col in df.columns for df in dataframes):
                logger.error(f"Required column '{col}' not found in any benchmark file")
                return pd.DataFrame()

        # Concatenate all dataframes with outer join (preserves all columns)
        # Missing columns will be filled with NaN
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)

        # Log all available columns
        logger.debug(f"All columns in combined data: {combined_df.columns.tolist()}")

        logger.info(f"Loaded {len(combined_df)} total benchmark records")
        self._cached_df = combined_df

        return combined_df

    def filter_benchmarks(
        self,
        df: pd.DataFrame,
        notebook: str,
        graph_mode: Optional[str] = None,
        spacing_nm: Optional[float] = None,
        backend: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter benchmark DataFrame by matching criteria.

        Args:
            df: Benchmark DataFrame to filter.
            notebook: Workflow name to match (exact match).
            graph_mode: Graph mode to match (e.g., 'fine', 'h3'). Optional.
            spacing_nm: Grid spacing in nautical miles (within 0.001 tolerance). Optional.
            backend: Backend type to match (e.g., 'PostGIS', 'GeoPackage'). Optional.

        Returns:
            Filtered DataFrame containing only matching records.
        """
        if df.empty:
            return df

        # Start with all rows
        mask = pd.Series([True] * len(df), index=df.index)

        # Exact workflow match (required)
        mask &= (df['workflow'] == notebook)

        # graph_mode match (optional, if column exists)
        if graph_mode is not None and 'graph_mode' in df.columns:
            mask &= (df['graph_mode'] == graph_mode)

        # spacing_nm match (optional, within tolerance, if column exists)
        if spacing_nm is not None and 'spacing_nm' in df.columns:
            # Convert to numeric, coercing errors to NaN
            spacing_col = pd.to_numeric(df['spacing_nm'], errors='coerce')
            mask &= (spacing_col.between(
                spacing_nm - 0.001,
                spacing_nm + 0.001
            ))

        # backend match (optional, if data_source column exists)
        if backend is not None and 'data_source' in df.columns:
            # Normalize backend names for matching
            backend_lower = backend.lower()
            mask &= (df['data_source'].str.lower() == backend_lower)

        filtered_df = df[mask].copy()

        logger.debug(
            f"Filtered {len(df)} records to {len(filtered_df)} matches "
            f"(notebook={notebook}, graph_mode={graph_mode}, "
            f"spacing_nm={spacing_nm}, backend={backend})"
        )

        return filtered_df

    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics from filtered benchmark data.

        Args:
            df: DataFrame with benchmark records (must have total_pipeline_sec column).

        Returns:
            Dictionary with statistics: mean, std_dev, min, max, count, matches.
            Returns empty dict if DataFrame is empty or missing required column.
        """
        if df.empty:
            return {}

        if 'total_pipeline_sec' not in df.columns:
            logger.error("Column 'total_pipeline_sec' not found in benchmark data")
            return {}

        times = df['total_pipeline_sec']

        stats = {
            'mean': float(times.mean()),
            'std_dev': float(times.std()) if len(times) > 1 else 0.0,
            'min': float(times.min()),
            'max': float(times.max()),
            'count': len(times),
            'matches': df.to_dict('records')  # Include raw records for debugging
        }

        logger.info(
            f"Calculated statistics from {stats['count']} records: "
            f"mean={stats['mean']:.1f}s, std_dev={stats['std_dev']:.1f}s"
        )

        return stats

    def load_estimates(
        self,
        notebook: str,
        graph_mode: Optional[str] = None,
        spacing_nm: Optional[float] = None,
        backend: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load estimates based on historical benchmark data.

        Main entry point that loads, filters, and calculates statistics
        from matching benchmark records.

        Args:
            notebook: Workflow name to match (exact match). Required.
            graph_mode: Graph mode filter (e.g., 'fine', 'h3'). Optional.
            spacing_nm: Grid spacing filter in nautical miles. Optional.
            backend: Backend type filter (e.g., 'PostGIS', 'GeoPackage'). Optional.

        Returns:
            Dictionary with statistics (mean, std_dev, min, max, count, matches)
            if matches found, None otherwise.
        """
        df = self.load_benchmark_data()

        if df.empty:
            logger.debug("No benchmark data available")
            return None

        filtered_df = self.filter_benchmarks(
            df=df,
            notebook=notebook,
            graph_mode=graph_mode,
            spacing_nm=spacing_nm,
            backend=backend
        )

        if filtered_df.empty:
            logger.debug(f"No matching benchmark records found for {notebook}")
            return None

        return self.calculate_statistics(filtered_df)


def load_estimates(
    notebook: str,
    graph_mode: Optional[str] = None,
    spacing_nm: Optional[float] = None,
    backend: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Load benchmark estimates from historical runs.

    Convenience function that creates a BenchmarkLoader instance
    and returns filtered statistics.

    This is the primary interface for use in Jupyter notebooks:
    ```python
    from nautical_graph_toolkit.utils.notebook_utils import load_estimates

    estimate = load_estimates(
        notebook='graph_PostGIS_v2',
        graph_mode='fine',
        spacing_nm=0.1,
        backend='PostGIS'
    )

    if estimate:
        print(f"⏱️  Estimated duration: {estimate['mean']:.1f} ± {estimate['std_dev']:.1f}s")
        print(f"   Based on {estimate['count']} previous runs")
    ```

    Args:
        notebook: Workflow name to match exactly (e.g., 'graph_PostGIS_v2'). Required.
        graph_mode: Graph mode filter (e.g., 'fine', 'h3'). Optional.
        spacing_nm: Grid spacing filter in nautical miles. Optional.
        backend: Backend type filter (e.g., 'PostGIS', 'GeoPackage'). Optional.
        output_dir: Custom directory containing benchmark CSV files. Optional.

    Returns:
        Dictionary with statistics (mean, std_dev, min, max, count, matches)
        if matches found, None otherwise.
    """
    loader = BenchmarkLoader(output_dir=output_dir)
    return loader.load_estimates(
        notebook=notebook,
        graph_mode=graph_mode,
        spacing_nm=spacing_nm,
        backend=backend
    )


class BenchmarkLogger:
    """Unified benchmark logging and historical data loader.

    Provides both performance tracking during notebook execution
    and historical benchmark data loading for time estimates.
    """

    # Workflow type constants
    WORKFLOW_BASE = "base"
    WORKFLOW_FINE = "fine"
    WORKFLOW_WEIGHTED = "weighted"

    # Metric key definitions by workflow type
    METRIC_KEYS = {
        WORKFLOW_BASE: [
            'port_selection_boundary_sec', 'enc_filtering_sec',
            'grid_creation_sec', 'graph_creation_sec',
            'save_gpkg_sec', 'save_postgis_sec', 'pathfinding_sec'
        ],
        WORKFLOW_FINE: [
            'load_base_route_sec', 'create_buffer_sec', 'slice_buffer_sec',
            'fine_grid_creation_sec', 'fine_graph_creation_sec', 'h3_graph_creation_sec',
            'save_gpkg_sec', 'save_postgis_original_sec', 'save_postgis_optimized_sec',
            'route_calculation_sec'
        ],
        WORKFLOW_WEIGHTED: [
            'conversion_to_directed_sec', 'edge_enrichment_sec', 'static_weights_sec',
            'directional_weights_sec', 'dynamic_weights_sec', 'graph_loading_sec',
            'route_calculation_sec'
        ]
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize with output directory for benchmark CSVs.

        Args:
            output_dir: Directory for benchmark CSV files. If None, will auto-detect.
        """
        self._output_dir = output_dir
        self._workflow_type: Optional[str] = None
        self._config: Dict[str, Any] = {}
        self._timings: Dict[str, float] = {}
        self._results: Dict[str, Any] = {}
        self._active_timers: Dict[str, float] = {}

        # Reuse BenchmarkLoader's output directory detection
        self._loader = BenchmarkLoader(output_dir=output_dir)

    def get_default_output_dir(self) -> Path:
        """Get default benchmark output directory.

        Delegates to BenchmarkLoader.get_default_output_dir().
        """
        return self._loader.get_default_output_dir()

    # === Configuration ===

    def configure_base_graph(
        self,
        spacing_nm: float,
        graph_mode: str,
        reduce_distance_nm: float,
        aoi: Optional[str] = None,
        db_schema: Optional[str] = None
    ) -> None:
        """Configure for base graph workflow.

        Args:
            spacing_nm: Grid spacing in nautical miles.
            reduce_distance_nm: Distance reduction in nautical miles.
            aoi: Area of interest (optional).
            db_schema: Database schema (optional).
        """
        self._workflow_type = self.WORKFLOW_BASE
        self._config = {
            'graph_mode': graph_mode,
            'spacing_nm': spacing_nm,
            'reduce_distance_nm': reduce_distance_nm,
        }
        if aoi is not None:
            self._config['aoi'] = aoi
        if db_schema is not None:
            self._config['db_schema'] = db_schema

        logger.debug(f"Configured for base graph workflow: {self._config}")

    def configure_fine_graph(
        self,
        graph_mode: str,
        spacing_nm: float,
        buffer_size_nm: float,
        buffer_sliced: bool = False,
        keep_largest_component: bool = True,
        db_schema: Optional[str] = None,
        max_points: Optional[int] = None
    ) -> None:
        """Configure for fine graph workflow.

        Args:
            graph_mode: Graph mode ('fine' or 'h3').
            spacing_nm: Grid spacing in nautical miles.
            buffer_size_nm: Buffer size in nautical miles.
            buffer_sliced: Whether buffer was sliced.
            keep_largest_component: Whether to keep largest component.
            db_schema: Database schema for PostGIS operations (optional).
            max_points: Maximum points per subdivision to avoid memory issues (optional).
        """
        self._workflow_type = self.WORKFLOW_FINE
        self._config = {
            'graph_mode': graph_mode,
            'spacing_nm': spacing_nm,
            'buffer_size_nm': buffer_size_nm,
            'buffer_sliced': buffer_sliced,
            'keep_largest_component': keep_largest_component,
        }
        if db_schema is not None:
            self._config['db_schema'] = db_schema
        if max_points is not None:
            self._config['max_points'] = max_points

        logger.debug(f"Configured for fine graph workflow: {self._config}")

    def configure_weighted_graph(
        self,
        vessel_draft_m: float,
        vessel_height_m: float,
        vessel_type: str = "cargo",
        weather_factor: float = 1.0,
        visibility_factor: float = 1.0,
        time_of_day: str = "day",
        enc_count: Optional[int] = None
    ) -> None:
        """Configure for weighted directed graph workflow.

        Args:
            vessel_draft_m: Vessel draft in meters.
            vessel_height_m: Vessel height in meters.
            vessel_type: Vessel type (default: "cargo").
            weather_factor: Weather factor (default: 1.0).
            visibility_factor: Visibility factor (default: 1.0).
            time_of_day: Time of day (default: "day").
            enc_count: Number of ENC charts (optional).
        """
        self._workflow_type = self.WORKFLOW_WEIGHTED
        self._config = {
            'vessel_draft_m': vessel_draft_m,
            'vessel_height_m': vessel_height_m,
            'vessel_type': vessel_type,
            'weather_factor': weather_factor,
            'visibility_factor': visibility_factor,
            'time_of_day': time_of_day,
        }
        if enc_count is not None:
            self._config['enc_count'] = enc_count

        logger.debug(f"Configured for weighted graph workflow: {self._config}")

    # === Performance Tracking ===

    def start_timer(self, step_name: str) -> None:
        """Start timing a workflow step.

        Args:
            step_name: Name of the step (will be suffixed with '_sec' in output).
        """
        self._active_timers[step_name] = time.time()
        logger.debug(f"Started timer: {step_name}")

    def end_step(self, step_name: str) -> float:
        """End timing a workflow step and return elapsed seconds.

        Args:
            step_name: Name of the step (must match a previous start_timer call).

        Returns:
            Elapsed time in seconds.

        Raises:
            ValueError: If no active timer exists for the step.
        """
        if step_name not in self._active_timers:
            raise ValueError(f"No active timer for step: {step_name}")

        elapsed = time.time() - self._active_timers[step_name]
        metric_key = f"{step_name}_sec"
        self._timings[metric_key] = elapsed
        del self._active_timers[step_name]

        logger.debug(f"Ended timer: {step_name} ({elapsed:.2f}s)")
        return elapsed

    def set_result(self, key: str, value: Any) -> None:
        """Set a result value (node_count, edge_count, etc.).

        Args:
            key: Result key name.
            value: Result value.
        """
        self._results[key] = value
        logger.debug(f"Set result: {key} = {value}")

    # === Export ===

    def _get_csv_filename(self) -> str:
        """Determine CSV filename based on workflow type and backend.

        Returns:
            CSV filename (e.g., 'benchmark_graph_base.csv').
        """
        if self._workflow_type == self.WORKFLOW_BASE:
            return 'benchmark_graph_base.csv'
        elif self._workflow_type == self.WORKFLOW_FINE:
            return 'benchmark_graph_fine.csv'
        elif self._workflow_type == self.WORKFLOW_WEIGHTED:
            # For weighted directed, append backend suffix
            data_source = self._results.get('data_source', '')
            if data_source and 'geopackage' in data_source.lower():
                return 'benchmark_graph_weighted_directed_gpkg.csv'
            else:
                return 'benchmark_graph_weighted_directed.csv'
        else:
            raise ValueError(f"Unknown workflow type: {self._workflow_type}")

    def export_benchmark(self) -> Path:
        """Export collected benchmark data to CSV and return file path.

        Returns:
            Path to the CSV file.

        Raises:
            ValueError: If workflow not configured or required fields are missing.
        """
        if self._workflow_type is None:
            raise ValueError("Workflow not configured. Call configure_*_graph() first.")

        if 'workflow' not in self._results:
            raise ValueError("Workflow name not set. Call set_result('workflow', name) first.")

        if 'data_source' not in self._results:
            raise ValueError("Data source not set. Call set_result('data_source', name) first.")

        output_dir = self._output_dir or self.get_default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_filename = self._get_csv_filename()
        csv_path = output_dir / csv_filename

        # Build benchmark record
        record = {
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'workflow': self._results['workflow'],
            'data_source': self._results['data_source'],
        }

        # Add configuration parameters
        record.update(self._config)

        # Add timing metrics
        record.update(self._timings)

        # Add results
        record.update(self._results)

        # Calculate total_pipeline_sec
        timing_values = [v for k, v in self._timings.items() if k.endswith('_sec')]
        if timing_values:
            record['total_pipeline_sec'] = sum(timing_values)
        else:
            record['total_pipeline_sec'] = 0.0

        # Calculate normalized metrics (per 100K nodes) if node_count is available
        node_count = self._results.get('node_count', 0)
        if node_count and node_count > 0:
            for key, value in self._timings.items():
                if key.endswith('_sec'):
                    normalized_key = key.replace('_sec', '_per_100k_nodes')
                    record[normalized_key] = (value / node_count) * 100000
            logger.debug(f"Calculated normalized metrics for {node_count} nodes")
        else:
            logger.debug("No node_count available, skipping normalized metrics")

        # Append to existing CSV or create new one
        if csv_path.exists():
            # Read existing data and append
            try:
                existing_df = pd.read_csv(csv_path)
                new_df = pd.DataFrame([record])
                combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
                combined_df.to_csv(csv_path, index=False)
                logger.info(f"Appended benchmark to existing file: {csv_path}")
            except Exception as e:
                logger.error(f"Failed to append to existing CSV, creating new file: {e}")
                pd.DataFrame([record]).to_csv(csv_path, index=False)
                logger.info(f"Created new benchmark file: {csv_path}")
        else:
            pd.DataFrame([record]).to_csv(csv_path, index=False)
            logger.info(f"Created new benchmark file: {csv_path}")

        return csv_path

    def _format_metric_label(self, key: str) -> str:
        """Format metric key to human-readable label.

        Args:
            key: Metric key (e.g., 'graph_creation_sec').

        Returns:
            Formatted label (e.g., 'Graph Creation').
        """
        # Remove '_sec' suffix and convert snake_case to Title Case
        return key.replace('_sec', '').replace('_', ' ').title()

    def get_current_benchmark_summary(
        self,
        csv_path: Optional[Path] = None,
        top_n: int = 5
    ) -> str:
        """Get formatted summary of the most recent benchmark record.

        Reads the last entry from the CSV file and returns a formatted
        string with key metrics and top time-consuming operations.

        Args:
            csv_path: Path to the benchmark CSV file. If None, will auto-detect
                     based on current workflow type and output directory.
            top_n: Number of top operations to display (default: 5).

        Returns:
            Formatted string with benchmark summary.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If workflow type is not set or CSV is empty.
        """
        # Determine CSV path if not provided
        if csv_path is None:
            if self._workflow_type is None:
                raise ValueError("Workflow not configured. Call configure_*_graph() first.")
            output_dir = self._output_dir or self.get_default_output_dir()
            csv_filename = self._get_csv_filename()
            csv_path = output_dir / csv_filename

        # Check if file exists
        if not csv_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {csv_path}")

        # Read CSV and get last row
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"Benchmark file is empty: {csv_path}")

        # Get most recent record (last row)
        record = df.iloc[-1]

        # Extract key fields with defaults for missing values
        timestamp = record.get('timestamp', 'N/A')
        workflow = record.get('workflow', 'N/A')
        data_source = record.get('data_source', 'N/A')
        node_count = record.get('node_count', 0)
        edge_count = record.get('edge_count', 0)
        total_time = record.get('total_pipeline_sec', 0.0)

        # Format node/edge counts with thousands separator
        nodes_str = f"{int(node_count):,}" if pd.notna(node_count) else 'N/A'
        edges_str = f"{int(edge_count):,}" if pd.notna(edge_count) else 'N/A'

        # Build output lines
        lines = [
            "=== Current Benchmark Record ===",
            f"Timestamp: {timestamp}",
            f"Workflow: {workflow}",
            f"Data Source: {data_source}",
            f"Nodes: {nodes_str}",
            f"Edges: {edges_str}",
            f"Total Pipeline Time: {total_time:.2f}s",
            "",
            "Most demanding operations:"
        ]

        # Find all timing columns (ending with '_sec', excluding 'total_pipeline_sec')
        timing_cols = [col for col in df.columns if col.endswith('_sec') and col != 'total_pipeline_sec']

        # Sort by time descending and take top N
        timing_data = []
        for col in timing_cols:
            value = record.get(col)
            if pd.notna(value) and value > 0:
                timing_data.append((col, value))

        timing_data.sort(key=lambda x: x[1], reverse=True)
        timing_data = timing_data[:top_n]

        # Add top operations to output
        if timing_data:
            for i, (col, value) in enumerate(timing_data, 1):
                label = self._format_metric_label(col)
                lines.append(f"  {i}. {label}: {value:.2f}s")
        else:
            lines.append("  No timing data available")

        return "\n".join(lines)

    def visualize_performance(
        self,
        title: Optional[str] = None,
        sort_by: str = 'time_descending',
        show: bool = True
    ) -> Optional[Any]:
        """Create interactive bar chart of performance metrics.

        Creates a horizontal bar chart showing timing breakdown of all
        recorded steps. Useful for identifying bottlenecks in notebook workflows.

        Args:
            title: Custom chart title. If None, auto-generated from workflow info.
            sort_by: How to sort bars ('time_descending', 'time_ascending', 'name').
            show: Whether to display the chart (uses fig.show() if True).

        Returns:
            Plotly Figure object for further customization, or None if plotly
            is not installed or no timing data is available.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("plotly not installed, skipping visualization")
            return None

        if not self._timings:
            logger.warning("No timing data available for visualization")
            return None

        # Convert timings to DataFrame for plotting
        # Remove _sec suffix for cleaner labels
        perf_data = {
            'Step': [k.replace('_sec', '').replace('_', ' ').title() for k in self._timings.keys()],
            'Time (seconds)': list(self._timings.values())
        }
        perf_df = pd.DataFrame(perf_data)

        # Sort based on preference
        if sort_by == 'time_descending':
            perf_df = perf_df.sort_values(by='Time (seconds)', ascending=False)
        elif sort_by == 'time_ascending':
            perf_df = perf_df.sort_values(by='Time (seconds)', ascending=True)
        elif sort_by == 'name':
            perf_df = perf_df.sort_values(by='Step')

        # Generate title if not provided
        if title is None:
            workflow_name = self._results.get('workflow', 'Workflow')
            data_source = self._results.get('data_source', '')
            title = f'{workflow_name} Pipeline Performance'
            if data_source:
                title += f' ({data_source})'

        # Create horizontal bar chart
        fig = px.bar(
            perf_df,
            x='Time (seconds)',
            y='Step',
            orientation='h',
            title=title,
            text_auto='.2f',
            labels={'Step': 'Pipeline Step', 'Time (seconds)': 'Time (seconds)'},
            category_orders={'Step': list(perf_df['Step'])}  # Maintain sort order
        )

        # Format: put text outside bars and use consistent color
        fig.update_traces(
            textposition='outside',
            marker_color='steelblue'
        )

        # Improve layout
        fig.update_layout(
            xaxis_title='Time (seconds)',
            yaxis_title='Pipeline Step',
            height=max(400, len(perf_df) * 40),  # Dynamic height based on step count
            margin=dict(l=20, r=20, t=40, b=20)
        )

        if show:
            fig.show()

        return fig

    # === Historical Data Loading ===

    def load_estimates(
        self,
        notebook: str,
        graph_mode: Optional[str] = None,
        spacing_nm: Optional[float] = None,
        backend: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load historical benchmark estimates.

        Delegates to BenchmarkLoader.load_estimates().

        Args:
            notebook: Workflow name to match (exact match). Required.
            graph_mode: Graph mode filter (e.g., 'fine', 'h3'). Optional.
            spacing_nm: Grid spacing filter in nautical miles. Optional.
            backend: Backend type filter (e.g., 'PostGIS', 'GeoPackage'). Optional.

        Returns:
            Dictionary with statistics (mean, std_dev, min, max, count, matches)
            if matches found, None otherwise.
        """
        return self._loader.load_estimates(
            notebook=notebook,
            graph_mode=graph_mode,
            spacing_nm=spacing_nm,
            backend=backend
        )