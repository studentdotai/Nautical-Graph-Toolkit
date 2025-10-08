#!/usr/bin/env python3
"""
Diagnostic script to analyze ft_depth_sources JSONB column.
Identifies which S57 layers are contributing ft_depth = 0 values.

Usage:
    .venv/bin/python analyze_depth_sources.py <graph_name> [--schema graph] [--enc-schema public]

Example:
    .venv/bin/python analyze_depth_sources.py fine_graph_01 --schema graph --enc-schema us_enc_all
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, 'src')

from sqlalchemy import create_engine, text
from maritime_module.core.graph import GraphConfigManager

def analyze_depth_sources(graph_name: str, schema: str = 'graph', enc_schema: str = 'public'):
    """
    Analyze ft_depth_sources to identify layers causing ft_depth = 0.

    Args:
        graph_name: Name of the graph (e.g., 'fine_graph_01')
        schema: Schema containing graph tables (default: 'graph')
        enc_schema: Schema containing S57 layers (default: 'public')
    """
    edges_table = f"{graph_name}_edges"

    # Load database connection from config
    config_path = Path(__file__).parent / 'config' / 'graph_config.yml'
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        print("Please provide database connection details manually or create config file")
        return

    config_manager = GraphConfigManager(config_path)
    db_config = config_manager.get_value('postgis')

    # Build connection string
    conn_str = (
        f"postgresql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

    engine = create_engine(conn_str)

    print("=" * 80)
    print(f"ANALYZING ft_depth_sources FOR: {schema}.{edges_table}")
    print("=" * 80)
    print()

    with engine.connect() as conn:
        # Query 1: Summary statistics
        print("SUMMARY STATISTICS")
        print("-" * 80)
        summary_sql = text(f"""
            SELECT
                'Total Edges' as metric,
                COUNT(*) as value
            FROM "{schema}"."{edges_table}"
            UNION ALL
            SELECT
                'Edges with ft_depth = 0' as metric,
                COUNT(*) as value
            FROM "{schema}"."{edges_table}"
            WHERE ft_depth = 0
            UNION ALL
            SELECT
                'Edges with ft_depth = 0 and sources tracked' as metric,
                COUNT(*) as value
            FROM "{schema}"."{edges_table}"
            WHERE ft_depth = 0 AND ft_depth_sources IS NOT NULL
            UNION ALL
            SELECT
                'Edges with blocking_factor >= 999' as metric,
                COUNT(*) as value
            FROM "{schema}"."{edges_table}"
            WHERE blocking_factor >= 999
            UNION ALL
            SELECT
                'Edges with ft_depth NULL' as metric,
                COUNT(*) as value
            FROM "{schema}"."{edges_table}"
            WHERE ft_depth IS NULL
        """)

        results = conn.execute(summary_sql)
        for row in results:
            print(f"{row[0]:50s}: {row[1]:,}")
        print()

        # Query 2: Layers and usage bands contributing to ft_depth = 0
        print("LAYERS AND USAGE BANDS CONTRIBUTING TO ft_depth = 0")
        print("-" * 80)
        layers_sql = text(f"""
            WITH depth_zero_edges AS (
                SELECT
                    id,
                    ft_depth_sources
                FROM "{schema}"."{edges_table}"
                WHERE ft_depth = 0
                  AND ft_depth_sources IS NOT NULL
            ),
            expanded_sources AS (
                SELECT
                    id,
                    jsonb_object_keys(ft_depth_sources) as source_key,
                    (ft_depth_sources -> jsonb_object_keys(ft_depth_sources) ->> 'depth')::float as depth_value,
                    (ft_depth_sources -> jsonb_object_keys(ft_depth_sources) ->> 'usage_band')::int as usage_band
                FROM depth_zero_edges
            )
            SELECT
                SPLIT_PART(source_key, '_', 1) as enc_name,
                SPLIT_PART(source_key, '_', 2) as layer_name,
                usage_band,
                COUNT(*) as edge_count,
                COUNT(CASE WHEN depth_value = 0 THEN 1 END) as zero_depth_count,
                MIN(depth_value) as min_depth,
                MAX(depth_value) as max_depth,
                ROUND(AVG(depth_value)::numeric, 2) as avg_depth
            FROM expanded_sources
            GROUP BY SPLIT_PART(source_key, '_', 1), SPLIT_PART(source_key, '_', 2), usage_band
            ORDER BY usage_band DESC, zero_depth_count DESC, edge_count DESC
        """)

        results = conn.execute(layers_sql)
        print(f"{'ENC Name':<12} {'Layer':<10} {'Band':<6} {'Edges':<8} {'Zero':<8} {'Min':<6} {'Max':<6} {'Avg':<6}")
        print("-" * 80)

        usage_band_names = {
            1: 'Overview',
            2: 'General',
            3: 'Coastal',
            4: 'Approach',
            5: 'Harbour',
            6: 'Berthing'
        }

        allowed_layers = {'depare', 'drgare', 'swpare'}
        for row in results:
            enc_name = row[0]
            layer = row[1]
            usage_band = row[2]
            edge_count = row[3]
            zero_count = row[4]
            min_d = row[5]
            max_d = row[6]
            avg_d = row[7]

            band_name = usage_band_names.get(usage_band, f'Band{usage_band}')

            print(f"{enc_name:<12} {layer:<10} {band_name:<6} {edge_count:<8,} {zero_count:<8,} {min_d:<6.1f} {max_d:<6.1f} {avg_d:<6}")
        print()

        # Query 3: Sample edges with details
        print("SAMPLE EDGES WITH ft_depth = 0 (First 10)")
        print("-" * 80)
        sample_sql = text(f"""
            SELECT
                id,
                ft_depth,
                blocking_factor,
                ukc_meters,
                ft_depth_sources
            FROM "{schema}"."{edges_table}"
            WHERE ft_depth = 0
              AND ft_depth_sources IS NOT NULL
            ORDER BY id
            LIMIT 10
        """)

        results = conn.execute(sample_sql)
        for row in results:
            edge_id = row[0]
            ft_depth = row[1]
            blocking = row[2]
            ukc = row[3]
            sources = row[4]

            print(f"Edge ID: {edge_id}")
            print(f"  ft_depth: {ft_depth}, blocking_factor: {blocking}, ukc_meters: {ukc}")
            print(f"  Source data (ENC_Layer: depth/usage_band):")
            if sources:
                for source_key, data in sources.items():
                    enc_name, layer = source_key.split('_', 1) if '_' in source_key else (source_key, '?')
                    depth_val = data.get('depth', '?') if isinstance(data, dict) else data
                    usage_band = data.get('usage_band', '?') if isinstance(data, dict) else '?'
                    band_name = usage_band_names.get(usage_band, f'Band{usage_band}') if usage_band != '?' else '?'
                    layer_allowed = layer in allowed_layers
                    status = "✓" if layer_allowed else "✗"
                    print(f"    {status} {enc_name}_{layer}: depth={depth_val}m, band={usage_band} ({band_name})")
            print()

        # Query 4: Depth distribution
        print("DEPTH VALUE DISTRIBUTION")
        print("-" * 80)
        dist_sql = text(f"""
            SELECT
                CASE
                    WHEN ft_depth IS NULL THEN 'NULL'
                    WHEN ft_depth = 0 THEN '0'
                    WHEN ft_depth > 0 AND ft_depth <= 5 THEN '0-5m'
                    WHEN ft_depth > 5 AND ft_depth <= 10 THEN '5-10m'
                    WHEN ft_depth > 10 AND ft_depth <= 20 THEN '10-20m'
                    WHEN ft_depth > 20 THEN '>20m'
                    ELSE 'negative'
                END as depth_range,
                COUNT(*) as edge_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM "{schema}"."{edges_table}"
            GROUP BY depth_range
            ORDER BY
                CASE depth_range
                    WHEN 'NULL' THEN 0
                    WHEN 'negative' THEN 1
                    WHEN '0' THEN 2
                    WHEN '0-5m' THEN 3
                    WHEN '5-10m' THEN 4
                    WHEN '10-20m' THEN 5
                    WHEN '>20m' THEN 6
                END
        """)

        results = conn.execute(dist_sql)
        print(f"{'Depth Range':<15} {'Edge Count':<15} {'Percentage':<10}")
        print("-" * 40)
        for row in results:
            print(f"{row[0]:<15} {row[1]:<15,} {row[2]:<10}%")
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze ft_depth_sources for depth anomalies')
    parser.add_argument('graph_name', help='Graph name (e.g., fine_graph_01)')
    parser.add_argument('--schema', default='graph', help='Schema containing graph tables (default: graph)')
    parser.add_argument('--enc-schema', default='public', help='Schema containing S57 layers (default: public)')

    args = parser.parse_args()

    try:
        analyze_depth_sources(args.graph_name, args.schema, args.enc_schema)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
