-- ============================================================================
-- DIAGNOSTIC QUERIES FOR ft_depth_sources ANALYSIS
-- ============================================================================
-- Purpose: Identify which S57 layers are contributing ft_depth = 0 values
-- causing false blocking_factor = 999 in coastal/harbour waters
--
-- Usage:
--   Replace 'graph.your_edges_table' with your actual table name
--   psql -d your_database -f analyze_depth_sources.sql
-- ============================================================================

-- Query 1: Find edges with ft_depth = 0 and their source layers
-- ============================================================================
SELECT
    id,
    ft_depth,
    blocking_factor,
    ukc_meters,
    ft_depth_sources,
    ST_AsText(ST_Centroid(geometry)) as edge_location
FROM graph.your_edges_table
WHERE ft_depth = 0
ORDER BY id
LIMIT 100;

-- Query 2: Count how many edges have ft_depth = 0 grouped by source layer
-- ============================================================================
WITH depth_zero_edges AS (
    SELECT
        id,
        ft_depth_sources
    FROM graph.your_edges_table
    WHERE ft_depth = 0
      AND ft_depth_sources IS NOT NULL
),
expanded_sources AS (
    SELECT
        id,
        jsonb_object_keys(ft_depth_sources) as layer_name,
        (ft_depth_sources ->> jsonb_object_keys(ft_depth_sources))::float as depth_value
    FROM depth_zero_edges
)
SELECT
    layer_name,
    COUNT(*) as edge_count,
    COUNT(CASE WHEN depth_value = 0 THEN 1 END) as zero_depth_count,
    MIN(depth_value) as min_depth,
    MAX(depth_value) as max_depth,
    AVG(depth_value) as avg_depth
FROM expanded_sources
GROUP BY layer_name
ORDER BY zero_depth_count DESC, edge_count DESC;

-- Query 3: Show distribution of all ft_depth values
-- ============================================================================
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
FROM graph.your_edges_table
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
    END;

-- Query 4: Detailed breakdown of layers contributing to each ft_depth = 0 edge
-- ============================================================================
WITH depth_zero_edges AS (
    SELECT
        id,
        ft_depth,
        blocking_factor,
        ft_depth_sources,
        ST_AsText(ST_Centroid(geometry)) as location
    FROM graph.your_edges_table
    WHERE ft_depth = 0
      AND ft_depth_sources IS NOT NULL
    LIMIT 50
)
SELECT
    id,
    ft_depth,
    blocking_factor,
    location,
    jsonb_pretty(ft_depth_sources) as source_layers_and_depths
FROM depth_zero_edges
ORDER BY id;

-- Query 5: Find which layer combinations frequently cause ft_depth = 0
-- ============================================================================
WITH depth_zero_edges AS (
    SELECT
        id,
        ft_depth_sources
    FROM graph.your_edges_table
    WHERE ft_depth = 0
      AND ft_depth_sources IS NOT NULL
),
layer_combinations AS (
    SELECT
        jsonb_object_keys(ft_depth_sources) as layers_contributing
    FROM depth_zero_edges
)
SELECT
    layers_contributing,
    COUNT(*) as occurrence_count
FROM layer_combinations
GROUP BY layers_contributing
ORDER BY occurrence_count DESC
LIMIT 20;

-- Query 6: Compare blocking_factor distribution for edges with/without ft_depth
-- ============================================================================
SELECT
    CASE
        WHEN ft_depth IS NULL THEN 'ft_depth = NULL'
        WHEN ft_depth = 0 THEN 'ft_depth = 0'
        ELSE 'ft_depth > 0'
    END as depth_status,
    CASE
        WHEN blocking_factor >= 999 THEN 'BLOCKED (999)'
        WHEN blocking_factor > 1 THEN 'HIGH (>1)'
        ELSE 'NORMAL (=1)'
    END as blocking_status,
    COUNT(*) as edge_count,
    ROUND(AVG(blocking_factor), 2) as avg_blocking_factor
FROM graph.your_edges_table
GROUP BY depth_status, blocking_status
ORDER BY depth_status, blocking_status;

-- Query 7: Sample of edges with ft_depth = 0 but NOT from filtered layers
-- This should help identify if the filter is working correctly
-- ============================================================================
WITH depth_zero_edges AS (
    SELECT
        id,
        ft_depth,
        ft_depth_sources
    FROM graph.your_edges_table
    WHERE ft_depth = 0
      AND ft_depth_sources IS NOT NULL
),
expanded_sources AS (
    SELECT
        id,
        jsonb_object_keys(ft_depth_sources) as layer_name
    FROM depth_zero_edges
)
SELECT
    layer_name,
    COUNT(*) as count,
    CASE
        WHEN layer_name IN ('depare', 'drgare', 'swpare')
        THEN '✓ ALLOWED (navigational)'
        ELSE '✗ SHOULD BE FILTERED'
    END as filter_status
FROM expanded_sources
GROUP BY layer_name
ORDER BY count DESC;

-- Query 8: Geographic distribution of ft_depth = 0 edges
-- (Useful to see if clustering in harbors/coastal areas)
-- ============================================================================
SELECT
    ST_AsText(ST_Centroid(ST_Collect(geometry))) as cluster_center,
    COUNT(*) as edge_count,
    AVG(blocking_factor) as avg_blocking_factor,
    ARRAY_AGG(DISTINCT jsonb_object_keys(ft_depth_sources)) as contributing_layers
FROM graph.your_edges_table
WHERE ft_depth = 0
  AND ft_depth_sources IS NOT NULL
GROUP BY ST_SnapToGrid(geometry, 0.01)  -- Group by ~1km grid
HAVING COUNT(*) > 5  -- Only show clusters with 5+ edges
ORDER BY edge_count DESC
LIMIT 20;

-- ============================================================================
-- SUMMARY STATISTICS
-- ============================================================================
SELECT
    'Total Edges' as metric,
    COUNT(*) as value
FROM graph.your_edges_table
UNION ALL
SELECT
    'Edges with ft_depth = 0' as metric,
    COUNT(*) as value
FROM graph.your_edges_table
WHERE ft_depth = 0
UNION ALL
SELECT
    'Edges with ft_depth = 0 and sources tracked' as metric,
    COUNT(*) as value
FROM graph.your_edges_table
WHERE ft_depth = 0 AND ft_depth_sources IS NOT NULL
UNION ALL
SELECT
    'Edges with blocking_factor = 999' as metric,
    COUNT(*) as value
FROM graph.your_edges_table
WHERE blocking_factor >= 999
UNION ALL
SELECT
    'Edges with ft_depth NULL' as metric,
    COUNT(*) as value
FROM graph.your_edges_table
WHERE ft_depth IS NULL;
