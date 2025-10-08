-- ============================================================================
-- VERIFICATION QUERY FOR USAGE BAND PRIORITIZATION
-- ============================================================================
-- This query shows edges where multiple ENCs overlap with different usage bands
-- and verifies that the highest usage band (most detailed chart) is selected
--
-- Expected behavior:
--   - If Band 2 (General) has depth=0m and Band 5 (Harbour) has depth=18m
--   - ft_depth should be 18m (from Band 5)
-- ============================================================================

-- Replace 'graph.your_edges_table' with your actual table name

WITH edges_with_multiple_bands AS (
    SELECT
        id,
        ft_depth,
        ft_depth_sources,
        jsonb_array_length(jsonb_object_keys_array(ft_depth_sources)) as num_sources
    FROM graph.your_edges_table
    WHERE ft_depth_sources IS NOT NULL
      AND jsonb_array_length(jsonb_object_keys_array(ft_depth_sources)) > 1
),
expanded_sources AS (
    SELECT
        e.id,
        e.ft_depth as selected_depth,
        e.num_sources,
        jsonb_object_keys(e.ft_depth_sources) as source_key,
        (e.ft_depth_sources -> jsonb_object_keys(e.ft_depth_sources) ->> 'depth')::float as source_depth,
        (e.ft_depth_sources -> jsonb_object_keys(e.ft_depth_sources) ->> 'usage_band')::int as usage_band
    FROM edges_with_multiple_bands e
),
ranked_sources AS (
    SELECT
        id,
        selected_depth,
        num_sources,
        source_key,
        source_depth,
        usage_band,
        ROW_NUMBER() OVER (PARTITION BY id ORDER BY usage_band DESC, source_depth ASC) as priority_rank
    FROM expanded_sources
)
SELECT
    id,
    selected_depth,
    num_sources,
    source_key as highest_priority_source,
    source_depth as highest_priority_depth,
    usage_band as highest_priority_band,
    CASE
        WHEN selected_depth = source_depth THEN '✓ CORRECT'
        ELSE '✗ WRONG (expected: ' || source_depth || 'm)'
    END as verification_status
FROM ranked_sources
WHERE priority_rank = 1
ORDER BY
    CASE
        WHEN selected_depth = source_depth THEN 1
        ELSE 0
    END ASC,  -- Show failures first
    id
LIMIT 50;

-- ============================================================================
-- USAGE BAND DISTRIBUTION
-- ============================================================================
-- Shows which usage bands are present in your data

SELECT
    usage_band,
    CASE usage_band
        WHEN 1 THEN 'Overview'
        WHEN 2 THEN 'General'
        WHEN 3 THEN 'Coastal'
        WHEN 4 THEN 'Approach'
        WHEN 5 THEN 'Harbour'
        WHEN 6 THEN 'Berthing'
        ELSE 'Unknown'
    END as band_name,
    COUNT(DISTINCT id) as edge_count
FROM (
    SELECT
        e.id,
        (e.ft_depth_sources -> jsonb_object_keys(e.ft_depth_sources) ->> 'usage_band')::int as usage_band
    FROM graph.your_edges_table e
    WHERE ft_depth_sources IS NOT NULL
) sub
GROUP BY usage_band
ORDER BY usage_band DESC;

-- ============================================================================
-- EXAMPLE: Edges with conflicting depths across usage bands
-- ============================================================================
-- Shows cases where General/Coastal charts have depth=0 but Harbour has depth>0

WITH source_depths AS (
    SELECT
        e.id,
        e.ft_depth,
        (e.ft_depth_sources -> jsonb_object_keys(e.ft_depth_sources) ->> 'depth')::float as source_depth,
        (e.ft_depth_sources -> jsonb_object_keys(e.ft_depth_sources) ->> 'usage_band')::int as usage_band
    FROM graph.your_edges_table e
    WHERE ft_depth_sources IS NOT NULL
),
aggregated_by_edge AS (
    SELECT
        id,
        ft_depth,
        MIN(CASE WHEN usage_band IN (1, 2, 3) THEN source_depth END) as low_band_depth,
        MAX(CASE WHEN usage_band IN (4, 5, 6) THEN source_depth END) as high_band_depth,
        MAX(usage_band) as max_band,
        MIN(usage_band) as min_band
    FROM source_depths
    GROUP BY id, ft_depth
    HAVING COUNT(DISTINCT usage_band) > 1
)
SELECT
    id,
    ft_depth as selected_depth,
    low_band_depth as general_coastal_depth,
    high_band_depth as approach_harbour_depth,
    min_band as lowest_band,
    max_band as highest_band,
    CASE
        WHEN ft_depth = high_band_depth THEN '✓ Using high-detail chart'
        WHEN ft_depth = low_band_depth THEN '✗ Using low-detail chart (WRONG!)'
        ELSE '? Unknown selection'
    END as priority_check
FROM aggregated_by_edge
WHERE low_band_depth = 0 AND high_band_depth > 0  -- Cases where general chart shows 0 but harbour shows depth
ORDER BY
    CASE
        WHEN ft_depth = high_band_depth THEN 1
        ELSE 0
    END ASC  -- Show problems first
LIMIT 20;
