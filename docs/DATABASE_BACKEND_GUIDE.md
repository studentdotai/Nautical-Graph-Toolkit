# Database Backend Guidance

The Nautical Graph Toolkit supports three database backends for storing S-57 ENC data and navigation graphs. Choosing the right backend depends on your operational requirements, scale, and infrastructure.

**Before getting started:** See [SETUP.md](./SETUP.md) for software prerequisites, GDAL version requirements, and backend installation instructions.

## 1. PostGIS (Recommended for Production)
**Use when:** Building enterprise-grade applications, requiring multi-user concurrent access, handling large datasets (e.g., full NOAA catalog), or performing frequent incremental updates.

*   **Key Benefits:**
    *   **Concurrency:** Supports multiple users reading/writing simultaneously.
    *   **Performance:** Enables server-side processing for graph weighting and parallel processing for data imports.
    *   **Reliability:** Transactional integrity (ACID) ensures data safety during updates (S57Updater).
*   **Trade-offs:** Requires a running PostgreSQL instance and network connectivity.
*   **Requirements:** PostgreSQL with PostGIS extension (see [SETUP.md](./SETUP.md) for installation)

## 2. GeoPackage (Recommended for Portability)
**Use when:** Sharing data between teams, working offline, archiving project snapshots, or visualizing directly in GIS desktop software (QGIS, ArcGIS).

*   **Key Benefits:**
    *   **Portability:** Entire database is a single `.gpkg` file.
    *   **Compatibility:** Open OGC standard widely supported by GIS tools.
    *   **Simplicity:** No server setup required.
*   **Trade-offs:** Single-writer limitation (database locking); not suitable for concurrent updates.

## 3. SpatiaLite
**Use when:** Performing local development or requiring the fastest file-based conversion speeds.

*   **Key Benefits:**
    *   **Speed:** Often faster than GeoPackage for initial S-57 conversion.
    *   **Lightweight:** Minimal overhead for local testing.
*   **Trade-offs:** Less broad software support than GeoPackage; same single-writer limitations.

---

## Backend Selection Quick Reference

| Feature | PostGIS | GeoPackage | SpatiaLite |
| :--- | :--- | :--- | :--- |
| **Best For** | Production, Enterprise, Multi-user | Sharing, Offline, Visualization | Local Dev, Fast Conversion |
| **Setup** | Complex (Server required) | Zero (Single file) | Zero (Single file) |
| **Multi-user Access** | ✅ Excellent (Concurrent) | ❌ Poor (File locking) | ❌ Poor (File locking) |
| **Large Datasets** | ⭐ Best (Optimized) | ⭐ Good | ⭐ Good |
| **Incremental Updates** | ✅ Safe (Transactional) | ⚠️ Risky (Concurrency issues) | ⚠️ Risky (Concurrency issues) |
| **Parallel Processing** | ✅ Supported | ❌ Not Supported | ❌ Not Supported |
| **GIS Compatibility** | ✅ Excellent | ✅ Excellent (QGIS/ArcGIS) | ✅ Good |

---

## Getting Started

Once you've chosen your backend:

1. **Review prerequisites**: See [SETUP.md](./SETUP.md) for software requirements and installation steps
2. **Follow the workflow guide** for your backend:
   - **PostGIS**: [WORKFLOW_POSTGIS_GUIDE.md](./WORKFLOW_POSTGIS_GUIDE.md)
   - **GeoPackage**: [WORKFLOW_GEOPACKAGE_GUIDE.md](./WORKFLOW_GEOPACKAGE_GUIDE.md)
   - **S-57 Import Process**: [WORKFLOW_S57_IMPORT_GUIDE.md](./WORKFLOW_S57_IMPORT_GUIDE.md) (applies to all backends)
3. **Start with quickstart**: [WORKFLOW_QUICKSTART.md](./WORKFLOW_QUICKSTART.md) for first-time setup