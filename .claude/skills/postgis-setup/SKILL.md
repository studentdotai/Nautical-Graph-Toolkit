---
name: postgis-setup
description: Creating and configuring PostGIS database for ENC storage and maritime routing. Use when setting up PostGIS backend, deploying to production, or optimizing database performance for large-scale conversions.
allowed-tools: [Bash]
status: Under Development
ready: false
---

⚠️ **STATUS: Under Development** - Not yet ready for use. This skill requires completion of production features (connection pooling, SSL/TLS, monitoring) before it can be used in production environments. Basic setup works, but advanced deployment scenarios are incomplete.

# PostGIS Database Setup

Creating and configuring PostGIS database for ENC storage and maritime routing applications.

## Quick Start (System PostgreSQL)

**For local development** with existing PostgreSQL:

```bash
# 1. Create database and enable PostGIS
createdb maritime_db
psql maritime_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# 2. Create .env file
cat > .env <<EOF
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=$USER
POSTGRES_DB=maritime_db
EOF

# 3. Verify
psql maritime_db -c "SELECT PostGIS_Version();"
```

**Expected output**: `3.4 USE_GEOS=1 USE_PROJ=1 USE_STATS=1` (or similar)

## Purpose

PostGIS provides the fastest backend for large-scale ENC processing (2.0-2.4× faster than GeoPackage). This skill covers complete PostGIS setup from database creation through connection configuration.

## Prerequisites

- PostgreSQL 16+ installed
- System permissions to create databases
- Basic SQL knowledge
- Python environment with psycopg2-binary

## Procedure

### Step 1: Create PostgreSQL Database

```bash
# Create database
createdb maritime_db

# Verify creation
psql -l | grep maritime_db
```

### Step 2: Enable PostGIS Extension

```bash
# Enable PostGIS
psql maritime_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# Verify PostGIS version
psql maritime_db -c "SELECT PostGIS_Version();"
```

Expected output:
```
            postgis_version
---------------------------------------
 3.4 USE_GEOS=1 USE_PROJ=1 USE_STATS=1
```

### Step 3: Create User (Optional but Recommended)

```bash
# Create dedicated user for application
psql -c "CREATE USER maritime_user WITH PASSWORD 'secure_password';"

# Grant privileges
psql -c "GRANT ALL PRIVILEGES ON DATABASE maritime_db TO maritime_user;"

# Grant schema privileges
psql maritime_db -c "GRANT ALL ON SCHEMA public TO maritime_user;"
psql maritime_db -c "GRANT ALL ON ALL TABLES IN SCHEMA public TO maritime_user;"
psql maritime_db -c "GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO maritime_user;"
```

### Step 4: Configure Environment Variables

Create `.env` file in project root:

```bash
cat > .env <<'EOF'
# PostGIS Connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=maritime_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=maritime_db
EOF

# Secure .env file
chmod 600 .env
```

### Step 5: Test Connection

```bash
# Test with psql
psql "host=localhost port=5432 dbname=maritime_db user=maritime_user password=secure_password" \
    -c "SELECT PostGIS_Version();"

# Test with Python
python -c "
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST'),
    port=os.getenv('POSTGRES_PORT'),
    dbname=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD')
)
print('✓ Connection successful!')
conn.close()
"
```

## Examples

### Example 1: Fresh Ubuntu Installation

```bash
# Install PostgreSQL and PostGIS
sudo apt-get update
sudo apt-get install postgresql-16 postgresql-16-postgis-3

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database as postgres user
sudo -u postgres createdb maritime_db
sudo -u postgres psql maritime_db -c "CREATE EXTENSION postgis;"

# Create application user
sudo -u postgres psql -c "CREATE USER maritime_user WITH PASSWORD 'secure_pass';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE maritime_db TO maritime_user;"

# Configure .env and test
```

### Example 2: macOS with Homebrew

```bash
# Install PostgreSQL and PostGIS
brew install postgresql@16 postgis

# Start PostgreSQL
brew services start postgresql@16

# Create database
createdb maritime_db
psql maritime_db -c "CREATE EXTENSION postgis;"

# Create user
psql -c "CREATE USER maritime_user WITH PASSWORD 'secure_pass';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE maritime_db TO maritime_user;"

# Configure .env and test
```

### Example 3: Docker Deployment

```bash
# Run PostGIS container
docker run -d \
    --name postgis \
    -e POSTGRES_DB=maritime_db \
    -e POSTGRES_USER=maritime_user \
    -e POSTGRES_PASSWORD=secure_pass \
    -p 5432:5432 \
    postgis/postgis:16-3.4

# Wait for startup
sleep 10

# Verify
docker exec postgis psql -U maritime_user -d maritime_db -c "SELECT PostGIS_Version();"
```

## Connection String Formats

### Format 1: Environment Variables (Recommended)

```python
import os
from dotenv import load_dotenv

load_dotenv()

db_config = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD")
}
```

### Format 2: GDAL Connection String

```python
# For GDAL/OGR operations
conn_str = (
    f"PG:dbname={os.getenv('POSTGRES_DB')} "
    f"host={os.getenv('POSTGRES_HOST')} "
    f"port={os.getenv('POSTGRES_PORT')} "
    f"user={os.getenv('POSTGRES_USER')} "
    f"password={os.getenv('POSTGRES_PASSWORD')}"
)
```

### Format 3: SQLAlchemy Engine

```python
from sqlalchemy import create_engine

engine = create_engine(
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)
```

## Performance Tuning

### Basic PostgreSQL Tuning

Edit `postgresql.conf` (location varies by system):

```ini
# Memory settings (adjust based on system RAM)
shared_buffers = 256MB          # 25% of RAM (for 1GB system)
effective_cache_size = 1GB      # 50-75% of RAM
work_mem = 16MB                 # Per-operation memory
maintenance_work_mem = 128MB    # For VACUUM, CREATE INDEX

# Connection settings
max_connections = 100

# Write-ahead log
wal_buffers = 16MB
checkpoint_completion_target = 0.9
```

Restart PostgreSQL:
```bash
sudo systemctl restart postgresql  # Linux
brew services restart postgresql@16  # macOS
```

### Spatial Index Optimization

```sql
-- Create spatial index on geometry column
CREATE INDEX idx_geom ON your_table USING GIST (geom);

-- Analyze table for query planner
ANALYZE your_table;
```

## Common Issues

### Issue: Permission Denied

**Symptom**: `psycopg2.OperationalError: FATAL: permission denied`

**Solution**:
```bash
# Grant necessary permissions
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE maritime_db TO maritime_user;"
sudo -u postgres psql maritime_db -c "GRANT ALL ON SCHEMA public TO maritime_user;"
sudo -u postgres psql maritime_db -c "GRANT ALL ON ALL TABLES IN SCHEMA public TO maritime_user;"
```

### Issue: Connection Refused

**Symptom**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql  # Linux
brew services list  # macOS

# Start if needed
sudo systemctl start postgresql  # Linux
brew services start postgresql@16  # macOS

# Check pg_hba.conf allows local connections
sudo cat /etc/postgresql/16/main/pg_hba.conf | grep local
# Should have: local all all trust (or md5/peer)
```

### Issue: PostGIS Extension Not Found

**Symptom**: `ERROR: could not open extension control file`

**Solution**:
```bash
# Install PostGIS package
sudo apt-get install postgresql-16-postgis-3  # Ubuntu/Debian
brew install postgis  # macOS

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### Issue: Role Does Not Exist

**Symptom**: `psycopg2.OperationalError: FATAL: role "maritime_user" does not exist`

**Solution**:
```bash
# Create user
sudo -u postgres psql -c "CREATE USER maritime_user WITH PASSWORD 'secure_pass';"

# Verify
sudo -u postgres psql -c "\du" | grep maritime_user
```

## Verification Checklist

Run these commands to verify your PostGIS setup:

```bash
# [ ] PostgreSQL version
psql --version  # Should be 16+

# [ ] Database exists
psql -l | grep maritime_db

# [ ] PostGIS extension enabled
psql maritime_db -c "SELECT PostGIS_Version();"

# [ ] User has privileges (if using dedicated user)
psql -c "\du" | grep maritime_user

# [ ] .env file exists and is secure
ls -la .env  # Should show -rw------- (600 permissions)

# [ ] Python connection works
python -c "
import os
from dotenv import load_dotenv
import psycopg2
load_dotenv()
conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST'),
    port=os.getenv('POSTGRES_PORT'),
    dbname=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD')
)
print('✓ PostGIS connection successful!')
conn.close()
"

# [ ] GDAL can connect (if using GDAL)
ogrinfo "PG:dbname=maritime_db host=localhost" -q
```

## Maintenance Tasks

### Vacuum and Analyze

```bash
# Regular maintenance
psql maritime_db -c "VACUUM ANALYZE;"

# Full vacuum (locks tables, run during low usage)
psql maritime_db -c "VACUUM FULL ANALYZE;"
```

### Backup and Restore

```bash
# Backup
pg_dump maritime_db > maritime_db_backup.sql

# Restore
createdb maritime_db_restored
psql maritime_db_restored < maritime_db_backup.sql
```

## Related Skills

- **environment-setup**: Environment Setup (PostgreSQL installation)
- **backend-optimization**: Backend Optimization (PostGIS vs GeoPackage performance)
- **integration-tests**: Integration Tests (requires PostGIS for backend tests)

## Cross-References

- **Project Knowledge**: `/dev/rules/CLAUDE.md` (Database Backend Patterns section)
- **Development Workflow**: `/dev/rules/WORKFLOW.md` (PostGIS Setup section)
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md` (Environment Variables section)