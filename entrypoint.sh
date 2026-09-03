#!/bin/bash
set -e

mkdir -p /app/database

# If the database is missing, download the ZIP and extract it
if [ ! -f /app/database/rezpharma.db ] && [ -n "$DB_DOWNLOAD_URL" ]; then
    echo "⚠️ Database not found. Downloading zip archive..."
    wget -q --show-progress -O /app/database/rezpharma.zip "$DB_DOWNLOAD_URL"
    
    echo "📦 Extracting database..."
    python -c "import zipfile; zipfile.ZipFile('/app/database/rezpharma.zip').extractall('/app/database/')"
    
    # Clean up the zip to save space
    rm /app/database/rezpharma.zip
    echo "✅ Database extracted and ready."
fi

exec "$@"