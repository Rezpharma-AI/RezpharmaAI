#!/bin/bash
set -e

# Ensure database directory exists
mkdir -p /app/database

# Download database if missing and URL is provided
if [ ! -f /app/database/rezpharma.db ] && [ -n "$DB_DOWNLOAD_URL" ]; then
    echo "⚠️ Database not found in volume. Downloading from cloud storage..."
    wget -q --show-progress -O /app/database/rezpharma.db "$DB_DOWNLOAD_URL"
    echo "✅ Database download complete."
fi

# Execute the main command (passed by Docker/Railway)
exec "$@"