#!/usr/bin/env python3
"""
Startup script for the registry API server with HTTPS support.
This script is designed to be called by systemd or run directly.
"""
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import uvicorn

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    # Define SSL certificate paths
    ssl_keyfile = script_dir / "ssl_certs" / "key.pem"
    ssl_certfile = script_dir / "ssl_certs" / "cert.pem"

    # Check if SSL certificates exist
    use_ssl = ssl_keyfile.exists() and ssl_certfile.exists()

    if use_ssl:
        logger.info("🔒 Starting server with HTTPS...")
        logger.info(f"   SSL Key: {ssl_keyfile}")
        logger.info(f"   SSL Cert: {ssl_certfile}")
        logger.info("   Access at: https://0.0.0.0:8000")

        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            workers=1,
            timeout_keep_alive=30,
            log_level="info",
            ssl_keyfile=str(ssl_keyfile),
            ssl_certfile=str(ssl_certfile),
        )
    else:
        logger.warning("⚠️  SSL certificates not found - starting with HTTP")
        logger.info(f"   Looking for certificates at: {ssl_certfile.parent}")
        logger.info("   Access at: http://0.0.0.0:8000")

        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            workers=1,
            timeout_keep_alive=30,
            log_level="info",
        )


if __name__ == "__main__":
    main()
