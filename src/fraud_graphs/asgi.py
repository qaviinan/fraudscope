from __future__ import annotations

import os

from .api_server import create_app


app = create_app(os.getenv("FRAUD_API_ARTIFACTS_DIR", "outputs/backend_api/artifacts"))

