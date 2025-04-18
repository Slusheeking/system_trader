import sys
import os
import subprocess
import time

# Ensure we're using the system MLflow, not the local one
sys.path = [p for p in sys.path if not p.endswith('system_trader')]

try:
    # Try to import mlflow directly
    import mlflow.server
    from mlflow.server import app
    
    # Start the server
    app.run(host='localhost', port=5000, backend_store_uri='sqlite:///mlflow.db', default_artifact_root='./mlruns')
except ImportError:
    # Fall back to subprocess if direct import fails
    subprocess.run([
        'python', '-m', 'mlflow', 'server',
        '--host', 'localhost',
        '--port', '5000',
        '--backend-store-uri', 'sqlite:///mlflow.db',
        '--default-artifact-root', './mlruns'
    ])
