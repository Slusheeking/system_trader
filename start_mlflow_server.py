import sys
import os
import subprocess

# Ensure we're using the system MLflow, not the local one
sys.path = [p for p in sys.path if not p.endswith('system_trader')]

# Try running mlflow directly as a command
try:
    print("Starting MLflow server using Python module")
    subprocess.run([
        'python', '-m', 'mlflow.server',
        '--host', 'localhost',
        '--port', '5000',
        '--backend-store-uri', 'sqlite:///mlflow.db',
        '--default-artifact-root', './mlruns'
    ], check=True)
except Exception as e:
    print(f"Error running MLflow as module: {e}")
    
    # Try using the MLflow executable
    try:
        print("Trying with MLflow executable")
        mlflow_path = "/home/ubuntu/.local/bin/mlflow"
        print(f"Using MLflow at: {mlflow_path}")
        
        subprocess.run([
            mlflow_path, 'server',
            '--host', 'localhost',
            '--port', '5000',
            '--backend-store-uri', 'sqlite:///mlflow.db',
            '--default-artifact-root', './mlruns'
        ], check=True)
    except Exception as e:
        print(f"Error running MLflow executable: {e}")
        
        # Try running a simple Python script that uses the MLflow API
        try:
            print("Trying with direct MLflow API")
            
            # Create a temporary script
            with open('/tmp/run_mlflow_server.py', 'w') as f:
                f.write("""
import os
import sys
# Add the site-packages to the path
sys.path.insert(0, '/home/ubuntu/.local/lib/python3.10/site-packages')
# Remove the current directory from the path
sys.path = [p for p in sys.path if not p.endswith('system_trader')]

import mlflow
from mlflow.server import app

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
""")
            
            # Run the temporary script
            subprocess.run(['python3', '/tmp/run_mlflow_server.py'], check=True)
        except Exception as e:
            print(f"Error running MLflow with API: {e}")
            print("All methods failed to start MLflow")
