"""
MLflow CLI module.

This module delegates to the system MLflow CLI to avoid conflicts between
the local mlflow module and the system-wide MLflow installation.
"""

import os
import sys
import subprocess

# The CLI object that will be imported by the mlflow command
cli = None

def main():
    """
    Main entry point for the MLflow CLI.
    This function delegates to the system MLflow CLI.
    """
    # Get the path to the system MLflow package
    try:
        # Try to find the system MLflow package
        import pkg_resources
        mlflow_path = pkg_resources.get_distribution('mlflow').location
        sys.path.insert(0, mlflow_path)
        
        # Import the system MLflow CLI
        from mlflow.cli import cli as system_cli
        global cli
        cli = system_cli
        return cli
    except (ImportError, pkg_resources.DistributionNotFound):
        # If the system MLflow package is not found, try to run mlflow as a subprocess
        print("Warning: System MLflow not found. Attempting to run as subprocess.")
        args = sys.argv[1:]
        cmd = ["python", "-m", "mlflow"] + args
        subprocess.run(cmd)
        sys.exit(0)

if __name__ == "__main__":
    main()