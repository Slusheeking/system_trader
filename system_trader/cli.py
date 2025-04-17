"""
Command-line interface for System Trader
"""

import argparse
import sys


def main():
    """
    Main entry point for the CLI
    """
    parser = argparse.ArgumentParser(
        description="System Trader - An autonomous algorithmic trading system"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest")
    backtest_parser.add_argument(
        "--strategy", type=str, required=True, help="Strategy to backtest"
    )
    backtest_parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    
    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live trading")
    live_parser.add_argument(
        "--strategy", type=str, required=True, help="Strategy to run"
    )
    live_parser.add_argument(
        "--paper", action="store_true", help="Use paper trading"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "backtest":
        print(f"Running backtest for strategy {args.strategy}")
        print(f"Period: {args.start_date} to {args.end_date}")
        # TODO: Implement backtest functionality
    elif args.command == "live":
        mode = "paper" if args.paper else "live"
        print(f"Running {mode} trading with strategy {args.strategy}")
        # TODO: Implement live trading functionality
    
    return 0


if __name__ == "__main__":
    sys.exit(main())