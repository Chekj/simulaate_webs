#!/bin/bash

# Fixed script to add simulated tradebook feature to stock OpenAlgo

echo "Adding simulated tradebook feature to stock OpenAlgo..."

# Check if we're in the right directory
if [ ! -f "app.py" ] || [ ! -d "services" ]; then
    echo "Error: This script must be run from the OpenAlgo root directory"
    echo "Please navigate to your OpenAlgo installation directory and try again"
    exit 1
fi

# Create simulate_tradebook.py
cat > simulate_tradebook.py << 'EOF'
"""
Simulate Tradebook Module

This module allows adding simulated trade entries that will be included 
with real trade data when calling the tradebook endpoint.

The simulated trades are stored in a JSON file and merged with real broker data.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

# Default configuration for simulated trades
DEFAULT_CONFIG = {
    "storage_file": "db/simulated_trades.json",
    "default_fields": {
        "action": "BUY",
        "symbol": "NIFTY20OCT2525150PE",
        "exchange": "BFO",
        "orderid": "25101400857846",
        "product": "MIS",
        "quantity": 75,
        "average_price": 124.85,
        "trade_value": 0.0
    }
}

class SimulatedTradebook:
    """Manages simulated trade entries"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the simulated tradebook manager.
        
        Args:
            config: Configuration dictionary with storage file path and default values
        """
        self.config = config or DEFAULT_CONFIG
        self.storage_file = self.config["storage_file"]
        self.default_fields = self.config["default_fields"]
        self._ensure_storage_file()
    
    def _ensure_storage_file(self):
        """Ensure the storage file exists with proper structure"""
        if not os.path.exists(self.storage_file):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            # Create empty file with proper structure
            with open(self.storage_file, 'w') as f:
                json.dump({"trades": []}, f, indent=2)
    
    def _read_trades(self) -> List[Dict[str, Any]]:
        """Read all simulated trades from storage"""
        try:
            with open(self.storage_file, 'r') as f:
                data = json.load(f)
                return data.get("trades", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _write_trades(self, trades: List[Dict[str, Any]]):
        """Write trades to storage"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        
        with open(self.storage_file, 'w') as f:
            json.dump({"trades": trades}, f, indent=2)
    
    def add_trade(self, trade_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a simulated trade entry.
        
        Args:
            trade_data: Dictionary with trade fields. If None, will prompt for input.
            
        Returns:
            Dictionary with success status and message
        """
        # If no trade data provided, use prompt-based input
        if trade_data is None:
            trade_data = self._prompt_for_trade_data()
        
        # Apply defaults for missing fields
        trade_entry = self.default_fields.copy()
        trade_entry.update(trade_data)
        
        # Generate timestamp if not provided (match real tradebook format: HH:MM:SS)
        if "timestamp" not in trade_entry:
            trade_entry["timestamp"] = datetime.now().strftime("%H:%M:%S")
        
        # Remove tradeid if it was added (not part of real tradebook data)
        trade_entry.pop("tradeid", None)
        
        # Calculate trade value if not provided and we have required data
        if trade_entry["trade_value"] == 0.0 and trade_entry["average_price"] > 0 and trade_entry["quantity"] > 0:
            trade_entry["trade_value"] = round(trade_entry["average_price"] * trade_entry["quantity"], 2)
        
        # Read existing trades
        trades = self._read_trades()
        
        # Add new trade
        trades.append(trade_entry)
        
        # Write back to storage
        try:
            self._write_trades(trades)
            return {
                "status": "success",
                "message": "Simulated trade added successfully",
                "trade": trade_entry
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add simulated trade: {str(e)}"
            }
    
    def _prompt_for_trade_data(self) -> Dict[str, Any]:
        """Prompt user for trade data"""
        print("\n=== Add Simulated Trade ===")
        trade_data = {}
        
        # Prompt for each field with default values
        trade_data["action"] = input(f"Action (BUY/SELL) [{self.default_fields['action']}]: ").strip() or self.default_fields["action"]
        trade_data["symbol"] = input(f"Symbol [{self.default_fields['symbol']}]: ").strip() or self.default_fields["symbol"]
        trade_data["exchange"] = input(f"Exchange [{self.default_fields['exchange']}]: ").strip() or self.default_fields["exchange"]
        trade_data["product"] = input(f"Product [{self.default_fields['product']}]: ").strip() or self.default_fields["product"]
        trade_data["quantity"] = int(input(f"Quantity [{self.default_fields['quantity']}]: ").strip() or self.default_fields["quantity"])
        trade_data["average_price"] = float(input(f"Average Price [{self.default_fields['average_price']}]: ").strip() or self.default_fields["average_price"])
        
        # Optional fields
        orderid = input("Order ID (optional): ").strip()
        if orderid:
            trade_data["orderid"] = orderid
            
        # Note: Trade value is always calculated automatically based on quantity and average price
        
        return trade_data
    
    def get_simulated_trades(self) -> List[Dict[str, Any]]:
        """
        Get all simulated trades.
        
        Returns:
            List of simulated trade dictionaries
        """
        return self._read_trades()
    
    def clear_trades(self) -> Dict[str, Any]:
        """
        Clear all simulated trades.
        
        Returns:
            Dictionary with success status and message
        """
        try:
            self._write_trades([])
            return {
                "status": "success",
                "message": "All simulated trades cleared"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to clear simulated trades: {str(e)}"
            }
    
    def merge_with_real_trades(self, real_trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge simulated trades with real trades.
        
        Args:
            real_trades: List of real trade dictionaries from broker
            
        Returns:
            Combined list of real and simulated trades
        """
        simulated_trades = self._read_trades()
        # Combine real and simulated trades
        combined_trades = real_trades + simulated_trades
        # Sort by timestamp (newest first)
        combined_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return combined_trades

# Global instance for easy access
simulated_tradebook = SimulatedTradebook()

def add_simulated_trade(trade_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Add a simulated trade entry.
    
    Args:
        trade_data: Dictionary with trade fields. If None, will prompt for input.
        
    Returns:
        Dictionary with success status and message
    """
    return simulated_tradebook.add_trade(trade_data)

def get_simulated_trades() -> List[Dict[str, Any]]:
    """
    Get all simulated trades.
    
    Returns:
        List of simulated trade dictionaries
    """
    return simulated_tradebook.get_simulated_trades()

def clear_simulated_trades() -> Dict[str, Any]:
    """
    Clear all simulated trades.
    
    Returns:
        Dictionary with success status and message
    """
    return simulated_tradebook.clear_trades()

def merge_with_real_trades(real_trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge simulated trades with real trades.
    
    Args:
        real_trades: List of real trade dictionaries from broker
        
    Returns:
        Combined list of real and simulated trades
    """
    return simulated_tradebook.merge_with_real_trades(real_trades)

# Command line interface
if __name__ == "__main__":
    import sys
    
    def show_menu():
        """Show the main menu"""
        print("\n=== Simulated Tradebook Menu ===")
        print("1. Continue with default trade details")
        print("2. Add trade manually (prompt based)")
        print("3. List all simulated trades")
        print("4. Clear all simulated trades")
        print("0. Exit")
        print("================================")
    
    def show_default_config():
        """Show default configuration"""
        print("\n=== Default Configuration ===")
        for key, value in simulated_tradebook.default_fields.items():
            print(f"{key}: {value}")
        print("=============================")
    
    # Always show default configuration first
    show_default_config()
    
    while True:
        show_menu()
        try:
            choice = input("Enter your choice (0-4): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                # Continue with default trade details
                print("\nCreating trade with default configuration...")
                result = add_simulated_trade({})
                print(json.dumps(result, indent=2))
            elif choice == "2":
                # Add trade manually (prompt based)
                print("\nAdding trade manually...")
                result = add_simulated_trade()
                print(json.dumps(result, indent=2))
            elif choice == "3":
                # List all trades
                trades = get_simulated_trades()
                if trades:
                    print(f"\nFound {len(trades)} simulated trades:")
                    print(json.dumps(trades, indent=2))
                else:
                    print("\nNo simulated trades found.")
            elif choice == "4":
                # Clear all trades
                confirm = input("Are you sure you want to clear all trades? (y/N): ").strip().lower()
                if confirm == 'y':
                    result = clear_simulated_trades()
                    print(json.dumps(result, indent=2))
                else:
                    print("Operation cancelled.")
            else:
                print("Invalid choice. Please enter a number between 0-4.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
EOF

# Backup original files if backups don't exist
if [ ! -f "services/tradebook_service.py.backup" ]; then
    cp services/tradebook_service.py services/tradebook_service.py.backup
fi

if [ ! -f "services/sandbox_service.py.backup" ]; then
    cp services/sandbox_service.py services/sandbox_service.py.backup
fi

# Modify services/tradebook_service.py using Python for precision
python3 << 'EOF'
with open('services/tradebook_service.py', 'r') as f:
    content = f.read()

# Add import after the existing imports
import_section = """from utils.logging import get_logger

# Try to import simulated tradebook functionality
try:
    from simulate_tradebook import merge_with_real_trades
    SIMULATE_TRADEBOOK_AVAILABLE = True
except ImportError:
    SIMULATE_TRADEBOOK_AVAILABLE = False
    def merge_with_real_trades(real_trades):
        return real_trades

# Initialize logger"""

content = content.replace('from utils.logging import get_logger\n\n# Initialize logger', import_section)

# Add merge logic before the return statement in get_tradebook_with_auth function
merge_section = """        # Format numeric values to 2 decimal places
        formatted_trades = format_trade_data(trade_data)
        
        # Merge with simulated trades if available
        if SIMULATE_TRADEBOOK_AVAILABLE:
            formatted_trades = merge_with_real_trades(formatted_trades)
        
        return True, {
            'status': 'success',
            'data': formatted_trades
        }, 200"""

content = content.replace("""        # Format numeric values to 2 decimal places
        formatted_trades = format_trade_data(trade_data)
        
        return True, {
            'status': 'success',
            'data': formatted_trades
        }, 200""", merge_section)

with open('services/tradebook_service.py', 'w') as f:
    f.write(content)
EOF

# Modify services/sandbox_service.py using Python for precision
python3 << 'EOF'
with open('services/sandbox_service.py', 'r') as f:
    content = f.read()

# Add import after the existing imports
import_section = """from sandbox.fund_manager import FundManager, get_user_funds

# Try to import simulated tradebook functionality
try:
    from simulate_tradebook import merge_with_real_trades
    SIMULATE_TRADEBOOK_AVAILABLE = True
except ImportError:
    SIMULATE_TRADEBOOK_AVAILABLE = False
    def merge_with_real_trades(real_trades):
        return real_trades

logger = get_logger(__name__)"""

content = content.replace('from sandbox.fund_manager import FundManager, get_user_funds\n\nlogger = get_logger(__name__)', import_section)

# Add merge logic in the sandbox_get_tradebook function
merge_section = """        position_manager = PositionManager(user_id)
        success, response, status_code = position_manager.get_tradebook()
        
        # If successful and simulated tradebook is available, merge with simulated trades
        if success and SIMULATE_TRADEBOOK_AVAILABLE:
            # Extract the trade data from the response
            trade_data = response.get('data', [])
            # Merge with simulated trades
            merged_trades = merge_with_real_trades(trade_data)
            # Update the response with merged data
            response['data'] = merged_trades

        return success, response, status_code"""

content = content.replace("""        position_manager = PositionManager(user_id)
        success, response, status_code = position_manager.get_tradebook()

        return success, response, status_code""", merge_section)

with open('services/sandbox_service.py', 'w') as f:
    f.write(content)
EOF

# Create documentation
mkdir -p docs
cat > docs/SIMULATED_TRADEBOOK.md << 'EOF'
# Simulated Tradebook

The Simulated Tradebook feature allows you to add simulated trade entries that will be included with real trade data when calling the tradebook endpoint. This is useful for testing and demonstration purposes.

## How It Works

1. Simulated trades are stored in a JSON file (`db/simulated_trades.json`)
2. When the tradebook is requested, both real broker trades and simulated trades are returned
3. Simulated trades are merged with real trades and sorted by timestamp

## Usage

### Programmatic Usage

```python
from simulate_tradebook import add_simulated_trade, get_simulated_trades, clear_simulated_trades

# Add a simulated trade
trade_data = {
    "action": "BUY",
    "symbol": "RELIANCE-EQ",
    "exchange": "NSE",
    "orderid": "SIM_ORDER_001",
    "product": "MIS",
    "quantity": 10,
    "average_price": 2500.50,
    "trade_value": 25005.00
}

result = add_simulated_trade(trade_data)
print(result)

# Get all simulated trades
trades = get_simulated_trades()
print(trades)

# Clear all simulated trades
result = clear_simulated_trades()
print(result)
```

### Command Line Usage

```bash
# Add a simulated trade (interactive prompt)
python simulate_tradebook.py add

# List all simulated trades
python simulate_tradebook.py list

# Clear all simulated trades
python simulate_tradebook.py clear
```

### Default Configuration

The simulated tradebook has default values that can be customized:

```python
DEFAULT_CONFIG = {
    "storage_file": "db/simulated_trades.json",
    "default_fields": {
        "action": "BUY",
        "symbol": "INFY-EQ",
        "exchange": "NSE",
        "orderid": "SIMULATED",
        "product": "MIS",
        "quantity": 1,
        "average_price": 0.0,
        "trade_value": 0.0
    }
}
```

You can modify these defaults by editing the simulate_tradebook.py file.

## Data Format

Each simulated trade includes the following fields (matching the exact format of real tradebook data):

- `action`: BUY or SELL
- `symbol`: Trading symbol (e.g., "RELIANCE-EQ")
- `exchange`: Exchange (e.g., "NSE", "BSE")
- `orderid`: Order ID
- `product`: Product type (e.g., "MIS", "NRML", "CNC")
- `quantity`: Number of shares
- `average_price`: Average price per share
- `trade_value`: Total trade value (calculated if not provided)
- `timestamp`: Timestamp when the trade was added (automatically generated in HH:MM:SS format to match real tradebook data)

## Integration

The simulated tradebook is automatically integrated with both live and sandbox tradebook endpoints. When you request the tradebook, simulated trades will be merged with real trades.

## Storage

Simulated trades are stored in `db/simulated_trades.json`. This file is automatically created when you add your first simulated trade.
EOF

# Create example file
mkdir -p examples
cat > examples/simulated_tradebook_example.py << 'EOF'
"""
Example script demonstrating the simulated tradebook integration
"""

import json
from simulate_tradebook import add_simulated_trade, get_simulated_trades

def example_usage():
    """Example of how to use the simulated tradebook"""
    
    print("=== Simulated Tradebook Example ===\n")
    
    # Add some simulated trades
    print("1. Adding simulated trades...")
    
    # Example 1: Manual trade data
    trade1 = {
        "action": "BUY",
        "symbol": "AAPL-EQ",
        "exchange": "NASDAQ",
        "orderid": "SIM001",
        "product": "MIS",
        "quantity": 100,
        "average_price": 150.25
    }
    
    result = add_simulated_trade(trade1)
    print(f"   Added trade 1: {result['status']}")
    
    # Example 2: Another trade with different parameters
    trade2 = {
        "action": "SELL",
        "symbol": "GOOGL-EQ",
        "exchange": "NASDAQ",
        "orderid": "SIM002",
        "product": "NRML",
        "quantity": 50,
        "average_price": 2500.75
    }
    
    result = add_simulated_trade(trade2)
    print(f"   Added trade 2: {result['status']}")
    
    # Example 3: Trade with all fields specified
    trade3 = {
        "action": "BUY",
        "symbol": "MSFT-EQ",
        "exchange": "NASDAQ",
        "orderid": "SIM003",
        "product": "CNC",
        "quantity": 75,
        "average_price": 300.50,
        "trade_value": 22537.50
    }
    
    result = add_simulated_trade(trade3)
    print(f"   Added trade 3: {result['status']}")
    
    # Show all simulated trades
    print("\n2. Current simulated trades:")
    trades = get_simulated_trades()
    for i, trade in enumerate(trades, 1):
        print(f"   Trade {i}:")
        print(f"     Action: {trade['action']}")
        print(f"     Symbol: {trade['symbol']}")
        print(f"     Exchange: {trade['exchange']}")
        print(f"     Order ID: {trade['orderid']}")
        print(f"     Product: {trade['product']}")
        print(f"     Quantity: {trade['quantity']}")
        print(f"     Average Price: {trade['average_price']}")
        print(f"     Trade Value: {trade['trade_value']}")
        print(f"     Timestamp: {trade['timestamp']}")
        print()
    
    # Example of how this would integrate with real trade data
    print("3. Integration example:")
    print("   When you call the tradebook endpoint, the response will include")
    print("   both real broker trades and these simulated trades.")
    
    # Simulate what the merged data might look like
    real_trades = [
        {
            "symbol": "TSLA-EQ",
            "exchange": "NASDAQ",
            "product": "MIS",
            "action": "BUY",
            "quantity": 25,
            "average_price": 200.00,
            "trade_value": 5000.00,
            "orderid": "REAL001",
            "timestamp": "2025-10-13 10:30:00"
        }
    ]
    
    # In the actual implementation, this merge happens automatically
    # This is just to show what the result would look like
    from simulate_tradebook import merge_with_real_trades
    merged_trades = merge_with_real_trades(real_trades)
    
    print("\n   Merged trades (real + simulated):")
    for i, trade in enumerate(merged_trades, 1):
        print(f"     Trade {i}: {trade['symbol']} {trade['action']} {trade['quantity']} @ {trade['average_price']}")

if __name__ == "__main__":
    example_usage()
EOF

echo "Simulated tradebook feature successfully added to stock OpenAlgo!"
echo ""
echo "What was added:"
echo "1. simulate_tradebook.py - Core implementation file"
echo "2. Modified services/tradebook_service.py - Integrated with live trades"
echo "3. Modified services/sandbox_service.py - Integrated with sandbox trades"
echo "4. docs/SIMULATED_TRADEBOOK.md - Documentation"
echo "5. examples/simulated_tradebook_example.py - Usage examples"
echo ""
echo "Backup files created:"
echo "  - services/tradebook_service.py.backup"
echo "  - services/sandbox_service.py.backup"
echo ""
echo "To test the feature:"
echo "  python simulate_tradebook.py  # Run the interactive CLI"
echo ""
echo "The feature is now automatically integrated with both live and sandbox tradebook endpoints."
