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
import uuid

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
