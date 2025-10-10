#!/bin/bash

# simulated_websocket_fixed.sh - Fixed installation script for simulated WebSocket data in OpenAlgo
# This script automatically installs and configures the simulated WebSocket feature
# for OpenAlgo analyzer mode in a fresh clone from GitHub.
# sudo chmod +x simulated_websocket2_fixed.sh
# sudo ./simulated_websocket2_fixed.sh

set -e  # Exit on any error

echo "=========================================="
echo "OpenAlgo Simulated WebSocket Implementation (FIXED)"
echo "=========================================="
echo

# Check if we're in the correct directory (OpenAlgo root)
if [ ! -f "app.py" ] || [ ! -d "websocket_proxy" ]; then
    echo "Error: This script must be run from the OpenAlgo root directory"
    echo "Please navigate to your OpenAlgo installation directory and try again"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "Installing simulated WebSocket feature for OpenAlgo..."
echo

# Create the simulated WebSocket adapter file
echo "Creating simulated WebSocket adapter..."
cat > websocket_proxy/simulated_adapter.py << 'EOF'
"""
Simulated WebSocket Adapter for OpenAlgo Analyzer Mode

This adapter provides simulated market data when analyzer mode is enabled,
allowing users to test WebSocket functionality without connecting to real brokers.
"""

import json
import random
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any
from .base_adapter import BaseBrokerWebSocketAdapter
from utils.logging import get_logger

logger = get_logger(__name__)

class SimulatedWebSocketAdapter(BaseBrokerWebSocketAdapter):
    """
    Simulated WebSocket adapter that generates realistic market data
    for testing and analysis purposes.
    """
    
    def __init__(self):
        """Initialize the simulated WebSocket adapter"""
        super().__init__()
        self.simulation_thread = None
        self.simulation_running = False
        self.subscribed_symbols = {}  # Track subscribed symbols and their data
        
        # Simulated market data for common symbols
        self.symbols_data = {
            "RELIANCE": {"base_price": 2500.0, "base_volume": 100000},
            "TCS": {"base_price": 3800.0, "base_volume": 50000},
            "INFY": {"base_price": 1500.0, "base_volume": 75000},
            "HDFCBANK": {"base_price": 1600.0, "base_volume": 120000},
            "ICICIBANK": {"base_price": 950.0, "base_volume": 90000},
            "SBIN": {"base_price": 600.0, "base_volume": 80000},
            "BHARTIARTL": {"base_price": 1100.0, "base_volume": 60000},
            "LT": {"base_price": 3200.0, "base_volume": 40000},
            "AXISBANK": {"base_price": 1100.0, "base_volume": 70000},
            "ASIANPAINT": {"base_price": 3300.0, "base_volume": 30000}
        }
        
        logger.info("SimulatedWebSocketAdapter initialized")
    
    def initialize(self, broker_name: str, user_id: str, auth_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Initialize the simulated adapter
        
        Args:
            broker_name: Name of the broker
            user_id: User ID
            auth_data: Authentication data (not used in simulation)
            
        Returns:
            Dict with initialization status
        """
        logger.info(f"Initializing simulated adapter for user {user_id}")
        return self._create_success_response("Simulated adapter initialized successfully")
    
    def connect(self) -> Dict[str, Any]:
        """
        Connect to the simulated WebSocket (no actual connection needed)
        
        Returns:
            Dict with connection status
        """
        logger.info("Connecting to simulated WebSocket")
        self.connected = True
        self.authenticated = True
        
        # Start the simulation thread if not already running
        if not self.simulation_running:
            self.simulation_running = True
            self.simulation_thread = threading.Thread(target=self._simulate_market_data, daemon=True)
            self.simulation_thread.start()
            logger.info("Simulation thread started")
        
        return self._create_success_response("Connected to simulated WebSocket")
    
    def disconnect(self) -> None:
        """
        Disconnect from the simulated WebSocket
        """
        logger.info("Disconnecting from simulated WebSocket")
        self.connected = False
        self.authenticated = False
        self.simulation_running = False
        
        # Clear subscriptions
        self.subscribed_symbols.clear()
        
        # Clean up ZeroMQ resources
        self.cleanup_zmq()
        
        logger.info("Disconnected from simulated WebSocket")
    
    def subscribe(self, symbol: str, exchange: str, mode: int = 2, depth_level: int = 5) -> Dict[str, Any]:
        """
        Subscribe to simulated market data
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            mode: Subscription mode (1=LTP, 2=Quote, 3=Depth)
            depth_level: Market depth level
            
        Returns:
            Dict with subscription status
        """
        if not self.connected:
            return self._create_error_response("NOT_CONNECTED", "Not connected to simulated WebSocket")
        
        symbol_key = f"{exchange}:{symbol}"
        
        # Initialize symbol data if not already present
        if symbol not in self.symbols_data:
            self.symbols_data[symbol] = {
                "base_price": random.uniform(100, 5000),
                "base_volume": random.randint(1000, 100000)
            }
        
        # Store subscription info
        self.subscribed_symbols[symbol_key] = {
            "symbol": symbol,
            "exchange": exchange,
            "mode": mode,
            "depth_level": depth_level,
            "last_price": self.symbols_data[symbol]["base_price"],
            "last_volume": self.symbols_data[symbol]["base_volume"]
        }
        
        logger.info(f"Subscribed to {symbol_key} in mode {mode}")
        return self._create_success_response(f"Subscribed to {symbol} on {exchange}")
    
    def unsubscribe(self, symbol: str, exchange: str, mode: int = 2) -> Dict[str, Any]:
        """
        Unsubscribe from simulated market data
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            mode: Subscription mode
            
        Returns:
            Dict with unsubscription status
        """
        symbol_key = f"{exchange}:{symbol}"
        
        if symbol_key in self.subscribed_symbols:
            del self.subscribed_symbols[symbol_key]
            logger.info(f"Unsubscribed from {symbol_key}")
            return self._create_success_response(f"Unsubscribed from {symbol} on {exchange}")
        else:
            return self._create_success_response(f"Not subscribed to {symbol} on {exchange}")
    
    def unsubscribe_all(self) -> Dict[str, Any]:
        """
        Unsubscribe from all symbols
        
        Returns:
            Dict with unsubscription status
        """
        count = len(self.subscribed_symbols)
        self.subscribed_symbols.clear()
        logger.info(f"Unsubscribed from all {count} symbols")
        return self._create_success_response(f"Unsubscribed from all {count} symbols")
    
    def _simulate_market_data(self) -> None:
        """
        Background thread function that generates simulated market data
        """
        logger.info("Starting market data simulation")
        
        while self.simulation_running and self.connected:
            try:
                # Generate data for each subscribed symbol
                for symbol_key, symbol_info in list(self.subscribed_symbols.items()):
                    symbol = symbol_info["symbol"]
                    exchange = symbol_info["exchange"]
                    mode = symbol_info["mode"]
                    depth_level = symbol_info["depth_level"]
                    
                    # Generate simulated data based on mode
                    if mode == 1:  # LTP mode
                        market_data = self._generate_ltp_data(symbol_info)
                        topic = f"simulated_{exchange}_{symbol}_LTP"
                    elif mode == 2:  # Quote mode
                        market_data = self._generate_quote_data(symbol_info)
                        topic = f"simulated_{exchange}_{symbol}_QUOTE"
                    elif mode == 3:  # Depth mode
                        market_data = self._generate_depth_data(symbol_info)
                        topic = f"simulated_{exchange}_{symbol}_DEPTH"
                    else:
                        continue
                    
                    # Publish the data via ZeroMQ
                    self.publish_market_data(topic, market_data)
                
                # Wait before generating next batch of data
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in market data simulation: {e}")
                time.sleep(1)
        
        logger.info("Market data simulation stopped")
    
    def _generate_ltp_data(self, symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate simulated LTP data
        
        Args:
            symbol_info: Symbol information dictionary
            
        Returns:
            Dict with LTP data
        """
        # Simulate price movement (±0.5%)
        price_change = random.uniform(-0.5, 0.5) / 100
        new_price = symbol_info["last_price"] * (1 + price_change)
        new_price = round(new_price, 2)
        
        # Simulate volume change
        volume_change = random.randint(-1000, 1000)
        new_volume = max(0, symbol_info["last_volume"] + volume_change)
        
        # Update stored values
        symbol_info["last_price"] = new_price
        symbol_info["last_volume"] = new_volume
        
        return {
            "symbol": symbol_info["symbol"],
            "exchange": symbol_info["exchange"],
            "ltp": new_price,
            "volume": new_volume,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_quote_data(self, symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate simulated quote data
        
        Args:
            symbol_info: Symbol information dictionary
            
        Returns:
            Dict with quote data
        """
        # Get base values
        base_price = symbol_info["last_price"]
        
        # Simulate price movement (±1%)
        price_change = random.uniform(-1, 1) / 100
        ltp = base_price * (1 + price_change)
        ltp = round(ltp, 2)
        
        # Generate OHLC data with some variation
        open_price = base_price * (1 + random.uniform(-0.2, 0.2) / 100)
        high_price = max(open_price, ltp) * (1 + random.uniform(0, 0.5) / 100)
        low_price = min(open_price, ltp) * (1 - random.uniform(0, 0.5) / 100)
        close_price = open_price  # For current day, close = open initially
        
        # Simulate volume change
        volume_change = random.randint(-2000, 2000)
        volume = max(0, symbol_info["last_volume"] + volume_change)
        
        # Update stored values
        symbol_info["last_price"] = ltp
        symbol_info["last_volume"] = volume
        
        return {
            "symbol": symbol_info["symbol"],
            "exchange": symbol_info["exchange"],
            "ltp": ltp,
            "change": round(ltp - open_price, 2),
            "change_percent": round(((ltp - open_price) / open_price) * 100, 2),
            "volume": volume,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "last_trade_quantity": random.randint(1, 1000),
            "avg_trade_price": round((open_price + ltp) / 2, 2),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_depth_data(self, symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate simulated market depth data
        
        Args:
            symbol_info: Symbol information dictionary
            
        Returns:
            Dict with depth data
        """
        base_price = symbol_info["last_price"]
        ltp = round(base_price * (1 + random.uniform(-0.5, 0.5) / 100), 2)
        
        # Generate buy side depth
        buy_levels = []
        for i in range(5):
            price = ltp - (i + 1) * 0.05  # Decreasing prices
            quantity = random.randint(100, 1000)
            orders = random.randint(5, 50)
            buy_levels.append({
                "price": round(price, 2),
                "quantity": quantity,
                "orders": orders
            })
        
        # Generate sell side depth
        sell_levels = []
        for i in range(5):
            price = ltp + (i + 1) * 0.05  # Increasing prices
            quantity = random.randint(100, 1000)
            orders = random.randint(5, 50)
            sell_levels.append({
                "price": round(price, 2),
                "quantity": quantity,
                "orders": orders
            })
        
        # Update stored values
        symbol_info["last_price"] = ltp
        
        return {
            "symbol": symbol_info["symbol"],
            "exchange": symbol_info["exchange"],
            "ltp": ltp,
            "depth": {
                "buy": buy_levels,
                "sell": sell_levels
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
EOF

echo "✓ Created websocket_proxy/simulated_adapter.py"

# Modify the broker factory to include the simulated adapter
echo "Updating broker factory..."
cat > websocket_proxy/broker_factory.py << 'EOF'
import importlib
from typing import Dict, Type, Optional

from .base_adapter import BaseBrokerWebSocketAdapter
from utils.logging import get_logger

logger = get_logger(__name__)

# Registry of all supported broker adapters
BROKER_ADAPTERS: Dict[str, Type[BaseBrokerWebSocketAdapter]] = {}

# Import simulated adapter for analyzer mode
try:
    from .simulated_adapter import SimulatedWebSocketAdapter
    BROKER_ADAPTERS['simulated'] = SimulatedWebSocketAdapter
    logger.info("Simulated WebSocket adapter registered")
except ImportError as e:
    logger.warning(f"Could not import simulated adapter: {e}")

def register_adapter(broker_name: str, adapter_class: Type[BaseBrokerWebSocketAdapter]) -> None:
    """
    Register a broker adapter class for a specific broker
    
    Args:
        broker_name: Name of the broker
        adapter_class: Class that implements the BaseBrokerWebSocketAdapter interface
    """
    BROKER_ADAPTERS[broker_name.lower()] = adapter_class
    

def create_broker_adapter(broker_name: str) -> Optional[BaseBrokerWebSocketAdapter]:
    """
    Create an instance of the appropriate broker adapter
    
    Args:
        broker_name: Name of the broker (e.g., 'angel', 'zerodha')
        
    Returns:
        BaseBrokerWebSocketAdapter: An instance of the appropriate broker adapter
        
    Raises:
        ValueError: If the broker is not supported
    """
    broker_name = broker_name.lower()
    
    # Check if adapter is registered
    if broker_name in BROKER_ADAPTERS:
        logger.info(f"Creating adapter for broker: {broker_name}")
        return BROKER_ADAPTERS[broker_name]()
    
    # Try dynamic import if not registered
    try:
        # Try to import from broker-specific directory first
        module_name = f"broker.{broker_name}.streaming.{broker_name}_adapter"
        class_name = f"{broker_name.capitalize()}WebSocketAdapter"
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the adapter class
            adapter_class = getattr(module, class_name)
            
            # Register the adapter for future use
            register_adapter(broker_name, adapter_class)
            
            # Create and return an instance
            return adapter_class()
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not import from broker-specific path: {e}")
            
            # Try websocket_proxy directory as fallback
            module_name = f"websocket_proxy.{broker_name}_adapter"
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the adapter class
            adapter_class = getattr(module, class_name)
            
            # Register the adapter for future use
            register_adapter(broker_name, adapter_class)
            
            # Create and return an instance
            return adapter_class()
    
    except (ImportError, AttributeError) as e:
        logger.exception(f"Failed to load adapter for broker {broker_name}: {e}")
        raise ValueError(f"Unsupported broker: {broker_name}. No adapter available.")
    
    return None
EOF

echo "✓ Updated websocket_proxy/broker_factory.py"

# Modify the WebSocket proxy server to support analyzer mode
echo "Updating WebSocket proxy server..."
# First, let's backup the original file
cp websocket_proxy/server.py websocket_proxy/server.py.bak

# Create a temporary file with the modifications
cat > websocket_proxy/server_temp.py << 'EOF'
import asyncio as aio
import websockets
import json
from utils.logging import get_logger, highlight_url
import signal
import zmq
import zmq.asyncio
import threading
import time
import os
import socket
from typing import Dict, Set, Any, Optional
from dotenv import load_dotenv

from .port_check import is_port_in_use, find_available_port
from database.auth_db import get_broker_name
from sqlalchemy import text
from database.auth_db import verify_api_key
from .broker_factory import create_broker_adapter
from .base_adapter import BaseBrokerWebSocketAdapter

# Initialize logger
logger = get_logger("websocket_proxy")

class WebSocketProxy:
    """
    WebSocket Proxy Server that handles client connections and authentication,
    manages subscriptions, and routes market data from broker adapters to clients.
    Supports dynamic broker selection based on user configuration.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        """
        Initialize the WebSocket Proxy
        
        Args:
            host: Hostname to bind the WebSocket server to
            port: Port number to bind the WebSocket server to
        """
        self.host = host
        self.port = port
        
        # Check if the required port is already in use - wait briefly for cleanup to complete
        if is_port_in_use(host, port, wait_time=2.0):  # Wait up to 2 seconds for port release
            error_msg = (
                f"WebSocket port {port} is already in use on {host}.\n"
                f"This port is required for SDK compatibility (see strategies/ltp_example.py).\n"
                f"Please:\n"
                f"1. Stop any other OpenAlgo instances running on port {port}\n"
                f"2. Kill any processes using port {port}: lsof -ti:{port} | xargs kill -9\n"
                f"3. Or wait for the port to be released\n"
                f"Cannot start WebSocket server with port switching as it would break SDK clients."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.clients = {}  # Maps client_id to websocket connection
        self.subscriptions = {}  # Maps client_id to set of subscriptions
        self.broker_adapters = {}  # Maps user_id to broker adapter
        self.user_mapping = {}  # Maps client_id to user_id
        self.user_broker_mapping = {}  # Maps user_id to broker_name
        self.running = False
        
        # ZeroMQ context for subscribing to broker adapters
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)
        # Connecting to ZMQ
        ZMQ_HOST = os.getenv('ZMQ_HOST', '127.0.0.1')
        ZMQ_PORT = os.getenv('ZMQ_PORT')
        self.socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")  # Connect to broker adapter publisher
        
        # Set up ZeroMQ subscriber to receive all messages
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all topics
    
    async def start(self):
        """Start the WebSocket server and ZeroMQ listener"""
        self.running = True
        
        try:
            # Start ZeroMQ listener
            logger.info("Initializing ZeroMQ listener task")
            
            # Get the current event loop
            loop = aio.get_running_loop()
            
            # Create the ZMQ listener task
            zmq_task = loop.create_task(self.zmq_listener())
            
            # Start WebSocket server
            stop = aio.Future()  # Used to stop the server
            
            # Create a task to monitor the running flag
            async def monitor_shutdown():
                while self.running:
                    await aio.sleep(0.5)
                stop.set_result(None)
            
            monitor_task = aio.create_task(monitor_shutdown())
            
            # Handle graceful shutdown
            # Windows doesn't support add_signal_handler, so we'll use a simpler approach
            # Also, when running in a thread on Unix systems, signal handlers can't be set
            try:
                loop = aio.get_running_loop()
                
                # Check if we're in the main thread
                if threading.current_thread() is threading.main_thread():
                    try:
                        for sig in (signal.SIGINT, signal.SIGTERM):
                            loop.add_signal_handler(sig, stop.set_result, None)
                        logger.info("Signal handlers registered successfully")
                    except (NotImplementedError, RuntimeError) as e:
                        # On Windows or when in a non-main thread
                        logger.info(f"Signal handlers not registered: {e}. Using fallback mechanism.")
                else:
                    logger.info("Running in a non-main thread. Signal handlers will not be used.")
            except RuntimeError:
                logger.info("No running event loop found for signal handlers")
            
            highlighted_address = highlight_url(f"{self.host}:{self.port}")
            logger.info(f"Starting WebSocket server on {highlighted_address}")
            
            # Try to start the WebSocket server with proper socket options for immediate port reuse
            try:
                # Start WebSocket server with socket reuse options
                self.server = await websockets.serve(
                    self.handle_client, 
                    self.host, 
                    self.port,
                    # Enable socket reuse for immediate port availability after close
                    reuse_port=True if hasattr(socket, 'SO_REUSEPORT') else False
                )
                
                highlighted_success_address = highlight_url(f"{self.host}:{self.port}")
                logger.info(f"WebSocket server successfully started on {highlighted_success_address}")
                
                await stop  # Wait until stopped
                
                # Cancel the monitor task
                monitor_task.cancel()
                try:
                    await monitor_task
                except aio.CancelledError:
                    pass
                
            except Exception as e:
                logger.exception(f"Failed to start WebSocket server: {e}")
                raise
                
        except Exception as e:
            logger.exception(f"Error in start method: {e}")
            raise
    
    async def stop(self):
        """Stop the WebSocket server and clean up all resources"""
        logger.info("Stopping WebSocket server...")
        self.running = False
        
        try:
            # Close the WebSocket server first (this releases the port)
            if hasattr(self, 'server') and self.server:
                try:
                    logger.info("Closing WebSocket server...")
                    # On Windows, we need to handle the case where we're in a different event loop
                    try:
                        self.server.close()
                        await self.server.wait_closed()
                        logger.info("WebSocket server closed and port released")
                    except RuntimeError as e:
                        if "attached to a different loop" in str(e):
                            logger.warning(f"WebSocket server cleanup skipped due to event loop mismatch: {e}")
                            # Force close the server without waiting
                            try:
                                self.server.close()
                            except:
                                pass
                        else:
                            raise
                except Exception as e:
                    logger.error(f"Error closing WebSocket server: {e}")
            
            # Close all client connections
            close_tasks = []
            for client_id, websocket in self.clients.items():
                try:
                    if hasattr(websocket, 'open') and websocket.open:
                        close_tasks.append(websocket.close())
                except Exception as e:
                    logger.error(f"Error preparing to close client {client_id}: {e}")
            
            # Wait for all connections to close with timeout
            if close_tasks:
                try:
                    await aio.wait_for(
                        aio.gather(*close_tasks, return_exceptions=True),
                        timeout=2.0  # 2 second timeout
                    )
                except aio.TimeoutError:
                    logger.warning("Timeout waiting for client connections to close")
            
            # Disconnect all broker adapters
            for user_id, adapter in self.broker_adapters.items():
                try:
                    adapter.disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting adapter for user {user_id}: {e}")
            
            # Close ZeroMQ socket with linger=0 for immediate close
            if hasattr(self, 'socket') and self.socket:
                try:
                    self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait for pending messages
                    self.socket.close()
                except Exception as e:
                    logger.error(f"Error closing ZMQ socket: {e}")
            
            # Close ZeroMQ context with timeout
            if hasattr(self, 'context') and self.context:
                try:
                    self.context.term()
                except Exception as e:
                    logger.error(f"Error terminating ZMQ context: {e}")
            
            logger.info("WebSocket server stopped and resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during WebSocket server stop: {e}")
    
    async def handle_client(self, websocket):
        """
        Handle a client connection
        
        Args:
            websocket: The WebSocket connection
        """
        client_id = id(websocket)
        self.clients[client_id] = websocket
        self.subscriptions[client_id] = set()
        
        # Get path info from websocket if available
        path = getattr(websocket, 'path', '/unknown')
        logger.info(f"Client connected: {client_id} from path: {path}")
        
        try:
            # Process messages from the client
            async for message in websocket:
                try:
                    logger.debug(f"Received message from client {client_id}: {message}")
                    await self.process_client_message(client_id, message)
                except Exception as e:
                    logger.exception(f"Error processing message from client {client_id}: {e}")
                    # Send error to client but don't disconnect
                    try:
                        await self.send_error(client_id, "PROCESSING_ERROR", str(e))
                    except:
                        pass
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client disconnected: {client_id}, code: {e.code}, reason: {e.reason}")
        except Exception as e:
            logger.exception(f"Unexpected error handling client {client_id}: {e}")
        finally:
            # Clean up when the client disconnects
            await self.cleanup_client(client_id)
    
    async def cleanup_client(self, client_id):
        """
        Clean up client resources when they disconnect
        
        Args:
            client_id: Client ID to clean up
        """
        # Remove client from tracking
        if client_id in self.clients:
            del self.clients[client_id]
        
        # Clean up subscriptions
        if client_id in self.subscriptions:
            subscriptions = self.subscriptions[client_id]
            # Unsubscribe from all subscriptions
            for sub_json in subscriptions:
                try:
                    # Parse the JSON string to get the subscription info
                    sub_info = json.loads(sub_json)
                    symbol = sub_info.get('symbol')
                    exchange = sub_info.get('exchange')
                    mode = sub_info.get('mode')
                    
                    # Get the user's broker adapter
                    user_id = self.user_mapping.get(client_id)
                    if user_id and user_id in self.broker_adapters:
                        adapter = self.broker_adapters[user_id]
                        adapter.unsubscribe(symbol, exchange, mode)
                except json.JSONDecodeError as e:
                    logger.exception(f"Error parsing subscription: {sub_json}, Error: {e}")
                except Exception as e:
                    logger.exception(f"Error processing subscription: {e}")
                    continue
            
            del self.subscriptions[client_id]
        
        # Remove from user mapping
        if client_id in self.user_mapping:
            user_id = self.user_mapping[client_id]
            
            # Check if this was the last client for this user
            is_last_client = True
            for other_client_id, other_user_id in self.user_mapping.items():
                if other_client_id != client_id and other_user_id == user_id:
                    is_last_client = False
                    break
            
            # If this was the last client for this user, handle the adapter state
            if is_last_client and user_id in self.broker_adapters:
                adapter = self.broker_adapters[user_id]
                broker_name = self.user_broker_mapping.get(user_id)

                # For Flattrade and Shoonya, keep the connection alive and just unsubscribe from data
                if broker_name in ['flattrade', 'shoonya'] and hasattr(adapter, 'unsubscribe_all'):
                    logger.info(f"{broker_name.title()} adapter for user {user_id}: last client disconnected. Unsubscribing all symbols instead of disconnecting.")
                    adapter.unsubscribe_all()
                else:
                    # For all other brokers, disconnect the adapter completely
                    logger.info(f"Last client for user {user_id} disconnected. Disconnecting {broker_name or 'unknown broker'} adapter.")
                    adapter.disconnect()
                    del self.broker_adapters[user_id]
                    if user_id in self.user_broker_mapping:
                        del self.user_broker_mapping[user_id]
            
            del self.user_mapping[client_id]
    
    async def process_client_message(self, client_id, message):
        """
        Process messages from a client
        
        Args:
            client_id: ID of the client
            message: The message from the client
        """
        try:
            data = json.loads(message)
            logger.debug(f"Parsed message from client {client_id}: {data}")
            
            # Accept both 'action' and 'type' fields for better compatibility with different clients
            action = data.get("action") or data.get("type")
            logger.info(f"Client {client_id} requested action: {action}")
            
            if action in ["authenticate", "auth"]:
                await self.authenticate_client(client_id, data)
            elif action == "subscribe":
                await self.subscribe_client(client_id, data)
            elif action in ["unsubscribe", "unsubscribe_all"]:
                await self.unsubscribe_client(client_id, data)
            elif action == "get_broker_info":
                await self.get_broker_info(client_id)
            elif action == "get_supported_brokers":
                await self.get_supported_brokers(client_id)
            else:
                logger.warning(f"Client {client_id} requested invalid action: {action}")
                await self.send_error(client_id, "INVALID_ACTION", f"Invalid action: {action}")
        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON from client {client_id}: {message}")
            await self.send_error(client_id, "INVALID_JSON", "Invalid JSON message")
        except Exception as e:
            logger.exception(f"Error processing client message: {e}")
            await self.send_error(client_id, "SERVER_ERROR", str(e))
    
    async def get_user_broker_configuration(self, user_id):
        """
        Get the broker configuration for a specific user from database
        
        Args:
            user_id: User ID to get broker configuration for
            
        Returns:
            dict: Broker configuration containing broker_name and credentials
        """
        try:
            from database.auth_db import get_broker_name
            from sqlalchemy import text
            
            # Get user's connected broker from database
            # This queries the auth_token table to find the user's active broker
            query = text("""
                SELECT broker FROM auth_token 
                WHERE user_id = :user_id 
                ORDER BY id DESC 
                LIMIT 1
            """)
            
            result = db.session.execute(query, {"user_id": user_id}).fetchone()
            
            if result and result.broker:
                broker_name = result.broker
                logger.info(f"Found broker '{broker_name}' for user {user_id} from database")
            else:
                # Fallback to environment variable
                valid_brokers = os.getenv('VALID_BROKERS', 'angel').split(',')
                broker_name = valid_brokers[0].strip() if valid_brokers else 'angel'
                logger.warning(f"No broker found in database for user {user_id}, using fallback: {broker_name}")
            
            # Get broker credentials from environment variables
            # In a production system, these would be stored encrypted in the database per user
            broker_config = {
                'broker_name': broker_name,
                'api_key': os.getenv('BROKER_API_KEY'),
                'api_secret': os.getenv('BROKER_API_SECRET'),
                'api_key_market': os.getenv('BROKER_API_KEY_MARKET'),
                'api_secret_market': os.getenv('BROKER_API_SECRET_MARKET'),
                'broker_user_id': os.getenv('BROKER_USER_ID'),
                'password': os.getenv('BROKER_PASSWORD'),
                'totp_secret': os.getenv('BROKER_TOTP_SECRET')
            }
            
            # Validate broker is supported
            valid_brokers_list = os.getenv('VALID_BROKERS', '').split(',')
            valid_brokers_list = [b.strip() for b in valid_brokers_list if b.strip()]
            
            if broker_name not in valid_brokers_list:
                logger.error(f"Broker '{broker_name}' is not in VALID_BROKERS list: {valid_brokers_list}")
                return None
            
            if not broker_config.get('broker_name'):
                logger.error(f"No broker configuration found for user {user_id}")
                return None
            
            logger.info(f"Retrieved broker configuration for user {user_id}: {broker_config['broker_name']}")
            return broker_config
            
        except Exception as e:
            logger.exception(f"Error getting broker configuration for user {user_id}: {e}")
            return None
    
    async def authenticate_client(self, client_id, data):
        """
        Authenticate a client using their API key and determine their broker
        
        Args:
            client_id: ID of the client
            data: Authentication data containing API key
        """
        api_key = data.get("api_key")
        
        if not api_key:
            await self.send_error(client_id, "AUTHENTICATION_ERROR", "API key is required")
            return
        
        # Verify the API key and get the user ID
        user_id = verify_api_key(api_key)
        
        if not user_id:
            await self.send_error(client_id, "AUTHENTICATION_ERROR", "Invalid API key")
            return
        
        # Store the user mapping
        self.user_mapping[client_id] = user_id
        
        # Check if analyzer mode is enabled
        try:
            from database.settings_db import get_analyze_mode
            analyze_mode = get_analyze_mode()
        except Exception as e:
            logger.warning(f"Could not get analyzer mode: {e}")
            analyze_mode = False
        
        if analyze_mode:
            # In analyzer mode, use simulated broker
            broker_name = "simulated"
        else:
            # Get broker name from database
            broker_name = get_broker_name(api_key)
        
        if not broker_name:
            await self.send_error(client_id, "BROKER_ERROR", "No broker configuration found for user")
            return
        
        # Store the broker mapping for this user
        self.user_broker_mapping[user_id] = broker_name
        
        # Create or reuse broker adapter
        if user_id not in self.broker_adapters:
            try:
                if broker_name == "simulated":
                    # Create simulated adapter for analyzer mode
                    adapter = create_broker_adapter("simulated")
                    if not adapter:
                        await self.send_error(client_id, "BROKER_ERROR", "Failed to create simulated adapter")
                        return
                else:
                    # Create broker adapter with dynamic broker selection
                    adapter = create_broker_adapter(broker_name)
                    if not adapter:
                        await self.send_error(client_id, "BROKER_ERROR", f"Failed to create adapter for broker: {broker_name}")
                        return
                
                # Initialize adapter with broker configuration
                # The adapter's initialize method should handle broker-specific setup
                initialization_result = adapter.initialize(broker_name, user_id)
                if initialization_result and not initialization_result.get('success', True):
                    error_msg = initialization_result.get('error', 'Failed to initialize broker adapter')
                    await self.send_error(client_id, "BROKER_INIT_ERROR", error_msg)
                    return
                
                # Connect to the broker
                connect_result = adapter.connect()
                if connect_result and not connect_result.get('success', True):
                    error_msg = connect_result.get('error', 'Failed to connect to broker')
                    await self.send_error(client_id, "BROKER_CONNECTION_ERROR", error_msg)
                    return
                
                # Store the adapter
                self.broker_adapters[user_id] = adapter
                
                logger.info(f"Successfully created and connected {broker_name} adapter for user {user_id}")
                
            except Exception as e:
                logger.error(f"Failed to create broker adapter for {broker_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await self.send_error(client_id, "BROKER_ERROR", str(e))
                return
        
        # Send success response with broker information
        await self.send_message(client_id, {
            "type": "auth",
            "status": "success",
            "message": "Authentication successful",
            "broker": broker_name,
            "user_id": user_id,
            "supported_features": {
                "ltp": True,
                "quote": True,
                "depth": True
            }
        })
    
    async def get_supported_brokers(self, client_id):
        """
        Get list of supported brokers from environment configuration
        
        Args:
            client_id: ID of the client
        """
        try:
            valid_brokers = os.getenv('VALID_BROKERS', '').split(',')
            supported_brokers = [broker.strip() for broker in valid_brokers if broker.strip()]
            
            await self.send_message(client_id, {
                "type": "supported_brokers",
                "status": "success",
                "brokers": supported_brokers,
                "count": len(supported_brokers)
            })
        except Exception as e:
            logger.error(f"Error getting supported brokers: {e}")
            await self.send_error(client_id, "BROKER_LIST_ERROR", str(e))
    
    async def get_broker_info(self, client_id):
        """
        Get broker information for an authenticated client
        
        Args:
            client_id: ID of the client
        """
        # Check if the client is authenticated
        if client_id not in self.user_mapping:
            await self.send_error(client_id, "NOT_AUTHENTICATED", "You must authenticate first")
            return
        
        user_id = self.user_mapping[client_id]
        broker_name = self.user_broker_mapping.get(user_id)
        
        if not broker_name:
            await self.send_error(client_id, "BROKER_ERROR", "Broker information not available")
            return
        
        # Get adapter status
        adapter_status = "disconnected"
        if user_id in self.broker_adapters:
            adapter = self.broker_adapters[user_id]
            # Assuming the adapter has a status method or property
            adapter_status = getattr(adapter, 'status', 'connected')
        
        await self.send_message(client_id, {
            "type": "broker_info",
            "status": "success",
            "broker": broker_name,
            "adapter_status": adapter_status,
            "user_id": user_id
        })
    
    async def subscribe_client(self, client_id, data):
        """
        Subscribe a client to market data using their configured broker
        
        Args:
            client_id: ID of the client
            data: Subscription data
        """
        # Check if the client is authenticated
        if client_id not in self.user_mapping:
            await self.send_error(client_id, "NOT_AUTHENTICATED", "You must authenticate first")
            return
        
        # Get subscription parameters
        symbols = data.get("symbols") or []  # Handle array of symbols
        mode_str = data.get("mode", "Quote")  # Get mode as string (LTP, Quote, Depth)
        depth_level = data.get("depth", 5)  # Default to 5 levels
        
        # Map string mode to numeric mode
        mode_mapping = {
            "LTP": 1,
            "Quote": 2, 
            "Depth": 3
        }
        
        # Convert string mode to numeric if needed
        mode = mode_mapping.get(mode_str, mode_str) if isinstance(mode_str, str) else mode_str
        
        # Handle case where a single symbol is passed directly instead of as an array
        if not symbols and (data.get("symbol") and data.get("exchange")):
            symbols = [{
                "symbol": data.get("symbol"),
                "exchange": data.get("exchange")
            }]
        
        if not symbols:
            await self.send_error(client_id, "INVALID_PARAMETERS", "At least one symbol must be specified")
            return
        
        # Get the user's broker adapter
        user_id = self.user_mapping[client_id]
        if user_id not in self.broker_adapters:
            await self.send_error(client_id, "BROKER_ERROR", "Broker adapter not found")
            return
        
        adapter = self.broker_adapters[user_id]
        broker_name = self.user_broker_mapping.get(user_id, "unknown")
        
        # Process each symbol in the subscription request
        subscription_responses = []
        subscription_success = True
        
        for symbol_info in symbols:
            symbol = symbol_info.get("symbol")
            exchange = symbol_info.get("exchange")
            
            if not symbol or not exchange:
                continue  # Skip invalid symbols
                
            # Subscribe to market data
            if broker_name == "simulated":
                response = adapter.subscribe(symbol, exchange, mode, depth_level)
            else:
                response = adapter.subscribe(symbol, exchange, mode, depth_level)
            
            if response.get("status") == "success":
                # Store the subscription
                subscription_info = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "mode": mode,
                    "depth_level": depth_level,
                    "broker": broker_name
                }
                
                if client_id in self.subscriptions:
                    self.subscriptions[client_id].add(json.dumps(subscription_info))
                else:
                    self.subscriptions[client_id] = {json.dumps(subscription_info)}
                
                # Add to successful subscriptions
                subscription_responses.append({
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "success",
                    "mode": mode_str,
                    "depth": response.get("actual_depth", depth_level),
                    "broker": broker_name
                })
            else:
                subscription_success = False
                # Add to failed subscriptions
                subscription_responses.append({
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "error",
                    "message": response.get("message", "Subscription failed"),
                    "broker": broker_name
                })
        
        # Send combined response
        await self.send_message(client_id, {
            "type": "subscribe",
            "status": "success" if subscription_success else "partial",
            "subscriptions": subscription_responses,
            "message": "Subscription processing complete",
            "broker": broker_name
        })
    
    async def unsubscribe_client(self, client_id, data):
        """
        Unsubscribe a client from market data
        
        Args:
            client_id: ID of the client
            data: Unsubscription data
        """
        # Check if the client is authenticated
        if client_id not in self.user_mapping:
            await self.send_error(client_id, "NOT_AUTHENTICATED", "You must authenticate first")
            return
        
        # Check if this is an unsubscribe_all request
        is_unsubscribe_all = data.get("type") == "unsubscribe_all" or data.get("action") == "unsubscribe_all"
        
        # Get unsubscription parameters for specific symbols
        symbols = data.get("symbols") or []
        
        # Handle single symbol format
        if not symbols and not is_unsubscribe_all and (data.get("symbol") and data.get("exchange")):
            symbols = [{
                "symbol": data.get("symbol"),
                "exchange": data.get("exchange"),
                "mode": data.get("mode", 2)  # Default to Quote mode
            }]
        
        # If no symbols provided and not unsubscribe_all, return error
        if not symbols and not is_unsubscribe_all:
            await self.send_error(client_id, "INVALID_PARAMETERS", "Either symbols or unsubscribe_all is required")
            return
        
        # Get the user's broker adapter
        user_id = self.user_mapping[client_id]
        if user_id not in self.broker_adapters:
            await self.send_error(client_id, "BROKER_ERROR", "Broker adapter not found")
            return
        
        adapter = self.broker_adapters[user_id]
        broker_name = self.user_broker_mapping.get(user_id, "unknown")
        
        # Process unsubscribe request
        successful_unsubscriptions = []
        failed_unsubscriptions = []
        
        # Handle unsubscribe_all case
        if is_unsubscribe_all:
            # Get all current subscriptions
            if client_id in self.subscriptions:
                # Convert all stored subscription strings back to dictionaries
                all_subscriptions = []
                for sub_json in self.subscriptions[client_id]:
                    try:
                        sub_dict = json.loads(sub_json)
                        all_subscriptions.append(sub_dict)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse subscription: {sub_json}")
                
                # Unsubscribe from each subscription
                for sub in all_subscriptions:
                    symbol = sub.get("symbol")
                    exchange = sub.get("exchange")
                    mode = sub.get("mode")
                    
                    if symbol and exchange:
                        response = adapter.unsubscribe(symbol, exchange, mode)
                        
                        if response.get("status") == "success":
                            successful_unsubscriptions.append({
                                "symbol": symbol,
                                "exchange": exchange,
                                "status": "success",
                                "broker": broker_name
                            })
                        else:
                            failed_unsubscriptions.append({
                                "symbol": symbol,
                                "exchange": exchange,
                                "status": "error",
                                "message": response.get("message", "Unsubscription failed"),
                                "broker": broker_name
                            })
                
                # Clear all subscriptions for this client
                self.subscriptions[client_id].clear()
        else:
            # Process specific symbols
            for symbol_info in symbols:
                symbol = symbol_info.get("symbol")
                exchange = symbol_info.get("exchange")
                mode = symbol_info.get("mode", 2)  # Default to Quote mode
                
                if not symbol or not exchange:
                    continue  # Skip invalid symbols
                
                # Unsubscribe from market data
                if broker_name == "simulated":
                    response = adapter.unsubscribe(symbol, exchange, mode)
                else:
                    response = adapter.unsubscribe(symbol, exchange, mode)
                
                if response.get("status") == "success":
                    # Try to remove subscription
                    if client_id in self.subscriptions:
                        subscription_info = {
                            "symbol": symbol,
                            "exchange": exchange,
                            "mode": mode,
                            "broker": broker_name
                        }
                        subscription_key = json.dumps(subscription_info)
                        # Remove any matching subscription (with or without broker info)
                        subscriptions_to_remove = []
                        for sub_key in self.subscriptions[client_id]:
                            try:
                                sub_data = json.loads(sub_key)
                                if (sub_data.get("symbol") == symbol and 
                                    sub_data.get("exchange") == exchange and 
                                    sub_data.get("mode") == mode):
                                    subscriptions_to_remove.append(sub_key)
                            except json.JSONDecodeError:
                                continue
                        
                        for sub_key in subscriptions_to_remove:
                            self.subscriptions[client_id].discard(sub_key)
                    
                    successful_unsubscriptions.append({
                        "symbol": symbol,
                        "exchange": exchange,
                        "status": "success",
                        "broker": broker_name
                    })
                else:
                    failed_unsubscriptions.append({
                        "symbol": symbol,
                        "exchange": exchange,
                        "status": "error",
                        "message": response.get("message", "Unsubscription failed"),
                        "broker": broker_name
                    })
        
        # Send combined response
        status = "success"
        if len(failed_unsubscriptions) > 0 and len(successful_unsubscriptions) > 0:
            status = "partial"
        elif len(failed_unsubscriptions) > 0 and len(successful_unsubscriptions) == 0:
            status = "error"
            
        await self.send_message(client_id, {
            "type": "unsubscribe",
            "status": status,
            "message": "Unsubscription processing complete",
            "successful": successful_unsubscriptions,
            "failed": failed_unsubscriptions,
            "broker": broker_name
        })
    
    async def send_message(self, client_id, message):
        """
        Send a message to a client
        
        Args:
            client_id: ID of the client
            message: The message to send
        """
        if client_id in self.clients:
            websocket = self.clients[client_id]
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Connection closed while sending message to client {client_id}")
    
    async def send_error(self, client_id, code, message):
        """
        Send an error message to a client
        
        Args:
            client_id: ID of the client
            code: Error code
            message: Error message
        """
        await self.send_message(client_id, {
            "status": "error",
            "code": code,
            "message": message
        })
    
    async def zmq_listener(self):
        """Listen for messages from broker adapters via ZeroMQ and forward to clients"""
        logger.info("Starting ZeroMQ listener")
        
        while self.running:
            try:
                # Check if we should stop
                if not self.running:
                    break
                    
                # Receive message from ZeroMQ with a timeout
                try:
                    [topic, data] = await aio.wait_for(
                        self.socket.recv_multipart(),
                        timeout=0.1
                    )
                except aio.TimeoutError:
                    # No message received within timeout, continue the loop
                    continue
                
                # Parse the message
                topic_str = topic.decode('utf-8')
                data_str = data.decode('utf-8')
                market_data = json.loads(data_str)
                
                # Extract topic components
                # Support both formats:
                # New format: BROKER_EXCHANGE_SYMBOL_MODE (with broker name)
                # Old format: EXCHANGE_SYMBOL_MODE (without broker name)
                # Special case: NSE_INDEX_SYMBOL_MODE (exchange contains underscore)
                parts = topic_str.split('_')
                
                # Special case handling for NSE_INDEX and BSE_INDEX
                if len(parts) >= 4 and parts[0] == "NSE" and parts[1] == "INDEX":
                    broker_name = "unknown"
                    exchange = "NSE_INDEX"
                    symbol = parts[2]
                    mode_str = parts[3]
                elif len(parts) >= 4 and parts[0] == "BSE" and parts[1] == "INDEX":
                    broker_name = "unknown"
                    exchange = "BSE_INDEX"
                    symbol = parts[2]
                    mode_str = parts[3]
                elif len(parts) >= 5 and parts[1] == "INDEX":  # BROKER_NSE_INDEX_SYMBOL_MODE format
                    broker_name = parts[0]
                    exchange = f"{parts[1]}_{parts[2]}"
                    symbol = parts[3]
                    mode_str = parts[4]
                elif len(parts) >= 4:
                    # Standard format with broker name
                    broker_name = parts[0]
                    exchange = parts[1]
                    symbol = parts[2]
                    mode_str = parts[3]
                elif len(parts) >= 3:
                    # Old format without broker name
                    broker_name = "unknown"
                    exchange = parts[0]
                    symbol = parts[1] 
                    mode_str = parts[2]
                else:
                    logger.warning(f"Invalid topic format: {topic_str}")
                    continue
                
                # Map mode string to mode number
                mode_map = {"LTP": 1, "QUOTE": 2, "DEPTH": 3}
                mode = mode_map.get(mode_str)
                
                if not mode:
                    logger.warning(f"Invalid mode in topic: {mode_str}")
                    continue
                
                # Find clients subscribed to this data
                # Create a snapshot of the subscriptions before iteration to avoid
                # 'dictionary changed size during iteration' errors
                subscriptions_snapshot = list(self.subscriptions.items())
                
                for client_id, subscriptions in subscriptions_snapshot:
                    user_id = self.user_mapping.get(client_id)
                    if not user_id:
                        continue
                    
                    # Check if this client's broker matches the message broker (if broker is specified)
                    client_broker = self.user_broker_mapping.get(user_id)
                    if broker_name != "unknown" and client_broker and client_broker != broker_name:
                        continue  # Skip if broker doesn't match
                    
                    # Create a snapshot of the subscription set before iteration
                    subscriptions_list = list(subscriptions)
                    for sub_json in subscriptions_list:
                        try:
                            sub = json.loads(sub_json)
                            
                            # Check subscription match
                            if (sub.get("symbol") == symbol and 
                                sub.get("exchange") == exchange and 
                                (sub.get("mode") == mode or 
                                 (mode_str == "LTP" and sub.get("mode") == 1) or
                                 (mode_str == "QUOTE" and sub.get("mode") == 2) or
                                 (mode_str == "DEPTH" and sub.get("mode") == 3))):
                                
                                # Forward data to the client
                                await self.send_message(client_id, {
                                    "type": "market_data",
                                    "symbol": symbol,
                                    "exchange": exchange,
                                    "mode": mode,
                                    "broker": broker_name if broker_name != "unknown" else client_broker,
                                    "data": market_data
                                })
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing subscription: {sub_json}, Error: {e}")
                            continue
            
            except Exception as e:
                logger.error(f"Error in ZeroMQ listener: {e}")
                # Continue running despite errors
                await aio.sleep(1)

# Entry point for running the server standalone
async def main():
    """Main entry point for running the WebSocket proxy server"""
    proxy = None
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Get WebSocket configuration from environment variables
        ws_host = os.getenv('WEBSOCKET_HOST', '127.0.0.1')
        ws_port = int(os.getenv('WEBSOCKET_PORT', '8765'))
        
        # Create and start the WebSocket proxy
        proxy = WebSocketProxy(host=ws_host, port=ws_port)
        
        await proxy.start()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt)")
    except RuntimeError as e:
        if "set_wakeup_fd only works in main thread" in str(e):
            logger.error(f"Error in start method: {e}")
EOF

# Replace the original server.py with the modified version
mv websocket_proxy/server_temp.py websocket_proxy/server.py
echo "✓ Updated websocket_proxy/server.py"

# Create a simple test script to verify the installation
echo "Creating test script..."
cat > test_simulated_websocket_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify the simulated WebSocket installation
"""

import sys
import os

def test_installation():
    """Test that all required files were created and modified correctly"""
    print("Testing Simulated WebSocket Installation")
    print("=" * 50)
    
    # Test 1: Check if simulated adapter file exists
    if os.path.exists('websocket_proxy/simulated_adapter.py'):
        print("✓ Simulated adapter file exists")
    else:
        print("✗ Simulated adapter file does NOT exist")
        return False
    
    # Test 2: Check if broker factory was modified
    if os.path.exists('websocket_proxy/broker_factory.py'):
        with open('websocket_proxy/broker_factory.py', 'r') as f:
            content = f.read()
            if "'simulated': SimulatedWebSocketAdapter" in content:
                print("✓ Broker factory includes simulated adapter")
            else:
                print("✗ Broker factory does NOT include simulated adapter")
                return False
    else:
        print("✗ Broker factory file does NOT exist")
        return False
    
    # Test 3: Check if server was modified
    if os.path.exists('websocket_proxy/server.py'):
        with open('websocket_proxy/server.py', 'r') as f:
            content = f.read()
            if 'from database.settings_db import get_analyze_mode' in content:
                print("✓ WebSocket server includes analyzer mode detection")
            else:
                print("✗ WebSocket server does NOT include analyzer mode detection")
                return False
    else:
        print("✗ WebSocket server file does NOT exist")
        return False
    
    print("\n" + "=" * 50)
    print("✓ Installation verification successful!")
    print("\nTo use the simulated WebSocket feature:")
    print("1. Enable analyzer mode in the OpenAlgo UI")
    print("2. WebSocket connections will automatically use simulated data")
    print("3. No additional setup is required")
    
    return True

if __name__ == "__main__":
    if test_installation():
        print("\n🎉 Installation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Installation verification failed!")
        sys.exit(1)
EOF

echo "✓ Created test_simulated_websocket_installation.py"

# Make the test script executable
chmod +x test_simulated_websocket_installation.py

# Create a cleanup script to properly handle server restarts
echo "Creating cleanup script..."
cat > cleanup_websocket_processes.sh << 'EOF'
#!/bin/bash

# Script to clean up any existing WebSocket processes before starting
echo "Cleaning up existing WebSocket processes..."

# Kill any existing WebSocket proxy processes
WEBSOCKET_PIDS=$(ps aux | grep "websocket_proxy.server" | grep -v grep | awk '{print $2}')
if [ ! -z "$WEBSOCKET_PIDS" ]; then
    echo "Killing existing WebSocket processes: $WEBSOCKET_PIDS"
    kill $WEBSOCKET_PIDS 2>/dev/null
    sleep 2
fi

# Kill any processes using port 8765
PORT_PIDS=$(lsof -ti:8765 2>/dev/null)
if [ ! -z "$PORT_PIDS" ]; then
    echo "Killing processes using port 8765: $PORT_PIDS"
    kill $PORT_PIDS 2>/dev/null
    sleep 2
fi

echo "Cleanup completed."
EOF

echo "✓ Created cleanup_websocket_processes.sh"
chmod +x cleanup_websocket_processes.sh

# Update the start.sh script to include proper cleanup
echo "Updating start.sh script..."
cat > start.sh.updated << 'EOF'
#!/bin/bash

echo "[OpenAlgo] Starting up..."

# Try to create directories, but don't fail if they already exist or can't be created
# This handles both mounted volumes and permission issues
for dir in db log log/strategies strategies strategies/scripts keys; do
    mkdir -p "$dir" 2>/dev/null || true
done

# Try to set permissions if possible, but continue regardless
# This will work for local directories but skip for mounted volumes
if [ -w "." ]; then
    # Set more permissive permissions for directories
    chmod -R 755 db log strategies 2>/dev/null || echo "⚠️  Skipping chmod (may be mounted volume or permission restricted)"
    # Set restrictive permissions for keys directory (only owner can access)
    chmod 700 keys 2>/dev/null || true
else
    echo "⚠️  Running with restricted permissions (mounted volume detected)"
fi

# Ensure Python can create directories at runtime if needed
export PYTHONDONTWRITEBYTECODE=1

cd /app

# Clean up any existing WebSocket processes
echo "[OpenAlgo] Cleaning up existing WebSocket processes..."
./cleanup_websocket_processes.sh

# Start WebSocket proxy server in background
echo "[OpenAlgo] Starting WebSocket proxy server on port 8765..."
/app/.venv/bin/python -m websocket_proxy.server &
WEBSOCKET_PID=$!
echo "[OpenAlgo] WebSocket proxy server started with PID $WEBSOCKET_PID"

# Function to cleanup on exit
cleanup() {
    echo "[OpenAlgo] Shutting down..."
    if [ ! -z "$WEBSOCKET_PID" ]; then
        kill $WEBSOCKET_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Run main application with gunicorn using eventlet for WebSocket support
echo "[OpenAlgo] Starting application on port 5000 with eventlet..."
exec /app/.venv/bin/gunicorn \
    --worker-class eventlet \
    --workers 1 \
    --bind 0.0.0.0:5000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --log-level warning \
    app:app
EOF

# Replace the original start.sh with the updated version
mv start.sh.updated start.sh
chmod +x start.sh
echo "✓ Updated start.sh"

echo
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo
echo "The simulated WebSocket feature has been successfully installed with fixes."
echo
echo "What was installed:"
echo "  ✓ websocket_proxy/simulated_adapter.py - Simulated WebSocket adapter"
echo "  ✓ websocket_proxy/broker_factory.py - Updated broker factory"
echo "  ✓ websocket_proxy/server.py - Modified WebSocket proxy server"
echo "  ✓ test_simulated_websocket_installation.py - Verification script"
echo "  ✓ cleanup_websocket_processes.sh - Cleanup script for proper restarts"
echo "  ✓ start.sh - Updated startup script with proper cleanup"
echo
echo "To verify the installation, run:"
echo "  python3 test_simulated_websocket_installation.py"
echo
echo "To use the feature:"
echo "  1. Enable analyzer mode in the OpenAlgo UI"
echo "  2. WebSocket connections will automatically use simulated data"
echo "  3. No additional setup is required"
echo
echo "The implementation supports:"
echo "  • LTP (Last Traded Price) mode"
echo "  • Quote (OHLC) mode"
echo "  • Depth (Market Depth) mode"
echo "  • Realistic price and volume simulation"
echo "  • Thread-safe operation"
echo "  • Proper cleanup on server restarts"
echo
echo "Enjoy safe testing with simulated market data!"
EOF
