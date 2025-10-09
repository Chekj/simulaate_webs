#!/bin/bash

# Minimal fix for simulated WebSocket in OpenAlgo
# This script adds analyzer mode support without over-engineering

set -e

echo "Applying minimal fix for simulated WebSocket..."

# Check if we're in the correct directory
if [ ! -f "app.py" ] || [ ! -d "websocket_proxy" ]; then
    echo "Error: Must run from OpenAlgo root directory"
    exit 1
fi

# Backup original files
cp websocket_proxy/broker_factory.py websocket_proxy/broker_factory.py.backup
cp websocket_proxy/server.py websocket_proxy/server.py.backup

# Create a minimal simulated adapter
cat > websocket_proxy/simulated_adapter.py << 'EOF'
import json
import random
import threading
import time
from datetime import datetime, timezone
from .base_adapter import BaseBrokerWebSocketAdapter
from utils.logging import get_logger

logger = get_logger(__name__)

class SimulatedWebSocketAdapter(BaseBrokerWebSocketAdapter):
    def __init__(self):
        # Initialize without binding to ports to avoid conflicts
        self.logger = get_logger("simulated_adapter")
        self.connected = False
        self.subscriptions = {}
        self.simulation_thread = None
        self.simulation_running = False
        
    def initialize(self, broker_name, user_id, auth_data=None):
        return {"status": "success", "message": "Simulated adapter initialized"}
        
    def connect(self):
        self.connected = True
        self.simulation_running = True
        self.simulation_thread = threading.Thread(target=self._simulate_data, daemon=True)
        self.simulation_thread.start()
        return {"status": "success", "message": "Connected to simulated data"}
        
    def disconnect(self):
        self.connected = False
        self.simulation_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1)
            
    def subscribe(self, symbol, exchange, mode=2, depth_level=5):
        if not self.connected:
            return {"status": "error", "message": "Not connected"}
        key = f"{exchange}:{symbol}:{mode}"
        self.subscriptions[key] = {"symbol": symbol, "exchange": exchange, "mode": mode}
        return {"status": "success", "message": f"Subscribed to {symbol}"}
        
    def unsubscribe(self, symbol, exchange, mode=2):
        key = f"{exchange}:{symbol}:{mode}"
        if key in self.subscriptions:
            del self.subscriptions[key]
        return {"status": "success", "message": f"Unsubscribed from {symbol}"}
        
    def unsubscribe_all(self):
        self.subscriptions.clear()
        return {"status": "success", "message": "Unsubscribed from all symbols"}
        
    def _simulate_data(self):
        while self.simulation_running and self.connected:
            try:
                time.sleep(1)
            except:
                break
EOF

# Update broker factory with minimal changes
cat > websocket_proxy/broker_factory.py << 'EOF'
import importlib
from typing import Dict, Type, Optional
from .base_adapter import BaseBrokerWebSocketAdapter
from utils.logging import get_logger

logger = get_logger(__name__)
BROKER_ADAPTERS: Dict[str, Type[BaseBrokerWebSocketAdapter]] = {}

def register_adapter(broker_name: str, adapter_class: Type[BaseBrokerWebSocketAdapter]) -> None:
    BROKER_ADAPTERS[broker_name.lower()] = adapter_class

# Safely import and register simulated adapter
try:
    from .simulated_adapter import SimulatedWebSocketAdapter
    register_adapter('simulated', SimulatedWebSocketAdapter)
except Exception as e:
    logger.warning(f"Could not import simulated adapter: {e}")

def create_broker_adapter(broker_name: str) -> Optional[BaseBrokerWebSocketAdapter]:
    broker_name = broker_name.lower()
    if broker_name in BROKER_ADAPTERS:
        return BROKER_ADAPTERS[broker_name]()
    
    # Try dynamic import for real brokers
    try:
        module_name = f"broker.{broker_name}.streaming.{broker_name}_adapter"
        class_name = f"{broker_name.capitalize()}WebSocketAdapter"
        module = importlib.import_module(module_name)
        adapter_class = getattr(module, class_name)
        register_adapter(broker_name, adapter_class)
        return adapter_class()
    except:
        try:
            module_name = f"websocket_proxy.{broker_name}_adapter"
            module = importlib.import_module(module_name)
            adapter_class = getattr(module, class_name)
            register_adapter(broker_name, adapter_class)
            return adapter_class()
        except Exception as e:
            logger.exception(f"Failed to load adapter for broker {broker_name}: {e}")
            return None
EOF

# Update server with minimal analyzer mode detection
cat > websocket_proxy/server.py << 'EOF'
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

logger = get_logger("websocket_proxy")

class WebSocketProxy:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        if is_port_in_use(host, port, wait_time=2.0):
            error_msg = f"WebSocket port {port} is already in use"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.clients = {}
        self.subscriptions = {}
        self.broker_adapters = {}
        self.user_mapping = {}
        self.user_broker_mapping = {}
        self.running = False
        
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)
        ZMQ_HOST = os.getenv('ZMQ_HOST', '127.0.0.1')
        ZMQ_PORT = os.getenv('ZMQ_PORT')
        self.socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
    
    async def start(self):
        self.running = True
        try:
            loop = aio.get_running_loop()
            stop = aio.Future()
            
            highlighted_address = highlight_url(f"{self.host}:{self.port}")
            logger.info(f"Starting WebSocket server on {highlighted_address}")
            
            self.server = await websockets.serve(
                self.handle_client, 
                self.host, 
                self.port,
                reuse_port=True if hasattr(socket, 'SO_REUSEPORT') else False
            )
            
            highlighted_success_address = highlight_url(f"{self.host}:{self.port}")
            logger.info(f"WebSocket server successfully started on {highlighted_success_address}")
            
            await stop
        except Exception as e:
            logger.exception(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop(self):
        logger.info("Stopping WebSocket server...")
        self.running = False
        
        try:
            if hasattr(self, 'server') and self.server:
                try:
                    self.server.close()
                    await self.server.wait_closed()
                except:
                    pass
            
            for user_id, adapter in self.broker_adapters.items():
                try:
                    adapter.disconnect()
                except:
                    pass
            
            if hasattr(self, 'socket') and self.socket:
                try:
                    self.socket.setsockopt(zmq.LINGER, 0)
                    self.socket.close()
                except:
                    pass
            
            if hasattr(self, 'context') and self.context:
                try:
                    self.context.term()
                except:
                    pass
            
            logger.info("WebSocket server stopped")
        except Exception as e:
            logger.error(f"Error during WebSocket server stop: {e}")
    
    async def handle_client(self, websocket):
        client_id = id(websocket)
        self.clients[client_id] = websocket
        self.subscriptions[client_id] = set()
        
        try:
            async for message in websocket:
                try:
                    await self.process_client_message(client_id, message)
                except:
                    pass
        except:
            pass
        finally:
            await self.cleanup_client(client_id)
    
    async def cleanup_client(self, client_id):
        if client_id in self.clients:
            del self.clients[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        if client_id in self.user_mapping:
            user_id = self.user_mapping[client_id]
            del self.user_mapping[client_id]
            
            # Check if this was the last client for this user
            is_last_client = True
            for other_client_id, other_user_id in self.user_mapping.items():
                if other_client_id != client_id and other_user_id == user_id:
                    is_last_client = False
                    break
            
            if is_last_client and user_id in self.broker_adapters:
                adapter = self.broker_adapters[user_id]
                broker_name = self.user_broker_mapping.get(user_id)
                try:
                    adapter.disconnect()
                except:
                    pass
                del self.broker_adapters[user_id]
                if user_id in self.user_broker_mapping:
                    del self.user_broker_mapping[user_id]
    
    async def process_client_message(self, client_id, message):
        try:
            data = json.loads(message)
            action = data.get("action") or data.get("type")
            
            if action in ["authenticate", "auth"]:
                await self.authenticate_client(client_id, data)
            elif action == "subscribe":
                await self.subscribe_client(client_id, data)
            elif action in ["unsubscribe", "unsubscribe_all"]:
                await self.unsubscribe_client(client_id, data)
        except:
            pass
    
    async def authenticate_client(self, client_id, data):
        api_key = data.get("api_key")
        if not api_key:
            await self.send_error(client_id, "AUTHENTICATION_ERROR", "API key is required")
            return
        
        user_id = verify_api_key(api_key)
        if not user_id:
            await self.send_error(client_id, "AUTHENTICATION_ERROR", "Invalid API key")
            return
        
        self.user_mapping[client_id] = user_id
        
        # Check analyzer mode with proper error handling
        broker_name = "unknown"
        try:
            from database.settings_db import get_analyze_mode
            if get_analyze_mode():
                broker_name = "simulated"
            else:
                broker_name = get_broker_name(api_key)
        except:
            broker_name = get_broker_name(api_key) or "unknown"
        
        if broker_name == "unknown":
            await self.send_error(client_id, "BROKER_ERROR", "No broker configuration found")
            return
        
        self.user_broker_mapping[user_id] = broker_name
        
        if user_id not in self.broker_adapters:
            try:
                adapter = create_broker_adapter(broker_name)
                if not adapter:
                    await self.send_error(client_id, "BROKER_ERROR", f"Failed to create adapter for: {broker_name}")
                    return
                
                result = adapter.initialize(broker_name, user_id)
                if result and not result.get('status') == 'success':
                    await self.send_error(client_id, "BROKER_INIT_ERROR", result.get('message', 'Init failed'))
                    return
                
                result = adapter.connect()
                if result and not result.get('status') == 'success':
                    await self.send_error(client_id, "BROKER_CONNECTION_ERROR", result.get('message', 'Connect failed'))
                    return
                
                self.broker_adapters[user_id] = adapter
                logger.info(f"Created and connected {broker_name} adapter for user {user_id}")
                
            except Exception as e:
                logger.error(f"Failed to create broker adapter: {e}")
                await self.send_error(client_id, "BROKER_ERROR", str(e))
                return
        
        await self.send_message(client_id, {
            "type": "auth",
            "status": "success",
            "message": "Authentication successful",
            "broker": broker_name,
            "user_id": user_id
        })
    
    async def subscribe_client(self, client_id, data):
        if client_id not in self.user_mapping:
            await self.send_error(client_id, "NOT_AUTHENTICATED", "Authenticate first")
            return
        
        symbols = data.get("symbols") or []
        if not symbols and (data.get("symbol") and data.get("exchange")):
            symbols = [{"symbol": data.get("symbol"), "exchange": data.get("exchange")}]
        
        if not symbols:
            await self.send_error(client_id, "INVALID_PARAMETERS", "Specify symbols")
            return
        
        user_id = self.user_mapping[client_id]
        if user_id not in self.broker_adapters:
            await self.send_error(client_id, "BROKER_ERROR", "Broker adapter not found")
            return
        
        adapter = self.broker_adapters[user_id]
        broker_name = self.user_broker_mapping.get(user_id, "unknown")
        
        for symbol_info in symbols:
            symbol = symbol_info.get("symbol")
            exchange = symbol_info.get("exchange")
            if not symbol or not exchange:
                continue
                
            response = adapter.subscribe(symbol, exchange)
            if response.get("status") == "success":
                subscription_info = {"symbol": symbol, "exchange": exchange}
                if client_id in self.subscriptions:
                    self.subscriptions[client_id].add(json.dumps(subscription_info))
                else:
                    self.subscriptions[client_id] = {json.dumps(subscription_info)}
        
        await self.send_message(client_id, {
            "type": "subscribe",
            "status": "success",
            "message": "Subscription processing complete",
            "broker": broker_name
        })
    
    async def unsubscribe_client(self, client_id, data):
        if client_id not in self.user_mapping:
            await self.send_error(client_id, "NOT_AUTHENTICATED", "Authenticate first")
            return
        
        user_id = self.user_mapping[client_id]
        if user_id not in self.broker_adapters:
            await self.send_error(client_id, "BROKER_ERROR", "Broker adapter not found")
            return
        
        adapter = self.broker_adapters[user_id]
        broker_name = self.user_broker_mapping.get(user_id, "unknown")
        
        is_unsubscribe_all = data.get("type") == "unsubscribe_all" or data.get("action") == "unsubscribe_all"
        symbols = data.get("symbols") or []
        
        if not symbols and not is_unsubscribe_all and (data.get("symbol") and data.get("exchange")):
            symbols = [{"symbol": data.get("symbol"), "exchange": data.get("exchange")}]
        
        if is_unsubscribe_all:
            if client_id in self.subscriptions:
                self.subscriptions[client_id].clear()
            adapter.unsubscribe_all()
        else:
            for symbol_info in symbols:
                symbol = symbol_info.get("symbol")
                exchange = symbol_info.get("exchange")
                if not symbol or not exchange:
                    continue
                adapter.unsubscribe(symbol, exchange)
        
        await self.send_message(client_id, {
            "type": "unsubscribe",
            "status": "success",
            "message": "Unsubscription processing complete",
            "broker": broker_name
        })
    
    async def send_message(self, client_id, message):
        if client_id in self.clients:
            websocket = self.clients[client_id]
            try:
                await websocket.send(json.dumps(message))
            except:
                pass
    
    async def send_error(self, client_id, code, message):
        await self.send_message(client_id, {
            "status": "error",
            "code": code,
            "message": message
        })

async def main():
    proxy = None
    try:
        load_dotenv()
        ws_host = os.getenv('WEBSOCKET_HOST', '127.0.0.1')
        ws_port = int(os.getenv('WEBSOCKET_PORT', '8765'))
        proxy = WebSocketProxy(host=ws_host, port=ws_port)
        await proxy.start()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        if proxy:
            try:
                await proxy.stop()
            except:
                pass

if __name__ == "__main__":
    aio.run(main())
EOF

echo "Fix applied successfully!"
echo "To enable analyzer mode, set analyze_mode = 1 in the settings table"
