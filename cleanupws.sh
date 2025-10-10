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
