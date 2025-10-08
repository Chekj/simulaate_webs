# Simulated WebSocket Implementation for Analyzer Mode

## Overview

This document details the implementation of a simulated WebSocket adapter for OpenAlgo's analyzer mode. When analyzer mode is enabled, the system uses simulated market data instead of connecting to real brokers, allowing for safe testing and analysis without financial risk.

## Key Components

### 1. Simulated WebSocket Adapter

**File:** `websocket_proxy/simulated_adapter.py`

The SimulatedWebSocketAdapter class extends the BaseBrokerWebSocketAdapter and provides all the functionality needed to simulate market data:

#### Features:
- Generates realistic price movements for subscribed symbols
- Simulates volume changes over time
- Supports all three data modes:
  - LTP (Last Traded Price)
  - Quote (OHLC data)
  - Depth (Market depth with buy/sell levels)
- Runs data generation in a background thread for real-time simulation
- Maintains subscription tracking for each symbol

#### Implementation Details:
```python
class SimulatedWebSocketAdapter(BaseBrokerWebSocketAdapter):
    def __init__(self):
        super().__init__()
        self.simulation_thread = None
        self.simulation_running = False
        self.subscribed_symbols = {}
```

The adapter stores information about each subscribed symbol including its last price, base volume, and subscription mode. It generates realistic price movements using random fluctuations (±0.5% per update) and volume changes.

### 2. WebSocket Proxy Server Modifications

**File:** `websocket_proxy/server.py`

The WebSocket proxy server was modified to check the analyzer mode setting and use the simulated adapter when enabled:

#### Key Changes:
1. Import statements added:
   ```python
   from .simulated_adapter import SimulatedWebSocketAdapter
   from database.settings_db import get_analyze_mode
   ```

2. Authentication method updated to check analyzer mode:
   ```python
   # Check if analyzer mode is enabled
   analyze_mode = get_analyze_mode()
   
   if user_id not in self.broker_adapters:
       try:
           if analyze_mode:
               # Use simulated adapter in analyzer mode
               adapter = SimulatedWebSocketAdapter()
               broker_name = "simulated"
               self.user_broker_mapping[user_id] = broker_name
           else:
               # Create broker adapter with dynamic broker selection
               adapter = create_broker_adapter(broker_name)
   ```

3. Subscription handling updated to work with simulated adapter:
   ```python
   # Subscribe to market data
   if broker_name == "simulated":
       response = adapter.subscribe(symbol, exchange, mode, depth_level)
   else:
       response = adapter.subscribe(symbol, exchange, mode, depth_level)
   ```

### 3. Broker Factory Registration

**File:** `websocket_proxy/broker_factory.py`

The simulated adapter is registered in the broker factory to make it available for use:

```python
# Registry of all supported broker adapters
BROKER_ADAPTERS: Dict[str, Type[BaseBrokerWebSocketAdapter]] = {
    'simulated': SimulatedWebSocketAdapter
}
```

### 4. Analyzer UI Enhancements

**File:** `templates/analyzer.html`

The analyzer page was enhanced to show the current mode status:

#### Added Elements:
- Visual indicator showing whether analyzer mode is enabled or disabled
- Dynamic status text that updates based on the current mode
- Informative description about the mode's behavior

#### JavaScript Integration:
```javascript
// Check analyzer mode status
fetch('/settings/analyze-mode')
    .then(response => response.json())
    .then(data => {
        const modeStatus = document.getElementById('mode-status');
        if (data.analyze_mode) {
            modeStatus.textContent = 'ENABLED';
            modeStatus.className = 'font-bold text-success';
        } else {
            modeStatus.textContent = 'DISABLED';
            modeStatus.className = 'font-bold text-warning';
        }
    })
```

## How It Works

### 1. Mode Switching

Users can toggle between Live Mode and Analyzer Mode using the switch in the navbar. This is handled by the existing mode-toggle.js functionality which:

1. Sends a POST request to `/settings/analyze-mode/{mode}` to update the setting
2. Reloads the page to apply the changes
3. Updates the UI to reflect the new mode

### 2. Data Simulation

When analyzer mode is enabled:

1. New WebSocket connections use the SimulatedWebSocketAdapter instead of real broker adapters
2. Subscriptions are tracked but don't connect to any external service
3. A background thread generates market data at regular intervals (1 second)
4. Generated data is published via ZeroMQ just like real broker data
5. Client applications receive the simulated data through the existing WebSocket infrastructure

### 3. Data Generation Algorithm

The simulation generates realistic market data using the following approach:

#### Price Simulation:
- Each symbol starts with a random base price between 100-5000
- Price changes are generated using random fluctuations (±0.5% per update)
- New prices are rounded to 2 decimal places

#### Volume Simulation:
- Each symbol starts with a random base volume between 1000-100000
- Volume changes are generated using random adjustments (±1000 per update)
- Volume is constrained to never go below zero

#### Mode-Specific Data:
- **LTP Mode**: Generates simple price and volume data
- **Quote Mode**: Generates OHLC data with open, high, low, close prices
- **Depth Mode**: Generates buy and sell order books with configurable depth levels

## Integration with Existing System

The implementation seamlessly integrates with the existing OpenAlgo architecture:

1. **ZeroMQ Integration**: Simulated data is published using the same ZeroMQ mechanism as real broker data
2. **WebSocket Proxy**: The proxy server handles simulated adapters the same way as real adapters
3. **Client Applications**: No changes needed in client applications - they receive simulated data through the same channels
4. **Database Settings**: Uses the existing analyzer mode setting in the settings database

## Benefits

1. **Safe Testing**: Users can test strategies without risking real money
2. **No External Dependencies**: Works without broker accounts or API keys
3. **Realistic Simulation**: Data behaves like real market data with natural price movements
4. **Easy Switching**: Toggle between modes with a single click
5. **Backward Compatibility**: No changes needed to existing client code

## Usage

1. Navigate to the OpenAlgo application
2. Click the mode toggle in the top right corner (next to the profile icon)
3. Switch to "Analyze Mode"
4. The system will automatically reload and start using simulated data
5. All WebSocket connections will now use the simulated adapter
6. Market data will be generated in real-time but will be simulated rather than real

## Technical Details

### Thread Safety

The simulated adapter uses proper thread synchronization:

- A lock is used to protect shared data structures
- The simulation thread can be safely stopped and restarted
- Subscription tracking is thread-safe

### Resource Management

- Proper cleanup of ZeroMQ resources when disconnecting
- Simulation thread is properly terminated when no longer needed
- Memory usage is optimized by limiting stored data to active subscriptions

### Error Handling

- Comprehensive error handling in all adapter methods
- Graceful degradation when errors occur
- Detailed logging for debugging purposes

## Testing

The implementation has been tested to ensure:

1. Proper switching between modes
2. Correct data generation for all modes (LTP, Quote, Depth)
3. Accurate subscription/unsubscription handling
4. Proper resource cleanup
5. Thread safety under concurrent access

## Future Enhancements

Potential improvements that could be made:

1. **Configurable Simulation Parameters**: Allow users to adjust volatility, volume ranges, etc.
2. **Scenario-Based Simulation**: Create specific market scenarios (high volatility, trending markets, etc.)
3. **Historical Data Playback**: Simulate using actual historical data patterns
4. **Performance Optimization**: Optimize for handling large numbers of symbols

## Conclusion

The simulated WebSocket implementation provides a safe and realistic way to test and analyze trading strategies without connecting to real brokers. It maintains full compatibility with the existing OpenAlgo architecture while providing valuable functionality for development, testing, and demonstration purposes.
