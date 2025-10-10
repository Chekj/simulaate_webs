The issue is that when the server restarts, the script is not properly handling the cleanup of the WebSocket server process. Let's examine the problem more closely.The root cause appears to be that the script modifies the WebSocket server to check for analyzer mode, but when the server restarts, there might be conflicts with the existing WebSocket process or the analyzer mode detection is not working correctly.Let's create a fixed version of the simulated_websocket2.sh script:


The main issues were:
Improper Process Management: The original script didn't properly handle cleanup of existing WebSocket processes during restarts, leading to port conflicts and zombie processes.
Incompatible Broker Factory Changes: The script completely replaced the broker_factory.py file with a hardcoded implementation, breaking the dynamic broker adapter loading mechanism used by the stock OpenAlgo.
Missing Cleanup Mechanisms: No proper cleanup of WebSocket processes before starting new ones, causing conflicts during server restarts.
Key Fixes Implemented
Enhanced Process Management: Added a cleanup script that kills any existing WebSocket proxy processes and releases port 8765 before starting new ones.
Improved Broker Factory Implementation: Maintained the original dynamic broker adapter loading mechanism while adding the simulated adapter as an optional component.
Updated Startup Script: Modified start.sh to include proper cleanup before starting and added signal handlers for graceful shutdown.
Robust Error Handling: Added proper error handling in the simulated adapter and improved logging.
