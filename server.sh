#!/bin/bash
# Helper script to start/stop the Flask API server

PORT=5001
SERVER_FILE="api_server.py"

start_server() {
    echo "🚀 Starting Flask API server on port $PORT..."
    echo ""
    echo "API Endpoints:"
    echo "  - POST /api/detect-human (upload image)"
    echo "  - POST /api/analyze-audio (upload video)"
    echo "  - GET  /api/health"
    echo ""
    echo "Server running on http://localhost:$PORT"
    echo "Press Ctrl+C to stop"
    echo ""
    python3 $SERVER_FILE
}

stop_server() {
    echo "🛑 Stopping Flask API server on port $PORT..."
    PIDS=$(lsof -ti:$PORT)
    
    if [ -z "$PIDS" ]; then
        echo "No server running on port $PORT"
    else
        echo "Found processes: $PIDS"
        kill -9 $PIDS 2>/dev/null
        echo "✅ Server stopped"
    fi
}

check_status() {
    echo "📊 Checking server status on port $PORT..."
    PIDS=$(lsof -ti:$PORT)
    
    if [ -z "$PIDS" ]; then
        echo "❌ Server is NOT running on port $PORT"
    else
        echo "✅ Server is running on port $PORT"
        echo "Process IDs: $PIDS"
        echo ""
        curl -s http://localhost:$PORT/api/health 2>/dev/null || echo "⚠️  Server not responding to health check"
    fi
}

case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    status)
        check_status
        ;;
    restart)
        stop_server
        sleep 1
        start_server
        ;;
    *)
        echo "Usage: ./server.sh {start|stop|status|restart}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the Flask API server"
        echo "  stop    - Stop the Flask API server"
        echo "  status  - Check if server is running"
        echo "  restart - Restart the Flask API server"
        exit 1
        ;;
esac

