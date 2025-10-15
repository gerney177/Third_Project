import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import traceback

# Import the main service
from investment_advisor_service import InvestmentAdvisorService


class InvestmentAdvisorAPI:
    """
    Flask-based REST API server for Investment Advisor Service.
    Provides endpoints for investment advice, knowledge management, and service statistics.
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 5000,
                 debug: bool = False,
                 enable_gpu: bool = False):
        """
        Initialize the Investment Advisor API server.
        
        Args:
            host (str): Host address to bind the server
            port (int): Port number for the server
            debug (bool): Enable Flask debug mode
            enable_gpu (bool): Enable GPU for LLM processing
        """
        self.host = host
        self.port = port
        self.debug = debug
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['JSON_AS_ASCII'] = False  # Support Korean characters
        
        # Enable CORS for all routes
        CORS(self.app, resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
        
        # Initialize the investment advisor service
        try:
            self.advisor_service = InvestmentAdvisorService(enable_gpu=enable_gpu)
            self.logger.info("Investment Advisor Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Investment Advisor Service: {e}")
            self.advisor_service = None
        
        # Register routes
        self._register_routes()
        
        # Register error handlers
        self._register_error_handlers()
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return self._create_response({
                "status": "healthy",
                "service": "Investment Advisor API",
                "timestamp": datetime.now().isoformat(),
                    "components": {
                        "advisor_service": self.advisor_service is not None,
                        "api_server": True
                    }
            })
        
        @self.app.route('/api/get-advice', methods=['POST'])
        def get_investment_advice():
            """
            Get investment advice based on provided parameters.
            
            Expected JSON payload:
            {
                "stock_symbol": "삼성전자 (005930)",
                "quantity": 10,
                "price": 70000,
                "strategy": "YouTube URL or investment strategy text",
                "reasoning_effort": "high" (optional)
            }
            """
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                data = request.get_json()
                
                # Validate required fields
                required_fields = ["stock_symbol", "quantity", "price", "strategy"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return self._create_error_response(
                        f"Missing required fields: {', '.join(missing_fields)}", 400
                    )
                
                # Validate data types
                try:
                    stock_symbol = str(data["stock_symbol"])
                    quantity = int(data["quantity"])
                    price = float(data["price"])
                    strategy = str(data["strategy"])
                    reasoning_effort = data.get("reasoning_effort", "high")
                except (ValueError, TypeError) as e:
                    return self._create_error_response(
                        f"Invalid data type in request: {str(e)}", 400
                    )
                
                # Validate ranges
                if quantity <= 0:
                    return self._create_error_response("Quantity must be positive", 400)
                
                if price <= 0:
                    return self._create_error_response("Price must be positive", 400)
                
                # Generate investment advice
                self.logger.info(f"Generating advice for {stock_symbol}, {quantity} shares at {price}")
                
                advice_result = self.advisor_service.generate_investment_advice(
                    stock_symbol=stock_symbol,
                    quantity=quantity,
                    price=price,
                    strategy=strategy,
                    reasoning_effort=reasoning_effort
                )
                
                return self._create_response(advice_result)
                
            except Exception as e:
                self.logger.error(f"Error in get_investment_advice: {e}")
                self.logger.error(traceback.format_exc())
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        
        @self.app.route('/api/search-knowledge', methods=['POST'])
        def search_knowledge():
            """
            Search the knowledge base.
            
            Expected JSON payload:
            {
                "query": "search query",
                "top_k": 5 (optional),
                "search_type": "ensemble" (optional: "ensemble", "dense", "bm25")
            }
            """
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                data = request.get_json()
                
                if "query" not in data:
                    return self._create_error_response("Missing 'query' field", 400)
                
                query = str(data["query"])
                top_k = int(data.get("top_k", 5))
                search_type = data.get("search_type", "ensemble")
                
                # Validate parameters
                if top_k <= 0 or top_k > 50:
                    return self._create_error_response("top_k must be between 1 and 50", 400)
                
                if search_type not in ["ensemble", "dense", "bm25"]:
                    return self._create_error_response(
                        "search_type must be 'ensemble', 'dense', or 'bm25'", 400
                    )
                
                self.logger.info(f"Searching knowledge: '{query}' (top_k={top_k}, type={search_type})")
                
                results = self.advisor_service.search_knowledge(
                    query=query,
                    top_k=top_k,
                    search_type=search_type
                )
                
                return self._create_response({
                    "query": query,
                    "search_type": search_type,
                    "results_count": len(results),
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in search_knowledge: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/knowledge-stats', methods=['GET'])
        def get_knowledge_stats():
            """Get knowledge base statistics."""
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                stats = self.advisor_service.get_knowledge_base_stats()
                return self._create_response(stats)
                
            except Exception as e:
                self.logger.error(f"Error in get_knowledge_stats: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/clear-knowledge', methods=['POST'])
        def clear_knowledge():
            """Clear all knowledge from the knowledge base."""
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                # Optional: Add authentication check here
                # if not self._is_authorized(request):
                #     return self._create_error_response("Unauthorized", 401)
                
                self.logger.info("Clearing knowledge base")
                
                result = self.advisor_service.clear_knowledge_base()
                
                if result["success"]:
                    return self._create_response(result)
                else:
                    return self._create_error_response(
                        result.get("error", "Failed to clear knowledge base"), 500
                    )
                
            except Exception as e:
                self.logger.error(f"Error in clear_knowledge: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/service-info', methods=['GET'])
        def get_service_info():
            """Get information about the service and its components."""
            try:
                info = {
                    "service": "Investment Advisor API",
                    "version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        "api_server": {
                            "status": "running",
                            "host": self.host,
                            "port": self.port,
                            "debug": self.debug
                        },
                        "advisor_service": {
                            "available": self.advisor_service is not None,
                            "components": {}
                        }
                    },
                    "endpoints": [
                        "GET /api/health",
                        "POST /api/get-advice",
                        "POST /api/analyze-strategy",
                        "POST /api/search-knowledge",
                        "GET /api/knowledge-stats",
                        "POST /api/clear-knowledge",
                        "GET /api/service-info"
                    ]
                }
                
                if self.advisor_service:
                    stats = self.advisor_service.get_knowledge_base_stats()
                    info["components"]["advisor_service"]["components"] = stats.get("service_info", {})
                
                return self._create_response(info)
                
            except Exception as e:
                self.logger.error(f"Error in get_service_info: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
        @self.app.route('/api/analyze-strategy', methods=['POST'])
        def analyze_strategy():
            """
            외부 투자 전략 글을 분석하여 지식베이스에 추가합니다.
            
            Expected JSON payload:
            {
                "article_url": "https://example.com/investment-strategy"
            }
            """
            try:
                if not self.advisor_service:
                    return self._create_error_response(
                        "Investment Advisor Service not available", 503
                    )
                
                data = request.get_json()
                if not data or 'article_url' not in data:
                    return self._create_error_response("Missing 'article_url' field", 400)
                
                article_url = str(data['article_url'])
                
                # URL 유효성 검사
                if not (article_url.startswith("http://") or article_url.startswith("https://")):
                    return self._create_error_response("Invalid URL format", 400)
                
                self.logger.info(f"Analyzing strategy article: {article_url}")
                
                result = self.advisor_service.analyze_strategy_article(article_url)
                
                if result.get("success"):
                    return self._create_response(result)
                else:
                    return self._create_error_response(
                        result.get("error", "Failed to analyze strategy article"), 400
                    )
                
            except Exception as e:
                self.logger.error(f"Error in analyze_strategy: {e}")
                return self._create_error_response(
                    f"Internal server error: {str(e)}", 500
                )
        
    
    def _register_error_handlers(self):
        """Register error handlers for the Flask app."""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return self._create_error_response("Endpoint not found", 404)
        
        @self.app.errorhandler(405)
        def method_not_allowed(error):
            return self._create_error_response("Method not allowed", 405)
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return self._create_error_response("Internal server error", 500)
    
    def _create_response(self, data: Dict[str, Any], status_code: int = 200) -> Any:
        """
        Create a standardized API response.
        
        Args:
            data (Dict[str, Any]): Response data
            status_code (int): HTTP status code
            
        Returns:
            Flask response object
        """
        response = make_response(jsonify(data), status_code)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    
    def _create_error_response(self, message: str, status_code: int) -> Any:
        """
        Create a standardized error response.
        
        Args:
            message (str): Error message
            status_code (int): HTTP status code
            
        Returns:
            Flask response object
        """
        error_data = {
            "error": True,
            "message": message,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat()
        }
        
        return self._create_response(error_data, status_code)
    
    def run(self):
        """Start the Flask development server."""
        try:
            self.logger.info(f"Starting Investment Advisor API server on {self.host}:{self.port}")
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            raise


def main():
    """Main function to start the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Investment Advisor API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--enable-gpu', action='store_true', help='Enable GPU for LLM')
    
    args = parser.parse_args()
    
    # Create and start the API server
    api_server = InvestmentAdvisorAPI(
        host=args.host,
        port=args.port,
        debug=args.debug,
        enable_gpu=args.enable_gpu
    )
    
    print(f"""
=== Investment Advisor API Server ===
Server starting on: http://{args.host}:{args.port}
Debug mode: {args.debug}
GPU enabled: {args.enable_gpu}

Available endpoints:
- GET  /api/health              - Health check
- POST /api/get-advice          - Get investment advice
- POST /api/analyze-strategy    - Analyze strategy article
- POST /api/search-knowledge    - Search knowledge base
- GET  /api/knowledge-stats     - Get knowledge statistics
- POST /api/clear-knowledge     - Clear knowledge base
- GET  /api/service-info        - Get service information

Example usage:
curl -X POST http://localhost:5000/api/get-advice \\
  -H "Content-Type: application/json" \\
  -d '{
    "stock_symbol": "005930",
    "quantity": 10,
    "price": 70000,
    "strategy": "External article URL or investment strategy"
  }'

Press Ctrl+C to stop the server.
    """)
    
    try:
        api_server.run()
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    main()
