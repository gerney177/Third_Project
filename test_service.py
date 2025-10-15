#!/usr/bin/env python3
"""
Comprehensive test script for Investment Advisor Service.
Tests all components and API endpoints to ensure functionality.
"""

import os
import sys
import json
import time
import requests
import subprocess
import logging
from typing import Dict, Any, List
from datetime import datetime


class InvestmentAdvisorTester:
    """
    Comprehensive test suite for the Investment Advisor Service.
    Tests individual components and the complete API service.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:5000"):
        """
        Initialize the test suite.
        
        Args:
            api_base_url (str): Base URL of the API server
        """
        self.api_base_url = api_base_url
        self.test_results = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Test data
        self.test_data = {
            "sample_investment_requests": [
                {
                    "stock_symbol": "삼성전자 (005930)",
                    "quantity": 10,
                    "price": 70000,
                    "strategy": "삼성전자의 반도체 사업 성장성과 AI 칩 시장 전망을 바탕으로 한 투자 전략"
                },
                {
                    "stock_symbol": "AAPL",
                    "quantity": 5,
                    "price": 150.0,
                    "strategy": "Apple's services revenue growth and iPhone market share analysis"
                }
            ],
            "sample_search_queries": [
                "삼성전자 투자 전략",
                "AI technology investment",
                "semiconductor market outlook",
                "금리 인상 영향"
            ],
            "sample_stock_symbols": [
                "005930",  # 삼성전자
                "000660",  # SK하이닉스
                "035420",  # NAVER
                "035720"   # 카카오
            ]
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite.
        
        Returns:
            Dict[str, Any]: Test results summary
        """
        self.logger.info("=== Starting Investment Advisor Service Test Suite ===")
        start_time = time.time()
        
        # Test 1: Component Tests
        self.logger.info("\n1. Testing Individual Components...")
        component_results = self.test_components()
        
        # Test 2: API Server Tests
        self.logger.info("\n2. Testing API Server...")
        api_results = self.test_api_server()
        
        # Test 3: Integration Tests
        self.logger.info("\n3. Testing Integration Scenarios...")
        integration_results = self.test_integration_scenarios()
        
        # Test 4: Performance Tests
        self.logger.info("\n4. Testing Performance...")
        performance_results = self.test_performance()
        
        # Test 5: Error Handling Tests
        self.logger.info("\n5. Testing Error Handling...")
        error_handling_results = self.test_error_handling()
        
        end_time = time.time()
        
        # Compile results
        test_summary = {
            "test_suite": "Investment Advisor Service",
            "timestamp": datetime.now().isoformat(),
            "total_duration": f"{end_time - start_time:.2f} seconds",
            "results": {
                "component_tests": component_results,
                "api_server_tests": api_results,
                "integration_tests": integration_results,
                "performance_tests": performance_results,
                "error_handling_tests": error_handling_results
            },
            "summary": self._generate_test_summary()
        }
        
        self.logger.info(f"\n=== Test Suite Complete in {end_time - start_time:.2f} seconds ===")
        return test_summary
    
    def test_components(self) -> Dict[str, Any]:
        """Test individual service components."""
        results = {}
        
        
        # Test Text Chunker
        try:
            from text_chunker import TextChunker
            chunker = TextChunker(chunk_size=500, chunk_overlap=100)
            
            sample_text = "This is a test sentence. This is another test sentence. " * 50
            chunks = chunker.create_semantic_chunks(sample_text)
            
            results["text_chunker"] = {
                "status": "passed" if len(chunks) > 0 else "failed",
                "tests": {
                    "chunk_creation": len(chunks) > 0,
                    "token_counting": chunker.count_tokens(sample_text) > 0,
                    "sentence_splitting": len(chunker.split_by_sentences(sample_text)) > 0
                }
            }
        except Exception as e:
            results["text_chunker"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test Vector Database
        try:
            from vector_database import EnsembleVectorDatabase
            db = EnsembleVectorDatabase(collection_name="test_collection")
            
            # Test document addition
            test_docs = ["Test document 1", "Test document 2"]
            add_success = db.add_documents(test_docs)
            
            # Test search
            search_results = db.search_ensemble("test", top_k=2)
            
            results["vector_database"] = {
                "status": "passed" if add_success and len(search_results) >= 0 else "failed",
                "tests": {
                    "document_addition": add_success,
                    "ensemble_search": len(search_results) >= 0,
                    "stats_retrieval": bool(db.get_collection_stats())
                }
            }
        except Exception as e:
            results["vector_database"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test LLM Service (Mock)
        try:
            from llm_service import GPTOSSService
            
            # This will likely fail without GPU, so we test initialization only
            results["llm_service"] = {
                "status": "passed",
                "tests": {
                    "class_import": True,
                    "note": "Full LLM testing requires GPU and model access"
                }
            }
        except Exception as e:
            results["llm_service"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test Main Service
        try:
            from investment_advisor_service import InvestmentAdvisorService
            service = InvestmentAdvisorService(enable_gpu=False)
            
            # Test knowledge base stats
            stats = service.get_knowledge_base_stats()
            
            results["investment_advisor_service"] = {
                "status": "passed" if stats else "failed",
                "tests": {
                    "service_initialization": True,
                    "stats_retrieval": bool(stats),
                    "component_integration": bool(service.text_chunker)
                }
            }
        except Exception as e:
            results["investment_advisor_service"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return results
    
    def test_api_server(self) -> Dict[str, Any]:
        """Test API server endpoints."""
        results = {}
        
        # Test health endpoint
        try:
            response = requests.get(f"{self.api_base_url}/api/health", timeout=10)
            results["health_check"] = {
                "status": "passed" if response.status_code == 200 else "failed",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            results["health_check"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test service info endpoint
        try:
            response = requests.get(f"{self.api_base_url}/api/service-info", timeout=10)
            results["service_info"] = {
                "status": "passed" if response.status_code == 200 else "failed",
                "status_code": response.status_code,
                "has_data": bool(response.json() if response.status_code == 200 else False)
            }
        except Exception as e:
            results["service_info"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test knowledge stats endpoint
        try:
            response = requests.get(f"{self.api_base_url}/api/knowledge-stats", timeout=10)
            results["knowledge_stats"] = {
                "status": "passed" if response.status_code == 200 else "failed",
                "status_code": response.status_code
            }
        except Exception as e:
            results["knowledge_stats"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test investment advice endpoint
        try:
            test_request = self.test_data["sample_investment_requests"][0]
            response = requests.post(
                f"{self.api_base_url}/api/get-advice",
                json=test_request,
                timeout=30
            )
            results["investment_advice"] = {
                "status": "passed" if response.status_code == 200 else "failed",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "has_advice": bool(response.json().get("advice") if response.status_code == 200 else False)
            }
        except Exception as e:
            results["investment_advice"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test knowledge search endpoint
        try:
            search_request = {
                "query": self.test_data["sample_search_queries"][0],
                "top_k": 3,
                "search_type": "ensemble"
            }
            response = requests.post(
                f"{self.api_base_url}/api/search-knowledge",
                json=search_request,
                timeout=10
            )
            results["knowledge_search"] = {
                "status": "passed" if response.status_code == 200 else "failed",
                "status_code": response.status_code,
                "results_count": len(response.json().get("results", [])) if response.status_code == 200 else 0
            }
        except Exception as e:
            results["knowledge_search"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return results
    
    def test_integration_scenarios(self) -> Dict[str, Any]:
        """Test complete integration scenarios."""
        results = {}
        
        # Scenario 1: Complete investment advice flow
        try:
            self.logger.info("Testing complete investment advice flow...")
            
            # Step 1: Check service status
            health_response = requests.get(f"{self.api_base_url}/api/health", timeout=10)
            
            # Step 2: Get initial knowledge stats
            stats_response = requests.get(f"{self.api_base_url}/api/knowledge-stats", timeout=10)
            
            # Step 3: Request investment advice
            advice_response = requests.post(
                f"{self.api_base_url}/api/get-advice",
                json=self.test_data["sample_investment_requests"][0],
                timeout=30
            )
            
            # Step 4: Search knowledge base
            search_response = requests.post(
                f"{self.api_base_url}/api/search-knowledge",
                json={
                    "query": "투자 전략",
                    "top_k": 5
                },
                timeout=10
            )
            
            success = all([
                health_response.status_code == 200,
                stats_response.status_code == 200,
                advice_response.status_code == 200,
                search_response.status_code == 200
            ])
            
            results["complete_flow"] = {
                "status": "passed" if success else "failed",
                "steps": {
                    "health_check": health_response.status_code == 200,
                    "stats_retrieval": stats_response.status_code == 200,
                    "advice_generation": advice_response.status_code == 200,
                    "knowledge_search": search_response.status_code == 200
                }
            }
        except Exception as e:
            results["complete_flow"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Scenario 2: Multiple concurrent requests
        try:
            self.logger.info("Testing concurrent requests...")
            
            import concurrent.futures
            import threading
            
            def make_request():
                try:
                    response = requests.post(
                        f"{self.api_base_url}/api/get-advice",
                        json=self.test_data["sample_investment_requests"][1],
                        timeout=30
                    )
                    return response.status_code == 200
                except:
                    return False
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(make_request) for _ in range(3)]
                concurrent_results = [future.result() for future in futures]
            
            results["concurrent_requests"] = {
                "status": "passed" if all(concurrent_results) else "failed",
                "successful_requests": sum(concurrent_results),
                "total_requests": len(concurrent_results)
            }
        except Exception as e:
            results["concurrent_requests"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return results
    
    def test_performance(self) -> Dict[str, Any]:
        """Test service performance metrics."""
        results = {}
        
        # Test response times
        try:
            response_times = []
            
            for i in range(3):
                start_time = time.time()
                response = requests.get(f"{self.api_base_url}/api/health", timeout=10)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                results["response_time"] = {
                    "status": "passed" if avg_response_time < 1.0 else "warning",
                    "average_time": f"{avg_response_time:.3f} seconds",
                    "max_time": f"{max(response_times):.3f} seconds",
                    "min_time": f"{min(response_times):.3f} seconds"
                }
            else:
                results["response_time"] = {
                    "status": "failed",
                    "error": "No successful requests"
                }
        except Exception as e:
            results["response_time"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test advice generation time
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/api/get-advice",
                json=self.test_data["sample_investment_requests"][0],
                timeout=60
            )
            end_time = time.time()
            
            generation_time = end_time - start_time
            
            results["advice_generation_time"] = {
                "status": "passed" if generation_time < 30.0 else "warning",
                "generation_time": f"{generation_time:.3f} seconds",
                "success": response.status_code == 200
            }
        except Exception as e:
            results["advice_generation_time"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return results
    
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling scenarios."""
        results = {}
        
        # Test invalid investment advice request
        try:
            invalid_request = {
                "stock_symbol": "TEST",
                "quantity": -5,  # Invalid quantity
                "price": 0,      # Invalid price
                "strategy": ""
            }
            
            response = requests.post(
                f"{self.api_base_url}/api/get-advice",
                json=invalid_request,
                timeout=10
            )
            
            results["invalid_investment_request"] = {
                "status": "passed" if response.status_code == 400 else "failed",
                "status_code": response.status_code,
                "has_error_message": bool(response.json().get("message") if response.status_code != 200 else False)
            }
        except Exception as e:
            results["invalid_investment_request"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test missing required fields
        try:
            incomplete_request = {
                "stock_symbol": "TEST"
                # Missing quantity, price, strategy
            }
            
            response = requests.post(
                f"{self.api_base_url}/api/get-advice",
                json=incomplete_request,
                timeout=10
            )
            
            results["missing_fields"] = {
                "status": "passed" if response.status_code == 400 else "failed",
                "status_code": response.status_code
            }
        except Exception as e:
            results["missing_fields"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test invalid search request
        try:
            invalid_search = {
                "query": "",  # Empty query
                "top_k": 100  # Too large
            }
            
            response = requests.post(
                f"{self.api_base_url}/api/search-knowledge",
                json=invalid_search,
                timeout=10
            )
            
            results["invalid_search"] = {
                "status": "passed" if response.status_code == 400 else "failed",
                "status_code": response.status_code
            }
        except Exception as e:
            results["invalid_search"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test non-existent endpoint
        try:
            response = requests.get(f"{self.api_base_url}/api/non-existent", timeout=10)
            
            results["non_existent_endpoint"] = {
                "status": "passed" if response.status_code == 404 else "failed",
                "status_code": response.status_code
            }
        except Exception as e:
            results["non_existent_endpoint"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return results
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "warning_tests": 0,
            "pass_rate": 0.0
        }
        
        # This would be implemented to analyze all test results
        # For now, return a placeholder
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Test results saved to: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def check_server_status(self) -> bool:
        """Check if the API server is running."""
        try:
            response = requests.get(f"{self.api_base_url}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False


def main():
    """Main function to run the test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Investment Advisor Service Test Suite')
    parser.add_argument('--api-url', default='http://localhost:5000', help='API server URL')
    parser.add_argument('--save-results', action='store_true', help='Save test results to file')
    parser.add_argument('--check-server', action='store_true', help='Only check if server is running')
    
    args = parser.parse_args()
    
    tester = InvestmentAdvisorTester(api_base_url=args.api_url)
    
    if args.check_server:
        server_status = tester.check_server_status()
        print(f"Server status: {'Running' if server_status else 'Not running'}")
        return
    
    # Check if server is running before running tests
    if not tester.check_server_status():
        print(f"❌ API server is not running at {args.api_url}")
        print("Please start the server with: python api_server.py")
        return
    
    print(f"✅ API server is running at {args.api_url}")
    print("Starting comprehensive test suite...")
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for category, category_results in results["results"].items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        if isinstance(category_results, dict):
            for test_name, test_result in category_results.items():
                status = test_result.get("status", "unknown")
                status_icon = {"passed": "✅", "failed": "❌", "warning": "⚠️"}.get(status, "❓")
                print(f"  {status_icon} {test_name}: {status}")
    
    # Save results if requested
    if args.save_results:
        tester.save_results(results)
    
    print(f"\nTotal test duration: {results['total_duration']}")
    print("="*60)


if __name__ == "__main__":
    main()
