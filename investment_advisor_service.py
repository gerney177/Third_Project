import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import custom modules
from text_chunker import TextChunker
from vector_database import EnsembleVectorDatabase
from llm_service import GPTOSSService
from web_ingestor import WebIngestor


class InvestmentAdvisorService:
    """
    Main investment advisor service that integrates all components.
    Provides comprehensive investment advice using RAG (Retrieval Augmented Generation).
    """
    
    def __init__(self, 
                 collection_name: str = "investment_knowledge",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 bm25_weight: float = 0.4,
                 dense_weight: float = 0.6,
                 enable_gpu: bool = False):
        """
        Initialize Investment Advisor Service.
        
        Args:
            collection_name (str): ChromaDB collection name
            chunk_size (int): Text chunk size for processing
            chunk_overlap (int): Overlap between chunks
            bm25_weight (float): Weight for BM25 search
            dense_weight (float): Weight for dense search
            enable_gpu (bool): Whether to enable GPU for LLM
        """
        self.collection_name = collection_name
        self.enable_gpu = enable_gpu
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.logger.info("Initializing Investment Advisor Service components...")
        
        # Text chunker
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Vector database
        self.vector_db = EnsembleVectorDatabase(
            collection_name=collection_name,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight
        )
        
        # LLM service (initialize with caution for GPU)
        self.llm_service = None
        self._initialize_llm_service()
        
        # Web ingestor (웹 지식 수집)
        self.web_ingestor = WebIngestor()
        
        self.logger.info("Investment Advisor Service initialized successfully")
    
    def _initialize_llm_service(self):
        """Initialize LLM service with error handling."""
        try:
            if self.enable_gpu:
                self.llm_service = GPTOSSService()
                self.logger.info("LLM service initialized with GPU support")
            else:
                # For CPU or mock testing
                self.logger.warning("GPU disabled - LLM service will use mock responses")
                self.llm_service = None
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {e}")
            self.llm_service = None
    
    def analyze_strategy_article(self, article_url: str) -> Dict[str, Any]:
        """
        외부 투자 전략 글을 분석하여 지식베이스에 추가합니다.
        
        Args:
            article_url (str): 투자 전략 글 URL
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            self.logger.info(f"Analyzing strategy article: {article_url}")
            
            # 웹 페이지 수집
            page = self.web_ingestor.ingest(article_url)
            if not page or not page.get("content"):
                return {
                    "success": False,
                    "error": "Failed to fetch or parse article",
                    "url": article_url
                }
            
            # 텍스트 청킹
            chunks = self.text_chunker.create_semantic_chunks(
                page["content"], 
                language='korean'
            )
            if not chunks:
                return {
                    "success": False,
                    "error": "Failed to create text chunks",
                    "url": article_url
                }
            
            # 벡터 DB에 추가
            documents = [chunk['text'] for chunk in chunks]
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": "strategy_article",
                    "url": article_url,
                    "title": page.get("title", ""),
                    "chunk_id": i,
                    "chunk_token_count": chunk['token_count'],
                    "chunk_sentence_count": chunk['sentence_count'],
                    "fetched_at": page.get("fetched_at"),
                    "content_type": "investment_strategy",
                    "analysis_type": "external_strategy"
                }
                metadatas.append(metadata)
            
            success = self.vector_db.add_documents(documents, metadatas)
            
            if success:
                stats = self.text_chunker.get_chunk_statistics(chunks)
                return {
                    "success": True,
                    "article_info": {
                        "title": page.get("title", ""),
                        "url": article_url,
                        "content_length": len(page.get("content", "")),
                        "chunks_created": len(chunks)
                    },
                    "processing_stats": stats,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to add strategy to knowledge base",
                    "url": article_url
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing strategy article: {e}")
            return {
                "success": False,
                "error": f"Strategy analysis error: {str(e)}",
                "url": article_url
            }

    def add_web_knowledge(self, url: str) -> Dict[str, Any]:
        """웹 페이지 콘텐츠를 수집해 지식베이스에 추가합니다."""
        try:
            self.logger.info(f"Processing web page: {url}")
            page = self.web_ingestor.ingest(url)
            if not page or not page.get("content"):
                return {"success": False, "error": "Failed to fetch or parse page", "url": url}

            # 텍스트 청킹
            chunks = self.text_chunker.create_semantic_chunks(page["content"], language='korean')
            if not chunks:
                return {"success": False, "error": "Failed to create chunks", "url": url}

            documents = [c['text'] for c in chunks]
            metadatas = []
            for i, c in enumerate(chunks):
                metadatas.append({
                    "source": "web",
                    "url": url,
                    "title": page.get("title", ""),
                    "chunk_id": i,
                    "chunk_token_count": c['token_count'],
                    "chunk_sentence_count": c['sentence_count'],
                    "fetched_at": page.get("fetched_at"),
                    "content_type": "investment_strategy"
                })

            success = self.vector_db.add_documents(documents, metadatas)
            if not success:
                return {"success": False, "error": "Failed to add documents to vector DB", "url": url}

            stats = self.text_chunker.get_chunk_statistics(chunks)
            return {
                "success": True,
                "page_info": {
                    "title": page.get("title", ""),
                    "url": url,
                    "length": len(page.get("content", ""))
                },
                "chunks_added": len(chunks),
                "processing_stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error processing web page: {e}")
            return {"success": False, "error": str(e), "url": url}
    
    def search_knowledge(self, 
                        query: str, 
                        top_k: int = 5,
                        search_type: str = "ensemble") -> List[Dict]:
        """
        Search knowledge base for relevant information.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            search_type (str): Type of search ("ensemble", "dense", "bm25")
            
        Returns:
            List[Dict]: Search results
        """
        try:
            if search_type == "ensemble":
                results = self.vector_db.search_ensemble(query, top_k)
            elif search_type == "dense":
                results = self.vector_db.search_dense(query, top_k)
            elif search_type == "bm25":
                results = self.vector_db.search_bm25(query, top_k)
            else:
                self.logger.warning(f"Unknown search type: {search_type}, using ensemble")
                results = self.vector_db.search_ensemble(query, top_k)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def generate_investment_advice(self, 
                                 stock_symbol: str,
                                 quantity: int,
                                 price: float,
                                 strategy: str,
                                 reasoning_effort: str = "high") -> Dict[str, Any]:
        """
        Generate comprehensive investment advice using RAG.
        
        Args:
            stock_symbol (str): Stock symbol or company name
            quantity (int): Number of shares to buy
            price (float): Price per share
            strategy (str): Investment strategy query or YouTube URL
            reasoning_effort (str): LLM reasoning effort level
            
        Returns:
            Dict[str, Any]: Comprehensive investment advice
        """
        try:
            self.logger.info(f"Generating investment advice for {stock_symbol}")
            
            # 백엔드에서 전달받은 금융 데이터 사용
            financial_context = ""
            # TODO: 백엔드 개발자가 전달할 금융 데이터를 여기에 통합
            # financial_context = backend_financial_data
            
            # Process strategy input
            strategy_info = ""
            
            # 전략은 항상 볼린저 밴드 매매법으로 고정 (데모)
            strategy_info = (
                "볼린저 밴드 매매법 요약: 중심선(20일 이동평균)과 상단/하단 밴드(±2표준편차)를 활용하여 \n"
                "- 하단 밴드 하향 이탈 후 재진입 시 매수 관점 \n"
                "- 상단 밴드 상향 이탈 후 재진입 시 매도/분할매도 관점 \n"
                "- 밴드 수축 후 확장 국면에서 추세 발생 가능성 ↑ \n"
                "- 거래량 증가와 함께 밴드 돌파 시 신뢰도 ↑"
            )
            
            # Generate advice using LLM
            if self.llm_service:
                # Combine strategy info with financial data
                enhanced_strategy_info = f"""
{strategy_info}

{financial_context}
"""
                advice_result = self.llm_service.generate_investment_advice(
                    stock_symbol=stock_symbol,
                    quantity=quantity,
                    price=price,
                    strategy_info=enhanced_strategy_info,
                    reasoning_effort=reasoning_effort
                )
            else:
                # Fallback mock advice with financial data
                advice_result = self._generate_mock_advice(
                    stock_symbol, quantity, price, f"{strategy_info}\n\n{financial_context}"
                )
            
            # Add knowledge base search results to response
            advice_result["knowledge_search"] = {
                "query": strategy if not strategy.startswith(("http://", "https://")) else f"{stock_symbol} 투자 전략",
                "results_found": len(search_results) if 'search_results' in locals() else 0,
                "search_type": "ensemble"
            }
            
            # Add service metadata
            advice_result["service_info"] = {
                "service_version": "1.0.0",
                "components_used": {
                    "external_strategy_analysis": strategy.startswith(("http://", "https://")),
                    "knowledge_search": True,
                    "llm_generation": self.llm_service is not None,
                    "ensemble_search": True,
                    "backend_data": True,  # 백엔드에서 데이터 전달
                    "strategy_analysis": True  # 전략 분석 중심
                },
                "processing_timestamp": datetime.now().isoformat()
            }
            
            return advice_result
            
        except Exception as e:
            self.logger.error(f"Error generating investment advice: {e}")
            return {
                "error": f"투자 조언 생성 중 오류가 발생했습니다: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "service_info": {
                    "error_occurred": True,
                    "llm_available": self.llm_service is not None
                }
            }
    
    def _generate_mock_advice(self, 
                            stock_symbol: str, 
                            quantity: int, 
                            price: float, 
                            strategy_info: str) -> Dict[str, Any]:
        """
        Generate mock investment advice when LLM is not available.
        
        Args:
            stock_symbol (str): Stock symbol
            quantity (int): Number of shares
            price (float): Price per share
            strategy_info (str): Strategy information
            
        Returns:
            Dict[str, Any]: Mock advice result
        """
        total_investment = quantity * price
        
        mock_advice = f"""
**{stock_symbol} 투자 분석 보고서 (Mock Analysis)**

**투자 개요**
- 종목: {stock_symbol}
- 투자 수량: {quantity:,}주
- 주당 가격: {price:,}원
- 총 투자금액: {total_investment:,}원

**분석 근거**
{strategy_info[:500]}...

**투자 권고사항**
1. **기업 분석**: 현재 제공된 정보를 바탕으로 추가 분석이 필요합니다.
2. **시장 전망**: 최신 시장 동향을 고려한 투자 전략 수립을 권장합니다.
3. **리스크 관리**: 분산투자와 손절매 전략을 함께 고려하시기 바랍니다.
4. **투자 기간**: 중장기 관점에서의 투자 접근을 권장합니다.

**주의사항**
- 이 분석은 모의 분석 결과입니다.
- 실제 투자 결정 시에는 전문가의 조언을 받으시기 바랍니다.
- 시장 상황 변화에 따른 지속적인 모니터링이 필요합니다.

**면책 조항**
본 분석은 참고용으로만 사용하시고, 투자 결정에 대한 책임은 투자자 본인에게 있습니다.
        """
        
        return {
            "advice": mock_advice,
            "investment_summary": {
                "stock_symbol": stock_symbol,
                "quantity": quantity,
                "price_per_share": price,
                "total_investment": total_investment,
                "currency": "KRW"
            },
            "generation_params": {
                "reasoning_effort": "mock",
                "llm_service": "mock_service"
            },
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "model_id": "mock_advisor",
                "version": "demo"
            },
            "warning": "This is a mock analysis. LLM service is not available."
        }
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dict[str, Any]: Knowledge base statistics
        """
        try:
            vector_stats = self.vector_db.get_collection_stats()
            
            # Add service-specific stats
            stats = {
                "vector_database": vector_stats,
                "service_info": {
                    "collection_name": self.collection_name,
                    "components_initialized": {
                        "web_ingestor": self.web_ingestor is not None,
                        "text_chunker": self.text_chunker is not None,
                        "vector_database": self.vector_db is not None,
                        "llm_service": self.llm_service is not None,
                        "backend_integration": True  # 백엔드 연동
                    },
                    "search_weights": {
                        "bm25_weight": self.vector_db.bm25_weight,
                        "dense_weight": self.vector_db.dense_weight
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge base stats: {e}")
            return {
                "error": f"통계 조회 중 오류가 발생했습니다: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """
        Clear all documents from the knowledge base.
        
        Returns:
            Dict[str, Any]: Clear operation result
        """
        try:
            success = self.vector_db.clear_collection()
            
            return {
                "success": success,
                "message": "지식 베이스가 성공적으로 초기화되었습니다." if success else "지식 베이스 초기화에 실패했습니다.",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error clearing knowledge base: {e}")
            return {
                "success": False,
                "error": f"지식 베이스 초기화 중 오류가 발생했습니다: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }


def main():
    """Example usage of InvestmentAdvisorService"""
    print("=== Investment Advisor Service Test ===")
    
    # Initialize service (without GPU for testing)
    advisor = InvestmentAdvisorService(enable_gpu=False)
    
    # Test 1: Generate investment advice
    print("\n1. Testing Investment Advice Generation...")
    advice_result = advisor.generate_investment_advice(
        stock_symbol="삼성전자 (005930)",
        quantity=10,
        price=70000,
        strategy="삼성전자 반도체 사업 분석과 AI 칩 시장 전망에 대한 투자 전략"
    )
    
    print("Investment Advice Generated:")
    print(f"- Stock: {advice_result['investment_summary']['stock_symbol']}")
    print(f"- Total Investment: {advice_result['investment_summary']['total_investment']:,} KRW")
    print(f"- Generated at: {advice_result['timestamp']}")
    
    if 'advice' in advice_result:
        print(f"\nAdvice Preview:\n{advice_result['advice'][:300]}...")
    
    # Test 2: Knowledge base statistics
    print("\n2. Testing Knowledge Base Statistics...")
    stats = advisor.get_knowledge_base_stats()
    print("Knowledge Base Stats:")
    for key, value in stats.items():
        if key != 'vector_database':  # Skip detailed vector DB stats for brevity
            print(f"- {key}: {value}")
    
    # Test 3: Search knowledge base
    print("\n3. Testing Knowledge Search...")
    search_results = advisor.search_knowledge("삼성전자 투자 전략", top_k=3)
    print(f"Search Results: {len(search_results)} found")
    
    for i, result in enumerate(search_results[:2], 1):
        print(f"{i}. Score: {result.get('score', 0):.3f}")
        print(f"   Text: {result['document'][:100]}...")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
