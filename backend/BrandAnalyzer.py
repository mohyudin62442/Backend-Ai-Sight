from llm_clients import call_openai, call_anthropic
from chromadb_utils import add_content_to_chromadb, query_chromadb
import re
from gap_analyzer import analyze_gaps_and_recommend

class BrandAnalyzer:
    def __init__(self, llms=("openai", "anthropic")):
        self.llms = llms

    async def analyze(self, brand_name: str, content_list=None, queries=None):
        # 1. Store content in ChromaDB
        if content_list:
            for idx, content in enumerate(content_list):
                add_content_to_chromadb(f"{brand_name}_{idx}", content, metadata={"brand": brand_name})

        # 2. Generate queries if not provided
        if not queries:
            queries = [f"What is {brand_name}?", f"Where can I buy {brand_name}?", f"Who owns {brand_name}?", f"Is {brand_name} a good brand?"]

        llm_results = []
        for query in queries:
            # 3. Semantic search using ChromaDB
            semantic_results = query_chromadb(query, top_k=3)
            responses = {}
            if "openai" in self.llms:
                responses["openai"] = await call_openai(query)
            if "anthropic" in self.llms:
                responses["anthropic"] = await call_anthropic(query)
            # 4. Score LLM responses for citation/brand mention
            scores = {}
            for llm, resp in responses.items():
                # Simple scoring: does the response mention the brand?
                mention = bool(re.search(rf"\\b{re.escape(brand_name)}\\b", resp, re.IGNORECASE))
                scores[llm] = int(mention)
            llm_results.append({
                'query': query,
                'semantic_results': semantic_results,
                'llm_responses': responses,
                'citation_scores': scores
            })
        # 5. Calculate metrics
        total = len(llm_results) * len(self.llms)
        cited = sum(sum(r['citation_scores'].values()) for r in llm_results)
        citation_rate = (cited / total * 100) if total else 0
        metrics = {
            'citation_rate': citation_rate,
            'total_queries': len(llm_results),
            'llms_used': self.llms
        }
        # 6. Recommendations (use gap analyzer)
        recommendations = analyze_gaps_and_recommend(metrics, llm_results)
        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'llm_results': llm_results
        } 