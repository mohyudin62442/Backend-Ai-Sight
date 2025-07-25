def analyze_gaps_and_recommend(metrics, llm_results):
    recommendations = []
    # Example: Low citation rate
    citation_rate = metrics.get('citation_rate', 0)
    if citation_rate < 50:
        recommendations.append({
            'priority': 'high',
            'issue': 'Low citation rate',
            'action': 'Increase brand mentions in your content and metadata.',
            'rationale': f'Citation rate is only {citation_rate:.1f}%. Aim for at least 70%.'
        })
    # Example: LLMs not mentioning brand
    for result in llm_results:
        for llm, score in result.get('citation_scores', {}).items():
            if score == 0:
                recommendations.append({
                    'priority': 'medium',
                    'issue': f'{llm} did not mention brand in response to "{result["query"]}"',
                    'action': 'Optimize content for LLM retrieval and citation.',
                    'rationale': f'{llm} failed to cite the brand for a key query.'
                })
    # Add more gap checks as needed
    if not recommendations:
        recommendations.append({
            'priority': 'info',
            'issue': 'No major gaps detected',
            'action': 'Continue monitoring and optimizing.',
            'rationale': 'All key metrics are healthy.'
        })
    return recommendations 