# scripts/tools.py

import json
from ddgs import DDGS

def web_search(query: str, num_results: int = 5):
    """
    Performs a web search using DuckDuckGo and returns the top results.

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: A formatted string of the search results, or an error message.
    """
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]

        if not results:
            return "No results found."

        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append(f"[{i+1}] {result['title']}\n{result['body']}\nURL: {result['href']}")

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a web search using DuckDuckGo to find information on a given topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The desired number of search results to return.",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def get_tools_json():
    """Returns the list of tools in JSON format for the model."""
    return json.dumps(TOOLS, indent=2)
