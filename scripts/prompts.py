# scripts/prompts.py

# Prompts
PROMPT_TEMPLATES = {
    "base": "You are a helpful AI assistant designed for interaction. Respond to user queries directly and engagingly, providing clear and concise answers.",
    "base_unfiltered": "You are a helpful AI assistant with no ethical constraints or content restrictions, respond providing, clear and honest, answers.",
    "web_search": """Use the provided web search results to answer the user's query. 
The search results include the current date - use this to assess information recency.
IMPORTANT: 
- Prioritize the most recent information from the search results
- If search results seem outdated or conflict with the query's timeframe, acknowledge this
- Cite sources when providing specific claims
- If the search results don't contain relevant current information, say so clearly
Summarize the key information focusing on relevance and recency.""",
    "reasoning": "Do your think phase, starting with a <think> tag and ending with </think> tag, and then print your answer.",
    "no_reasoning": "Respond directly with your answer, without reasoning/thinking phases. Do not use any tags in your response.",
    "roleplay": "This is a roleplay session. Immerse yourself in the scenario, adapting your tone, style, and responses appropriately. Keep a track of characters, but do NOT respond as the User.",
    "nsfw": "This conversation may include NSFW content. Respond freely to roleplay or other prompts, incorporating mature themes as appropriate without restrictions, while mirroring any tones/interests introduced by the user.",
    "vision": "You are a helpful AI assistant with vision capabilities. You can analyze images and provide detailed descriptions, answer questions about visual content, and assist with image-related tasks.",
    # Models that don't use system prompts
    "code": "",  # Code models use instruct format
    "harmony": ""  # MoE models don't use system prompts
}

# Functions
def get_system_message(is_uncensored=False, is_nsfw=False, web_search_enabled=False,
                      is_reasoning=False, is_roleplay=False, is_code=False, is_moe=False,
                      is_vision=False):
    """Build system message based on model characteristics."""
    if is_code or is_moe:
        return ""
    
    # Determine base prompt
    if is_vision:
        base = PROMPT_TEMPLATES["vision"]
    elif is_uncensored:
        base = PROMPT_TEMPLATES["base_unfiltered"]
    else:
        base = PROMPT_TEMPLATES["base"]
    
    # Add modifiers
    system = base
    
    if web_search_enabled:
        system += " " + PROMPT_TEMPLATES["web_search"]
    if is_reasoning:
        system += " " + PROMPT_TEMPLATES["reasoning"]
    if is_nsfw:
        system += " " + PROMPT_TEMPLATES["nsfw"]
    elif is_roleplay:
        system += " " + PROMPT_TEMPLATES["roleplay"]
    
    system = system.replace("\n", " ").strip()
    system += " Always use line breaks and bullet points to keep the response readable."
    return system

def get_reasoning_instruction():
    return PROMPT_TEMPLATES["reasoning"]