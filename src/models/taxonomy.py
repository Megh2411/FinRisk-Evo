def classify_risk_type(fres_score, fy25_text, topic_distribution=None):
    if fres_score == 0.0: 
        return "STATUS QUO"
    
    text = fy25_text.lower()
    
    # Define a default mapping of the 20 topics to categories
    # (Based on standard NVDM latent structures for annual reports)
    topic_mapping = {
        0: "OPERATIONAL COMPLEXITY",   # operations, manufacturing, capacity
        1: "ESG DRIFT",                 # emissions, environmental, sustainability
        2: "REGULATORY SHIFT",          # compliance, legal, audit, sebi
        3: "MACRO-ECONOMIC EXPOSURE",   # export, trade, forex, global
        4: "GOVERNANCE VARIANCE",       # board, meeting, remuneration, shareholder
        5: "STRATEGIC EVOLUTION",       # strategy, digital, focus, goals
        6: "OPERATIONAL COMPLEXITY",   # supply chain, logistics, components
        7: "MACRO-ECONOMIC EXPOSURE",   # inflation, interest rates, capital
        8: "REGULATORY SHIFT",          # tax, gst, customs, policies
        9: "OPERATIONAL COMPLEXITY",   # product launch, design, r&d
        10: "ESG DRIFT",                # social responsibility, csr, community
        11: "GOVERNANCE VARIANCE",      # promoter, equity, dividend
        12: "STRATEGIC EVOLUTION",      # joint venture, merger, acquisition
        13: "MACRO-ECONOMIC EXPOSURE",  # commodity pricing, steel, raw material
        14: "OPERATIONAL COMPLEXITY",   # labor, unions, safety, factory
        15: "STRATEGIC EVOLUTION",      # customer service, digital platform
        16: "REGULATORY SHIFT",         # government schemes, subsidies
        17: "ESG DRIFT",                # carbon footprint, waste management
        18: "GOVERNANCE VARIANCE",      # internal controls, disclosure
        19: "STRATEGIC EVOLUTION"       # vision, brand, market leadership
    }
    
    # If we have a topic distribution, use the argmax of the topic vector
    if topic_distribution is not None:
        import torch
        import numpy as np
        if isinstance(topic_distribution, torch.Tensor):
            topic_distribution = topic_distribution.detach().cpu().numpy()
        # Find dominant topic index
        dominant_topic = np.argmax(topic_distribution.flatten())
        if dominant_topic in topic_mapping:
            return topic_mapping[dominant_topic]
            
    # Fallback to keyword classification if no topic distribution is passed
    if any(x in text for x in ['launch', 'new product', 'thrx', 'xuv', 'tractor', 'capacity', 'supply', 'logistics', 'strike']):
        return "OPERATIONAL COMPLEXITY"
    if any(x in text for x in ['emissions', 'esg', 'carbon', 'sdg', 'sustainability', 'solar', 'green', 'csr', 'planet']):
        return "ESG DRIFT"
    if any(x in text for x in ['regulatory', 'sebi', 'compliance', 'act', 'legal', 'law', 'audit', 'tax', 'gst']):
        return "REGULATORY SHIFT"
    if any(x in text for x in ['export', 'global', 'markets', 'usd', 'forex', 'inflation', 'slowdown', 'trade']):
        return "MACRO-ECONOMIC EXPOSURE"
    if any(x in text for x in ['remuneration', 'board', 'salary', 'governance', 'meeting', 'director', 'promoter']):
        return "GOVERNANCE VARIANCE"
        
    return "STRATEGIC EVOLUTION"