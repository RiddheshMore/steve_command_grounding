#!/usr/bin/env python3
import json
import os
from pathlib import Path
from dotenv import load_dotenv
try:
  from huggingface_hub import InferenceClient
  HAS_HF = True
except ImportError:
  HAS_HF = False

# Load environment variables from .env in workspace root
env_path = Path('/home/ritz/steve_ros2_ws/.env')
load_dotenv(env_path)

class SemanticReasoner:
  """
  Semantic Reasoner for Steve.
  Uses DeepSeek via HuggingFace for intelligent grounding proposals.
  """
  def __init__(self, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
    self.model_id = model_id
    self.client = None
    
    # Fallback associations
    self.associations = {
      "bottle": ["table", "kitchen island", "desk"],
      "cup": ["table", "kitchen island", "coffee table"],
      "mug": ["table", "kitchen island", "desk"],
      "book": ["desk", "table", "wardrobe"],
      "remote": ["coffee table", "table"],
      "pillow": ["bed", "sofa"],
      "phone": ["table", "desk", "bed"],
      "screwdriver": ["desk", "table", "drawer"],
      "hammer": ["desk", "table", "drawer"],
      "apple": ["kitchen island", "table"]
    }

    if HAS_HF:
      api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
      if api_token:
        self.client = InferenceClient(token=api_token)
        print(f"DeepSeek client initialized with model: {model_id}")
      else:
        print("Warning: HUGGINGFACEHUB_API_TOKEN not found. Using local reasoning.")

  def suggest_locations(self, item, available_furniture):
    """
    Suggests furniture locations for an item.
    Uses LLM if available, otherwise uses association heuristic.
    """
    if self.client:
      proposals = self._ask_deepseek(item, available_furniture)
      if proposals:
        return proposals
    
    # Fallback if LLM disabled or returns nothing
    return self._fallback_reasoning(item, available_furniture)

  def _ask_deepseek(self, item, furniture_dict):
    # Format furniture for prompt
    furniture_list = []
    for fid, data in furniture_dict.items():
      furniture_list.append({"id": fid, "label": data.get("label")})
    
    system_prompt = """You are a Semantic Mapping Agent.
Your task is to identify which item the user is searching for and find the most likely furniture in the scene graph for this item.
Return ONLY valid JSON.
Format: {"locations": [{"furniture_id": "id", "furniture_name": "name", "relation": "on/in", "reasoning": "summary"}]}
Constraint 1: Use the EXACT 'id' and 'name' (label) from the provided furniture list.
Constraint 2: Do NOT suggest any furniture not present in the list.
Constraint 3: Suggest at most 3 locations, ordered by likelihood.
Constraint 4: Prioritize stable flat surfaces (tables, benches, cabinets) for small handheld objects. Avoid suggesting 'chair' or 'floor' unless specifically appropriate.
"""
    
    user_prompt = f"Furniture list (scene graph): {json.dumps(furniture_list)}\n\nQuery: Where is the '{item}' most likely to be? List top 3 locations."
    
    try:
      response = self.client.chat_completion(
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
        ],
        model=self.model_id,
        max_tokens=1024,
        temperature=0.3
      )
      clean_resp = response.choices[0].message.content
      return self._parse_llm_response(clean_resp, furniture_dict)
    except Exception as e:
      print(f"DeepSeek error: {e}")
      return self._fallback_reasoning(item, furniture_dict)

  def _parse_llm_response(self, response, furniture_dict):
    try:
      # Find JSON block, ignoring <think> reasoning if present
      clean_resp = response
      if "</think>" in str(response):
        clean_resp = str(response).split("</think>")[-1].strip()
        
      if "{" in clean_resp:
        start = clean_resp.find("{")
        end = clean_resp.rfind("}") + 1
        try:
          data = json.loads(clean_resp[start:end])
          proposals = []
          for i, loc in enumerate(data.get("locations", [])):
            fid = str(loc.get('furniture_id', ''))
            if fid not in furniture_dict:
                # LLM might have returned a label instead of ID or a hallucinated ID
                # Try to find the closest ID by label match
                label = loc.get('furniture_name', '').lower()
                for real_id, real_data in furniture_dict.items():
                    if label in real_data.get('label', '').lower():
                        fid = real_id
                        break
            
            if fid in furniture_dict:
                proposals.append({
                  'score': 10 - i,
                  'furniture_id': fid,
                  'furniture_name': furniture_dict[fid].get('label', 'unknown')
                })
          return proposals[:3]
        except json.JSONDecodeError:
          # Regex Fallback for malformed JSON
          import re
          ids = re.findall(r'"furniture_id":\s*"(\d+)"', clean_resp)
          # Greedily match either furniture_name or furniture_label
          names = re.findall(r'"furniture_(?:name|label)":\s*"([^"]+)"', clean_resp)
          
          proposals = []
          for i in range(min(len(ids), len(names), 3)):
            proposals.append({
              'score': 10 - i,
              'furniture_id': ids[i],
              'furniture_name': names[i]
            })
          if proposals:
            return proposals
    except Exception as e:
      print(f"Error parsing LLM response: {e}")
    return []

  def _fallback_reasoning(self, item, furniture_dict):
    item = item.lower()
    
    # Extended dictionary with partial match terms
    # If key is found in item string, use these locations
    partial_matches = {
        "bottle": ["table", "kitchen island", "desk"],
        "cup": ["table", "kitchen island", "coffee table"],
        "mug": ["table", "kitchen island", "desk"],
        "book": ["desk", "table", "wardrobe"],
        "remote": ["coffee table", "table"],
        "pillow": ["bed", "sofa"],
        "phone": ["table", "desk", "bed"],
        "sofa": ["living room"],
        "bed": ["bedroom"],
        
        # Tools
        "screwdriver": ["desk", "table", "drawer"],
        "hammer": ["desk", "table", "drawer"],
        
        # Food/Drinks
        "apple": ["kitchen island", "table"],
        "can": ["table", "desk", "kitchen cabinet", "cooking bench"],
        "coke": ["table", "desk", "kitchen cabinet"],
        "beer": ["table", "desk", "kitchen cabinet"],
        "soda": ["table", "desk", "kitchen cabinet"],
        "water": ["table", "desk", "kitchen cabinet"],
        "wine": ["kitchen cabinet", "table"],
        
        # General
        "plant": ["table", "floor", "window"],
        "pot": ["table", "floor", "window"]
    }
    
    likely_types = []
    
    # 1. Direct dict lookup
    if item in self.associations:
        likely_types = self.associations[item]
    else:
        # 2. Substring lookup
        for key, locs in partial_matches.items():
            if key in item:
                likely_types.extend(locs)
        
        # Deduplicate
        likely_types = list(set(likely_types))
    
    # 3. Generic fallback if no matches found
    if not likely_types:
        # Default to heavy furniture, avoid "chair" for random items
        likely_types = ["table", "desk", "kitchen cabinet", "cooking bench"]

    proposals = []
    for fid, fdata in furniture_dict.items():
      label = fdata.get('label', '').lower()
      score = 0
      for ltype in likely_types:
        if ltype in label:
          score += 1
      if score > 0:
        proposals.append({
          'score': score,
          'furniture_id': fid,
          'furniture_name': fdata['label']
        })
    proposals.sort(key=lambda x: x['score'], reverse=True)
    return proposals[:3]
