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
      "mug": ["table", "kitchen island", "desk", "chair"],
      "book": ["desk", "table", "wardrobe"],
      "remote": ["coffee table", "table", "chair"],
      "pillow": ["bed", "chair", "sofa"],
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
    
    system_prompt = """You are a robot assistant specialized in spatial reasoning.
Return ONLY valid JSON.
Format: {"locations": [{"furniture_id": "id", "furniture_name": "label"}]}
Constraint 1: Use the EXACT 'id' and 'label' from the provided furniture list.
Constraint 2: Do NOT suggest any furniture not present in the list.
Constraint 3: Suggest at most 3 locations."""
    
    user_prompt = f"Furniture: {json.dumps(furniture_list)}\n\nWhere is the '{item}' most likely to be? List top 3 locations."
    
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
      return self._parse_llm_response(clean_resp)
    except Exception as e:
      print(f"DeepSeek error: {e}")
      return self._fallback_reasoning(item, furniture_dict)

  def _parse_llm_response(self, response):
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
            proposals.append({
              'score': 10 - i,
              'furniture_id': str(loc['furniture_id']),
              'furniture_name': loc.get('furniture_name') or loc.get('furniture_label') or "unknown"
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
    likely_types = self.associations.get(item, ["table", "desk", "chair"])
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
