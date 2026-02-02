#!/usr/bin/env python3
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from steve_command_grounding.semantic_reasoner import SemanticReasoner

# Load environment variables
env_path = Path('/home/ritz/steve_ros2_ws/.env')
load_dotenv(env_path)

print(f"HUGGINGFACEHUB_API_TOKEN found: {bool(os.getenv('HUGGINGFACEHUB_API_TOKEN'))}")

# Mock furniture data
mock_furniture = {
  "0": {"label": "bed", "centroid": [-6.44, 1.94, 0.6]},
  "1": {"label": "chair", "centroid": [0.34, 4.09, 0.43]},
  "2": {"label": "table", "centroid": [6.16, 0.93, 0.66]},
  "26": {"label": "desk", "centroid": [-8.86, 1.62, 0.63]}
}

def test_reasoning():
  reasoner = SemanticReasoner()
  item = "mug"
  
  print(f"\n--- Testing Semantic Reasoner for item: '{item}' ---")
  
  if reasoner.client:
    print("DeepSeek client initialized.")
    # We can't easily see the raw response unless we modify SemanticReasoner,
    # so let's just see what it returns.
    proposals = reasoner.suggest_locations(item, mock_furniture)
    print(f"Proposals: {json.dumps(proposals, indent=2)}")
  else:
    print("DeepSeek client NOT initialized. Using fallback.")
    proposals = reasoner.suggest_locations(item, mock_furniture)
    print(f"Fallback Proposals: {json.dumps(proposals, indent=2)}")

if __name__ == "__main__":
  test_reasoning()
