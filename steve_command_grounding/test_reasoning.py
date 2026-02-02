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

# Mock furniture data from small_house
mock_furniture = {
  "0": {"label": "bed"},
  "1": {"label": "chair"},
  "2": {"label": "table"},
  "26": {"label": "desk"}
}

def test_reasoning():
  # We reuse the model ID from SemanticReasoner
  reasoner = SemanticReasoner()
  item = "mug"
  
  print(f"\n--- Testing Semantic Reasoner for item: '{item}' ---")
  
  if reasoner.client:
    print("DeepSeek client initialized.")
    # suggest_locations will call _ask_deepseek which uses chat_completion
    proposals = reasoner.suggest_locations(item, mock_furniture)
    print(f"\nFinal Proposals Output:")
    print(json.dumps(proposals, indent=2))
  else:
    print("DeepSeek client NOT initialized. Using fallback.")
    proposals = reasoner.suggest_locations(item, mock_furniture)
    print(f"Fallback Proposals: {json.dumps(proposals, indent=2)}")

if __name__ == "__main__":
  test_reasoning()
