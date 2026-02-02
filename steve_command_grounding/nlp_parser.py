#!/usr/bin/env python3
import re

class NLPParser:
  """
  Simple NLP Parser for Steve's commands.
  Extracts action and target object.
  """
  def __init__(self):
    self.actions = ['find', 'go to', 'navigate to', 'bring', 'get', 'pick up', 'locate']
    
  def parse(self, command):
    command = command.lower().strip()
    # Remove trailing punctuation
    command = re.sub(r'[.!?]+$', '', command)
    
    parsed = {
      'action': 'none',
      'object': 'none',
      'raw': command
    }
    
    # Simple regex matching for common patterns
    for action in self.actions:
      if command.startswith(action):
        parsed['action'] = action
        obj_part = command[len(action):].strip()
        # Remove articles and 'me' for commands like 'bring me the mug'
        obj_part = re.sub(r'^(me\s+)?(the\s+|a\s+|an\s+)', '', obj_part)
        # Remove trailing punctuation from object
        obj_part = re.sub(r'[^a-zA-Z0-9\s]+$', '', obj_part).strip()
        parsed['object'] = obj_part
        return parsed
    
    # Pattern 2: Where is (the) [Object]
    match = re.search(r'where is (?:the |a |an )?(.+)', command)
    if match:
      parsed['action'] = 'locate'
      obj_part = match.group(1).strip()
      # Remove trailing punctuation
      obj_part = re.sub(r'[^a-zA-Z0-9\s]+$', '', obj_part).strip()
      parsed['object'] = obj_part
      return parsed
        
    return parsed
