#!/usr/bin/env python3
import os
from steve_command_grounding.scene_graph_manager import SceneGraphManager

class Perception:
  """
  Handles robot perception and scene graph queries.
  """
  def __init__(self, node, scene_graph_path, map_yaml_path=None):
    self.node = node
    self.sg_manager = SceneGraphManager(scene_graph_path, map_yaml_path=map_yaml_path)

  def find_object(self, label):
    """Find object in the scene graph."""
    return self.sg_manager.find_object(label)

  def find_objects(self, label):
    """Find all objects matching label. Returns list of matches."""
    return self.sg_manager.find_objects(label)

  def get_available_furniture(self):
    """Return all available furniture from scene graph."""
    return self.sg_manager.furniture

  def is_navigable(self, x, y):
    """Check if position is navigable on the map."""
    return self.sg_manager.is_navigable(x, y)
