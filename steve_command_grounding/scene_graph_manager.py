#!/usr/bin/env python3
import os
import json
import yaml
import numpy as np
from PIL import Image

class SceneGraphManager:
  """
  Manages the scene graph data from generated_graph/ and 2D map data
  """
  def __init__(self, data_root, map_yaml_path=None):
    self.data_root = data_root
    self.graph_path = os.path.join(data_root, 'graph.json')
    self.furniture_path = os.path.join(data_root, 'furniture.json')
    self.objects_dir = os.path.join(data_root, 'objects')
    
    self.furniture = {}
    self.objects = {}
    self.connections = {}
    
    # Map data
    self.map_data = None
    self.map_origin = None
    self.map_resolution = None
    
    self.load_data()
    
    # Load 2D map if provided
    if map_yaml_path:
      self.load_map(map_yaml_path)
    
    print(f"SceneGraphManager: Loaded {len(self.furniture)} furniture and {len(self.objects)} objects")

  def load_map(self, yaml_path):
    """Load 2D occupancy grid map from yaml file."""
    try:
      with open(yaml_path, 'r') as f:
        map_config = yaml.safe_load(f)
      
      self.map_resolution = map_config.get('resolution', 0.05)
      self.map_origin = map_config.get('origin', [0, 0, 0])
      
      # Load the PGM image
      map_dir = os.path.dirname(yaml_path)
      image_path = os.path.join(map_dir, map_config['image'])
      
      if os.path.exists(image_path):
        img = Image.open(image_path)
        self.map_data = np.array(img)
        print(f"SceneGraphManager: Loaded map {image_path} ({self.map_data.shape[1]}x{self.map_data.shape[0]})")
        print(f"  Origin: {self.map_origin}, Resolution: {self.map_resolution} m/pixel")
      else:
        print(f"SceneGraphManager: Map image not found: {image_path}")
    except Exception as e:
      print(f"SceneGraphManager: Failed to load map: {e}")

  def world_to_map(self, x, y):
    """Convert world coordinates to map pixel coordinates."""
    if self.map_origin is None or self.map_resolution is None:
      return None
    mx = int((x - self.map_origin[0]) / self.map_resolution)
    my = int((y - self.map_origin[1]) / self.map_resolution)
    # Flip Y because image origin is top-left
    if self.map_data is not None:
      my = self.map_data.shape[0] - my
    return (mx, my)

  def map_to_world(self, mx, my):
    """Convert map pixel coordinates to world coordinates."""
    if self.map_origin is None or self.map_resolution is None:
      return None
    # Flip Y because image origin is top-left
    if self.map_data is not None:
      my = self.map_data.shape[0] - my
    x = mx * self.map_resolution + self.map_origin[0]
    y = my * self.map_resolution + self.map_origin[1]
    return (x, y)

  def is_navigable(self, x, y, threshold=250):
    """Check if a world coordinate is navigable (free space on map)."""
    if self.map_data is None:
      return True  # Assume navigable if no map
    
    coords = self.world_to_map(x, y)
    if coords is None:
      return True
    
    mx, my = coords
    if 0 <= mx < self.map_data.shape[1] and 0 <= my < self.map_data.shape[0]:
      # In PGM: 254/255 = free, 0 = occupied, 205 = unknown
      return self.map_data[my, mx] >= threshold
    return False

  def load_data(self):
    if not os.path.exists(self.graph_path):
      return
    
    with open(self.graph_path, 'r') as f:
      graph_data = json.load(f)
      self.node_ids = graph_data.get('node_ids', [])
      self.node_labels = graph_data.get('node_labels', [])
      self.connections = graph_data.get('connections', {})
        
    if os.path.exists(self.furniture_path):
      with open(self.furniture_path, 'r') as f:
        furniture_data = json.load(f)
        self.furniture = furniture_data.get('furniture', {})

    if os.path.exists(self.objects_dir):
      for filename in os.listdir(self.objects_dir):
        if filename.endswith('.json'):
          obj_id = filename.split('.')[0]
          with open(os.path.join(self.objects_dir, filename), 'r') as f:
            self.objects[obj_id] = json.load(f)

  def find_objects(self, label):
    """
    Find all objects matching label. Returns list of matches.
    """
    label = label.lower()
    matches = []
    
    # Check all matches in map/graph first
    for i, node_label in enumerate(self.node_labels):
      if label in node_label.lower():
        node_id = str(self.node_ids[i])
        obj_data = self.find_object_by_id(node_id)
        if obj_data:
            matches.append(obj_data)
    
    # Check furniture specifically if no direct graph matches
    if not matches:
        for fid, fdata in self.furniture.items():
          if label in fdata.get('label', '').lower():
            matches.append({
              'type': 'furniture',
              'id': fid,
              'label': fdata['label'],
              'centroid': fdata['centroid'],
              'dimensions': fdata['dimensions'],
              'pose': fdata.get('pose')
            })
            
    return matches

  def find_object(self, label):
    matches = self.find_objects(label)
    return matches[0] if matches else None

  def find_object_by_id(self, obj_id):
    obj_id = str(obj_id)
    if obj_id in self.furniture:
      fdata = self.furniture[obj_id]
      return {
        'type': 'furniture',
        'id': obj_id,
        'label': fdata['label'],
        'centroid': fdata['centroid'],
        'dimensions': fdata['dimensions'],
        'pose': fdata.get('pose')
      }

    obj_info = {
      'type': 'object',
      'id': obj_id,
      'label': 'none',
      'centroid': [0, 0, 0],
      'dimensions': [0, 0, 0],
      'pose': None
    }

    if obj_id in self.objects:
      obj_info['label'] = self.objects[obj_id].get('label', 'none')
      obj_info['centroid'] = self.objects[obj_id].get('centroid', [0, 0, 0])
      obj_info['dimensions'] = self.objects[obj_id].get('dimensions', [0, 0, 0])
      obj_info['pose'] = self.objects[obj_id].get('pose')
    
    furn_id = self.connections.get(obj_id)
    if furn_id is not None:
      furn_id = str(furn_id)
      if furn_id in self.furniture:
        obj_info['furniture'] = self.furniture[furn_id]
        obj_info['furniture_id'] = furn_id
    
    return obj_info
