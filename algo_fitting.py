from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import pickle
from dataclasses import dataclass
from datetime import datetime
import torch
import os
import pandas as pd

from agent import SimpleDQN, PrioritizedReplayBuffer, create_agent
import time
import random
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


from config import GAMMA, LEARNING_RATE, BUFFER_SIZE, BATCH_SIZE, UPDATE_TARGET_EVERY, EPSILON_START, EPSILON_END, EPSILON_DECAY, AGENT_CONFIG


DATA_DIRECTORY = "/Users/morgandixon/Desktop/data/crypto_csv/combined"
DEFAULT_NUM_FILES = 5
DEFAULT_NUM_ROWS = 25


ENTER = 0
EXIT = 1
HOLD = 2


GLOBAL_PATHS = []
AGENT_HISTORY = []
NUM_CRYPTOS = DEFAULT_NUM_FILES
NUM_ACTIONS = 3 + DEFAULT_NUM_FILES
VERTICAL_BASE = NUM_CRYPTOS
EXIT = VERTICAL_BASE
HOLD = VERTICAL_BASE + 1
VERTICAL_START = VERTICAL_BASE + 2
TOTAL_ACTIONS = VERTICAL_START + NUM_CRYPTOS
ACTION_NAMES = {
    **{i: f"ENTER_{i}" for i in range(NUM_CRYPTOS)},
    EXIT: "EXIT",
    HOLD: "HOLD",
    **{VERTICAL_START + i: f"VERTICAL_{i}" for i in range(NUM_CRYPTOS)}
}

def process_csv_data(directory=DATA_DIRECTORY, num_files=DEFAULT_NUM_FILES, num_rows=DEFAULT_NUM_ROWS):
    """Process cryptocurrency data from CSV files and create graph structure."""
    print("\nProcessing CSV data...")
    
    try:
        files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])
        if len(files) < num_files:
            raise ValueError(f"Not enough CSV files in directory. Required: {num_files}, Found: {len(files)}")

        crypto_data = {}
        crypto_names = []
        horizontal_edges = []
        vertical_edges = []
        G = nx.Graph()
        pos = {}
        state_data = defaultdict(dict)

        for crypto_idx in range(num_files):
            file_path = os.path.join(directory, files[crypto_idx])
            crypto_name = os.path.splitext(files[crypto_idx])[0]
            crypto_names.append(crypto_name)

            try:
                df = pd.read_csv(
                    file_path,
                    skiprows=range(1, 2050 + 1),
                    nrows=num_rows + 1,
                    header=0,
                    na_values=['nan', 'NaN', 'NULL', ''],
                    float_precision='high'
                )
                
                required_columns = ['o', 'h', 'l', 'c', 'v', 'vwap', 'candle_usd_volume', 'market_cap']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise KeyError(f"Missing required columns: {missing_columns}")
                
                df['close'] = df['c']
                df['returns'] = df['c'].pct_change()
                df['volatility'] = df['returns'].rolling(window=5).std()
                df['volume_profile'] = df['v'] / df['v'].rolling(window=5).mean()
                
                df = df.iloc[1:num_rows+1].reset_index(drop=True)

                for time_idx in range(1, num_rows + 1):
                    node = (crypto_idx, time_idx)
                    idx = time_idx - 1
                    
                    try:
                        returns = float(df.iloc[idx]['returns'])
                        state = 'BR' if time_idx == 1 else ('AR' if returns >= 0 else 'BR')
                        state_data[crypto_idx][time_idx] = state
                        price_data = {
                            'close': float(df.iloc[idx]['c']),
                            'returns': returns,
                            'volatility': float(df.iloc[idx]['volatility']),
                            'volume_profile': float(df.iloc[idx]['volume_profile']),
                            'market_cap': float(df.iloc[idx]['market_cap']),
                            'vwap': float(df.iloc[idx]['vwap']),
                            'candle_usd_volume': float(df.iloc[idx]['candle_usd_volume'])
                        }
                    except (IndexError, KeyError, ValueError) as e:
                        print(f"Error processing {crypto_name} at time {time_idx}: {e}")
                        state = 'BR'
                        state_data[crypto_idx][time_idx] = state
                        price_data = crypto_data.get((crypto_idx, time_idx-1), {}).get('price_data', {})
                        if not price_data:
                            price_data = {key: 0.0 for key in ['close', 'returns', 'volatility', 'volume_profile', 
                                                             'market_cap', 'vwap', 'candle_usd_volume']}

                    G.add_node(node, state=state, crypto=crypto_name, price_data=price_data)
                    pos[node] = (time_idx, crypto_idx)
                    
                    crypto_data[node] = {
                        'state': state,
                        'crypto': crypto_name,
                        'price_data': price_data,
                        'metrics': {},
                        'horizontal_edges': [],
                        'vertical_edges': []
                    }

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        for crypto_idx in range(num_files):
            crypto_nodes = sorted(
                [node for node in crypto_data.keys() if node[0] == crypto_idx],
                key=lambda x: x[1]
            )
            
            for i in range(len(crypto_nodes)-1):
                current = crypto_nodes[i]
                next_node = crypto_nodes[i+1]
                
                if next_node[1] == current[1] + 1:
                    current_state = state_data[current[0]][current[1]]
                    next_state = state_data[next_node[0]][next_node[1]]
                    
                    if (current_state == 'AR' and next_state == 'AR') or \
                       (current_state == 'BR' and next_state == 'AR'):
                        
                        G.add_edge(current, next_node)
                        crypto_data[current]['horizontal_edges'].append(next_node)
                        crypto_data[next_node]['horizontal_edges'].append(current)
                        horizontal_edges.append((current, next_node))
        
        for time_idx in range(1, num_rows + 1):
            nodes_at_timestep = [(c, time_idx) for c in range(num_files) if (c, time_idx) in crypto_data]
            ar_nodes = [node for node in nodes_at_timestep if crypto_data[node]['state'] == 'AR']
            snippet_start_nodes = [node for node in nodes_at_timestep if 
                                crypto_data[node]['state'] == 'BR' and 
                                crypto_data[node]['horizontal_edges']]
            
            for i, node1 in enumerate(ar_nodes):
                for node2 in ar_nodes[i+1:]:
                    G.add_edge(node1, node2)
                    vertical_edges.append((node1, node2))
                    crypto_data[node1]['vertical_edges'].append(node2)
                    crypto_data[node2]['vertical_edges'].append(node1)
            
            for ar_node in ar_nodes:
                for br_node in snippet_start_nodes:
                    G.add_edge(ar_node, br_node)
                    vertical_edges.append((ar_node, br_node))
                    crypto_data[ar_node]['vertical_edges'].append(br_node)
                    crypto_data[br_node]['vertical_edges'].append(ar_node)

        print(f"Created {len(vertical_edges)} vertical edges")

        for node, data in crypto_data.items():
            data['horizontal_edges'] = [n for n in data['horizontal_edges'] if n[1] == node[1] + 1]
            
            data['vertical_edges'] = [
                n for n in data['vertical_edges']
                if n[1] == node[1]
                and n != node
            ]
        
        for node, data in crypto_data.items():
            current_horizontal = set(data['horizontal_edges'])
            existing_horizontal = set([n for n in G.neighbors(node) if (n[0] == node[0] and n[1] == node[1] + 1)])
            edges_to_remove = existing_horizontal - set(data['horizontal_edges'])
            G.remove_edges_from([(node, n) for n in edges_to_remove])

            current_vertical = set(data['vertical_edges'])
            existing_vertical = set([n for n in G.neighbors(node) if n[1] == node[1]])
            edges_to_remove = existing_vertical - set(data['vertical_edges'])
            G.remove_edges_from([(node, n) for n in edges_to_remove])

        color_map = {'AR': '#2ecc71', 'BR': '#e74c3c'}
        node_colors = [color_map.get(data['state'], '#95a5a6') for node, data in crypto_data.items()]

        structured_graph_data = StructuredGraphData(G, pos, node_colors, horizontal_edges, vertical_edges)
        structured_data = structured_graph_data.get_node_data()

        return G, pos, node_colors, horizontal_edges, vertical_edges, state_data, crypto_names, structured_data

    def build_path_adherence_string(new_node, old_idx, new_idx, global_path):
        """Helper to make the 'path adherence' part of the printout more readable."""
        if new_node not in global_path:
            return '-3.0 (Off path entirely)'
        else:
            if old_idx != -1 and new_idx == old_idx + 1:
                return '+2.0 (Correct forward step)'
            else:
                return '-2.0 (Wrong order in path)'

    class GraphNode:
        def __init__(self, coords: Tuple[int, int], state: str, crypto: str, price_data: Dict = None, metrics: Dict = None):
            self.coords = coords
            self.state = state
            self.crypto = crypto
            self.price_data = price_data or {}
            self.metrics = metrics or {}
            self.horizontal_edges: List[Tuple[int, int]] = []
            self.vertical_edges: List[Tuple[int, int]] = []

        def to_dict(self) -> Dict:
            return {
                'coords': self.coords,
                'state': self.state,
                'crypto': self.crypto,
                'price_data': self.price_data,
                'metrics': self.metrics,
                'horizontal_edges': self.horizontal_edges,
                'vertical_edges': self.vertical_edges
            }
    
    class StructuredGraphData:
        def __init__(self, G: nx.Graph, pos: Dict, node_colors: List, 
                     horizontal_edges: List, vertical_edges: List):
            self.nodes: Dict[Tuple[int, int], GraphNode] = {}
            self.pos = pos
            self._build_node_data(G)
            self._add_edges(horizontal_edges, vertical_edges)
    
        def _build_node_data(self, G: nx.Graph) -> None:
            for node, data in G.nodes(data=True):
                price_data = data.get('price_data', {})
                metrics = data.get('metrics', {})
                
                self.nodes[node] = GraphNode(
                    coords=node,
                    state=data['state'],
                    crypto=data['crypto'],
                    price_data=price_data,
                    metrics=metrics
                )

        def _add_edges(self, horizontal_edges: List, vertical_edges: List) -> None:
            for edge in horizontal_edges:
                if edge[1][1] > edge[0][1]:
                    self.nodes[edge[0]].horizontal_edges.append(edge[1])
    
            for edge in vertical_edges:
                self.nodes[edge[0]].vertical_edges.append(edge[1])
                self.nodes[edge[1]].vertical_edges.append(edge[0])
        
        def get_node_data(self) -> Dict[Tuple[int, int], Dict]:
            """Convert all nodes to dictionary format."""
            return {coords: node.to_dict() for coords, node in self.nodes.items()}
        
        def get_node_colors(self):
            """Returns list of colors for each node based on state."""
            color_map = {'AR': '#2ecc71', 'BR': '#e74c3c'}
            return [color_map.get(self.nodes[node].state, '#95a5a6') for node in self.nodes]
    
    class PathBuilder:
        def __init__(self, structured_data: Dict[Tuple[int, int], Dict]):
            self.structured_data = structured_data
            self.visited = set()
            self.max_right = -1

        def build_paths(self):
            """Build optimal paths and store in GLOBAL_PATHS"""
            print("\nBuilding optimal path...")
            start_node = (0, 2)
            
            if start_node not in self.structured_data:
                print(f"Start node {start_node} not found!")
                return
            
            final_path = self._build_path(start_node, [])
            if final_path:
                GLOBAL_PATHS.clear()
                GLOBAL_PATHS.append(final_path)
                print(f"Optimal path found: {len(final_path)} nodes")
            else:
                print("No valid path could be constructed")

        def _build_path(self, current_node, current_path):
            """Recursive path builder with horizontal priority and vertical fallback"""
            if current_node in self.visited:
                return current_path
                
            self.visited.add(current_node)
            updated_path = current_path + [current_node]
            
            current_time = current_node[1]
            if current_time > self.max_right:
                self.max_right = current_time

            horizontal_next = self._get_valid_horizontal(current_node)
            if horizontal_next:
                return self._build_path(horizontal_next, updated_path)
                
            vertical_options = self._get_promising_vertical(current_node)
            
            best_path = updated_path
            for v_node in vertical_options:
                if v_node not in self.visited and v_node[1] >= self.max_right:
                    sub_path = self._build_path(v_node, updated_path)
                    
                    if self._get_path_end(sub_path)[1] > self._get_path_end(best_path)[1]:
                        best_path = sub_path
                        
            return best_path

        def _get_valid_horizontal(self, node):
            """Get next valid horizontal node with earliest timestamp"""
            candidates = [
                n for n in self.structured_data[node]['horizontal_edges']
                if self._is_valid_transition(node, n) 
                and n[1] == node[1] + 1
            ]
            return min(candidates, key=lambda x: x[1]) if candidates else None

        def _get_promising_vertical(self, node):
            """Get vertical nodes sorted by their potential to progress right"""
            current_time = node[1]
            vertical_nodes = [
                n for n in self.structured_data[node]['vertical_edges']
                if n[1] == current_time
                and self.structured_data[n]['state'] == 'AR'
            ]
            
            return sorted(
                vertical_nodes,
                key=lambda x: len(self.structured_data[x]['horizontal_edges']),
                reverse=True
            )

        def _is_valid_transition(self, current, next_node):
            """Ensure transition follows state rules"""
            current_state = self.structured_data[current]['state']
            next_state = self.structured_data[next_node]['state']
            return (current_state == 'AR' and next_state == 'AR') or \
                   (current_state == 'BR' and next_state == 'AR')

        def _get_path_end(self, path):
            """Get final node's position in a path"""
            return path[-1] if path else (0, 0)

        def print_paths(self):
            """Visualize all optimized paths with navigation details"""
            print("\n=== Optimized Paths ===")
            if GLOBAL_PATHS:
                for idx, path in enumerate(GLOBAL_PATHS, 1):
                    print(f"Path {idx}:")
                    for i, node in enumerate(path):
                        if i == 0:
                            print(f"  Start at {node} ({self.structured_data[node]['crypto']})")
                        else:
                            print(f"  → {node} ({self.structured_data[node]['crypto']})")
            else:
                print("No paths to display")

    class GraphPlotter:
        """
        Handles the visualization of the cryptocurrency graph, including node and edge plotting,
        highlighting paths, and integrating with the reinforcement learning agent.
        """
        def __init__(self, structured_data, pos, node_colors, state_data, path_builder):
            self.base_structured_data = self._filter_data(structured_data)
            self.base_pos = pos
            self.base_node_colors = [mcolors.to_rgba(c) for c in node_colors]
            self.state_data = state_data
            
            self.fig = plt.figure(figsize=(12, 8))
            gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1])
            self.ax_graph = self.fig.add_subplot(gs[0])
            self.ax_metrics = self.fig.add_subplot(gs[1])
            
            plt.style.use('default')
            self.ax_graph.set_facecolor('#f0f0f0')
            self.ax_metrics.set_facecolor('#ffffff')
            self.fig.patch.set_facecolor('white')
            
            self.node_collection = None
            self.edge_collections = {}
            self.current_indicators = []
            
            self.agent = None
            self.env = None
            self.current_agent_path = []
            self.agent_animation = None
            self.episode_rewards = []
            self.episode_accuracies = []
            self.current_episode_reward = 0.0
            self.loss_history = []
            self.valid_paths = GLOBAL_PATHS
            
            self.paused = False
            self.pause_text = None
            self.training_enabled = True
            self.step_queued = False
            self._redraw_queued = False
            self._last_redraw = time.time()
            
            self._generate_all_paths()
            self._draw_base_graph()
            self._initialize_metrics_plot()
            
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        def _get_horizontal_edges(self):
            """Extract horizontal edges from structured data"""
            edges = []
            for node, data in self.base_structured_data.items():
                edges.extend([(node, target) 
                            for target in data.get('horizontal_edges', []) 
                            if target in self.base_structured_data])
            return edges

        def _get_vertical_edges(self):
            """Extract vertical edges from structured data"""
            edges = []
            for node, data in self.base_structured_data.items():
                edges.extend([(node, target) 
                            for target in data.get('vertical_edges', []) 
                            if target in self.base_structured_data])
            return edges

        def reset_graph_display(self):
            """Completely clear and redraw the base graph"""
            for coll in self.edge_collections.values():
                coll.remove()
            if self.node_collection:
                self.node_collection.remove()
            for indicator in self.current_indicators:
                indicator.remove()
            self.current_indicators.clear()
            
            self._draw_base_graph()

        def update_agent_display(self, current_node, history):
            """Update display with current agent state"""
            self.reset_graph_display()
            
            node_idx = list(self.base_structured_data.keys()).index(current_node)
            colours = self.node_collection.get_facecolors()
            colours[node_idx] = mcolors.to_rgba('black')
            
            history_color = mcolors.to_rgba('#8A2BE2')
            for node in history[-3:]:
                if node in self.base_structured_data:
                    idx = list(self.base_structured_data.keys()).index(node)
                    colours[idx] = history_color
                    
            self.node_collection.set_facecolors(colours)
            
            node_pos = self.base_pos[current_node]
            indicator = Rectangle(
                (node_pos[0]-0.1, node_pos[1]-0.1),
                0.2, 0.2,
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            self.ax_graph.add_patch(indicator)
            self.current_indicators.append(indicator)
            
            self.fig.canvas.draw_idle()

        def _draw_base_graph(self):
            """Draw static graph elements once"""
            G = nx.Graph()
            nodes = list(self.base_structured_data.keys())
            G.add_nodes_from(nodes)
            
            self.node_collection = nx.draw_networkx_nodes(
                G, self.base_pos, nodelist=nodes,
                node_color=self.base_node_colors,
                node_size=150, ax=self.ax_graph
            )
            
            edge_types = [
                ('horizontal_edges', self._get_horizontal_edges(), 'lightblue', 'solid'),
                ('vertical_edges', self._get_vertical_edges(), 'orange', 'dashed')
            ]
            
            for edge_type, edges, color, style in edge_types:
                segments = [(self.base_pos[u], self.base_pos[v]) for u, v in edges]
                ecoll = LineCollection(segments, colors=color, linewidths=1,
                                     linestyles=style, zorder=1, alpha=0.6)
                self.ax_graph.add_collection(ecoll)
                self.edge_collections[edge_type] = ecoll

        def _filter_data(self, data):
            """Filter data based on DEFAULT_NUM_FILES and DEFAULT_NUM_ROWS dimensions."""
            max_crypto = DEFAULT_NUM_FILES
            max_time = DEFAULT_NUM_ROWS

            filtered_data = {}
            for k, v in data.items():
                crypto_idx, time_idx = k
                if crypto_idx < max_crypto and time_idx <= max_time:
                    fv = v.copy()
                    
                    fv['horizontal_edges'] = [
                        e for e in v['horizontal_edges']
                        if e[0] < max_crypto and e[1] <= max_time
                    ]

                    fv['vertical_edges'] = [
                        e for e in v['vertical_edges']
                        if e[0] < max_crypto and e[1] <= max_time
                    ]
                    filtered_data[k] = fv

            print(f"Filtered data dimensions: {max_crypto} cryptocurrencies x {max_time} time steps")
            print(f"Total nodes after filtering: {len(filtered_data)}")
            return filtered_data

        def _generate_all_paths(self):
            """Use GLOBAL_PATHS directly"""
            if GLOBAL_PATHS:
                self.paths = GLOBAL_PATHS
            else:
                print("No paths to generate - ensure path building completed")

        def plot_graph(self):
            """Draw the complete graph visualization"""
            self.ax = self.ax_graph
            self._plot_single_graph()
            self._highlight_all_paths()
            self.ax_graph.set_title("Cryptocurrency Graph Analysis")
            self._initialize_metrics_plot()
            plt.tight_layout()

        def _highlight_edge(self, node1, node2, color='purple'):
            """Optimized edge highlighting with numpy masking"""
            def is_edge_match(segment, n1, n2):
                return (segment[0] == self.base_pos[n1] and segment[1] == self.base_pos[n2]) or \
                       (segment[0] == self.base_pos[n2] and segment[1] == self.base_pos[n1])
            
            highlight_color = mcolors.to_rgba(color)
            
            for edge_type in ['horizontal_edges', 'vertical_edges']:
                ecoll = self.edge_collections[edge_type]
                segments = ecoll.get_segments()
                
                if len(segments) == 0:
                    continue
                    
                edge_mask = np.array([is_edge_match(seg, node1, node2) for seg in segments])
                
                new_colors = np.array(ecoll.get_colors())
                new_colors[edge_mask] = highlight_color
                ecoll.set_colors(new_colors)
                
                new_widths = np.array(ecoll.get_linewidths())
                new_widths[edge_mask] = 3
                ecoll.set_linewidths(new_widths)
            
            self.schedule_redraw()

        def _reset_highlighting(self):
            """Reset all node and edge highlights to their original states"""
            color_map = {'AR': '#2ecc71', 'BR': '#e74c3c'}
            nodes = list(self.base_structured_data.keys())
            node_colors = [mcolors.to_rgba(color_map[self.base_structured_data[node]['state']]) 
                        for node in nodes]
            
            self.node_collection.set_facecolors(node_colors)

            edge_original_colors = {
                'horizontal_edges': 'lightblue',
                'vertical_edges': 'orange'
            }
            for edge_type, ecoll in self.edge_collections.items():
                original_color = edge_original_colors.get(edge_type, 'gray')
                ecoll.set_colors([mcolors.to_rgba(original_color)] * len(ecoll.get_colors()))
                ecoll.set_linewidths([1] * len(ecoll.get_linewidths()))
            
            if hasattr(self.env, 'current_node') and self.env.current_node in self.base_structured_data:
                current_node = self.env.current_node
                if current_node in self.base_structured_data:
                    idx = list(self.base_structured_data.keys()).index(current_node)
                    colors = self.node_collection.get_facecolors()
                    colors[idx] = mcolors.to_rgba('black')
                    self.node_collection.set_facecolors(colors)

        def _highlight_node_connections(self, node, is_click=False):
            """Highlight connections of a specific node."""
            self._reset_highlighting()

            if is_click:
                self._highlight_paths_from_node(node)

            self.fig.canvas.draw_idle()

        def _highlight_path(self, path: list, selected_node: tuple, color: str = 'purple'):
            """Highlight a path on the main graph"""
            if not path:
                return
                
            nodes = list(self.base_structured_data.keys())
            color_map = {'AR': '#2ecc71', 'BR': '#e74c3c'}
            
            base_colors = np.array([mcolors.to_rgba(color_map[self.base_structured_data[node]['state']]) 
                                for node in nodes])
            
            path_nodes = np.array([node in path for node in nodes])
            highlight_color = np.array(mcolors.to_rgba(color))
            
            current_colors = base_colors.copy()
            current_colors[path_nodes] = highlight_color
            
            if selected_node in nodes:
                idx = nodes.index(selected_node)
                current_colors[idx] = mcolors.to_rgba('black')
            
            self.node_collection.set_facecolors(current_colors)
            self.schedule_redraw()

        def schedule_redraw(self):
            if not self._redraw_queued:
                self._redraw_queued = True
                self.fig.canvas.draw_idle()

        def _optimized_draw(self):
            if time.time() - self._last_redraw > 0.033:
                self.fig.canvas.draw()
                self._last_redraw = time.time()
                self._redraw_queued = False

        def _coord_to_node(self, coord):
            """Convert matplotlib coordinates to node tuple"""
            return (int(round(coord[1])), int(round(coord[0])))

        def _highlight_agent_path(self):
            """Highlight the agent's current path on the graph"""
            if not self.env:
                return
            current_node = self.env.current_node
            node_to_idx = {node: idx for idx, node in enumerate(self.base_structured_data.keys())}

            current_colors = self.node_collection.get_facecolors().copy()

            color_map = {'AR': '#2ecc71', 'BR': '#e74c3c'}
            for idx, node in enumerate(self.base_structured_data.keys()):
                current_colors[idx] = mcolors.to_rgba(color_map[self.base_structured_data[node]['state']])

            if current_node in node_to_idx:
                idx = node_to_idx[current_node]
                current_colors[idx] = mcolors.to_rgba('black')

            traversal_color = '#8A2BE2'
            for node in AGENT_HISTORY:
                if node in node_to_idx:
                    idx = node_to_idx[node]
                    current_colors[idx] = mcolors.to_rgba(traversal_color)

            self.node_collection.set_facecolors(current_colors)

            self._update_current_timestep_indicator(current_node)

            self.fig.canvas.draw_idle()

        def _update_current_timestep_indicator(self, node):
            """Update indicator on main graph"""
            if hasattr(self, 'current_indicator'):
                self.current_indicator.remove()
            
            node_pos = self.base_pos.get(node)
            if node_pos:
                rect_size = 0.07
                self.current_indicator = Rectangle(
                    (node_pos[0]-rect_size, node_pos[1]-rect_size),
                    2*rect_size,
                    2*rect_size,
                    linewidth=3,
                    edgecolor='yellow',
                    facecolor='none',
                    zorder=5
                )
                self.ax_graph.add_patch(self.current_indicator)

        def _update_edge_collections(self):
            """Update all edge collections."""
            edge_original_colors = {
                'horizontal_edges': 'lightblue',
                'vertical_edges': 'orange'
            }
            for edge_type, ecoll in self.edge_collections.items():
                original_color = edge_original_colors.get(edge_type, 'gray')
                ecoll.set_color([mcolors.to_rgba(original_color)] * len(ecoll.get_colors()))
                ecoll.set_linewidths([1] * len(ecoll.get_linewidths()))
            
            self.fig.canvas.draw_idle()

        def _highlight_all_paths(self):
            """Highlight all paths on main graph"""
            colors = plt.cm.rainbow(np.linspace(0, 1, len(GLOBAL_PATHS)))
            for path_idx, path in enumerate(GLOBAL_PATHS):
                self._highlight_path(path, path[0], color=colors[path_idx])
            self._update_edge_collections()

        def _initialize_metrics_plot(self):
            """Initialize metrics plot with empty datasets"""
            self.ax_metrics.set_title("Training Metrics")
            self.ax_metrics.set_xlabel("Episode")
            self.ax_metrics.set_ylabel("Value")
            self.metrics_lines = {
                'reward': self.ax_metrics.plot([], [], label='Reward')[0],
                'accuracy': self.ax_metrics.plot([], [], label='Accuracy')[0]
            }
            self.ax_metrics.legend()
            self.ax_metrics.grid(True)

        def update_metrics(self, episode_rewards, episode_accuracies):
            """Update metrics plot with new data"""
            x = range(len(episode_rewards))
            self.metrics_lines['reward'].set_data(x, episode_rewards)
            self.metrics_lines['accuracy'].set_data(x, episode_accuracies)
            
            self.ax_metrics.relim()
            self.ax_metrics.autoscale_view()
            self.fig.canvas.draw_idle()

        def _plot_single_graph(self):
            """Draw the graph on the main axis"""
            self.ax_graph.clear()
            self.ax_graph.set_axis_off()
            
            nodes = list(self.base_structured_data.keys())
            color_map = {'AR': '#2ecc71', 'BR': '#e74c3c'}
            node_colors = [color_map[self.base_structured_data[node]['state']] 
                        for node in nodes]
            
            G = nx.Graph()
            G.add_nodes_from(nodes)
            self.node_collection = nx.draw_networkx_nodes(
                G, self.base_pos, nodelist=nodes,
                node_color=node_colors, node_size=150,
                ax=self.ax_graph
            )
            
            edge_types = [
                ('horizontal_edges', self._get_horizontal_edges(), 'lightblue', 'solid'),
                ('vertical_edges', self._get_vertical_edges(), 'orange', 'dashed')
            ]
            
            for edge_type, edges, color, style in edge_types:
                segments = [(self.base_pos[u], self.base_pos[v]) for u, v in edges]
                ecoll = LineCollection(
                    segments, colors=color, linewidths=1,
                    linestyles=style, zorder=1, alpha=0.6
                )
                self.ax_graph.add_collection(ecoll)
                self.edge_collections[edge_type] = ecoll
            
            labels = {
                n: f"{self.base_structured_data[n]['crypto']}\n{n[1]}"
                for n in nodes
            }
            nx.draw_networkx_labels(
                G, self.base_pos, labels, 
                font_size=6, ax=self.ax_graph
            )

        def _highlight_paths_from_node(self, node):
            """Highlight all paths originating from the given node"""
            filtered_paths = []
            for path in GLOBAL_PATHS:
                if node in path:
                    idx = path.index(node)
                    remaining = path[idx + 1:]
                    if remaining:
                        filtered_paths.append(remaining)

            for path in filtered_paths:
                self._highlight_path(path, node)

            self._update_edge_collections()

        def setup_agent(self, agent, env):
            # Calculate correct state dimension
            state_dim = (
                env.num_nodes +
                1 +
                1 +
                1
            )
            
            self.agent = create_agent(env)

            self.env = env
            self.replay_buffer = PrioritizedReplayBuffer(capacity=BUFFER_SIZE)
            self.optimizer = optim.Adam(self.agent.parameters(), lr=LEARNING_RATE)

            self.target_agent = create_agent(env)

            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_agent.eval()

            self.epsilon = EPSILON_START
            self.epsilon_min = EPSILON_END
            self.epsilon_decay = EPSILON_DECAY
            self.batch_size = BATCH_SIZE
            self.gamma = GAMMA
            self.target_update_freq = UPDATE_TARGET_EVERY
            self.step_count = 0

            self.env.reset()
            self.current_agent_path = [self.env.current_node]

            self.episode_counter = 0
            self.correct_moves = 0
            self.total_moves = 0
            self.current_episode_reward = 0.0

            self.agent_animation = self.fig.canvas.new_timer(interval=200)
            self.agent_animation.add_callback(self._check_and_step_agent)
            self.agent_animation.start()
            self.epsilon_reset_steps = 100
            self.base_epsilon = EPSILON_START
            self.boosted_epsilon = EPSILON_START

        def _get_effective_epsilon(self):
            return self.base_epsilon  
        
        def _step_agent(self):
            if self.env.done:
                self._handle_episode_done()
                return

            state = self.env._get_state()
            current_node = self.env.current_node
            failure_count = self.env.failure_counts.get(current_node, 0)
            effective_epsilon = self._get_effective_epsilon()
            current_in_position = self.env.in_position
            valid_actions = self.env.get_valid_actions()
            invalid_actions = [a for a in range(NUM_ACTIONS) if a not in valid_actions]
            
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = self.agent(state.unsqueeze(0)).clone()
                    q_values[:, invalid_actions] = -float('inf')
                    action = int(torch.argmax(q_values, dim=1)[0])

                    if failure_count > 0:
                        noise = torch.randn_like(q_values) * 0.05 * failure_count
                        q_values += noise

                    action = int(torch.argmax(q_values, dim=1)[0])

            previous_node = self.env.current_node
            was_in_position = self.env.in_position

            next_state, reward, done = self.env.step(action)

            if done:
                self.replay_buffer.push(state, action, reward, next_state, done)
                self._handle_episode_done()
                return

            self.current_agent_path = self.env.current_path
            self.current_episode_reward += reward
            self.total_moves += 1

            current_segment = self.env.get_current_path_segment()
            global_path = GLOBAL_PATHS[0] if GLOBAL_PATHS else []
            correct_actions = self.env.get_correct_actions(previous_node, self.env.current_node, was_in_position)
            print(f"""
                Agent Node: {self.env.current_node}
                Last 10 Positions: {AGENT_HISTORY[-10:]}
                Predicted Action: {ACTION_NAMES[action]}
                Current Path Segment: {current_segment}
                Correct Action(s): {[ACTION_NAMES[a] for a in correct_actions]}
                Full Global Path: {global_path}
                Reward: {reward}
                _______________
            """.strip())

            if self._is_correct_transition(previous_node, self.env.current_node):
                self.correct_moves += 1
                self.env.failure_counts[self.env.current_node] = 0
            else:
                self.env.failure_counts[self.env.current_node] += 1

            self.replay_buffer.push(state, action, reward, next_state, done)

            if self.training_enabled and len(self.replay_buffer) >= self.batch_size:
                sample = self.replay_buffer.sample(self.batch_size)
                if sample is not None:
                    indices, batch, weights = sample
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states_t = torch.stack(states)
                    actions_t = torch.tensor(actions, dtype=torch.long)
                    rewards_t = torch.tensor(rewards, dtype=torch.float32)
                    next_states_t = torch.stack(next_states)
                    dones_t = torch.tensor(dones, dtype=torch.float32)
                    weights_t = torch.tensor(weights, dtype=torch.float32)

                    current_q = self.agent(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_actions = self.agent(next_states_t).argmax(dim=1)
                        next_q = self.target_agent(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

                    td_errors = F.smooth_l1_loss(current_q, target_q, reduction='none')
                    self.replay_buffer.update_priorities(indices, td_errors + 1e-5)

                    if self.env.failure_counts.get(current_node, 0) > 2:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = LEARNING_RATE * 2
                    else:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = LEARNING_RATE

                    loss = (weights_t * td_errors).mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                    self.optimizer.step()

                    self.loss_history.append(loss.item())
                    self.step_count += 1

                    if self.step_count % self.target_update_freq == 0:
                        self.target_agent.load_state_dict(self.agent.state_dict())

            self.base_epsilon = max(self.base_epsilon * EPSILON_DECAY, EPSILON_END)
            if self.step_count % self.epsilon_reset_steps == 0:
                self.boosted_epsilon = self.base_epsilon

            self._reset_highlighting()
            self._highlight_agent_path()
            self.fig.canvas.draw_idle()
            self.update_agent_display(self.env.current_node, AGENT_HISTORY)
            self.update_metrics(self.episode_rewards, self.episode_accuracies)

        def _handle_episode_done(self):
            """
            A small helper to handle end-of-episode housekeeping:
            - Clear agent history
            - Update metrics
            - Reset environment
            """
            AGENT_HISTORY.clear()
            ep_accuracy = self.correct_moves / self.total_moves if self.total_moves else 0.0
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_accuracies.append(ep_accuracy)
            self.update_metrics(self.episode_rewards, self.episode_accuracies)

            self.ax = self.ax_metrics
            self._reset_highlighting()
            self._plot_single_graph()
            self.ax_metrics.set_title(f"Agent's View (Episode {self.episode_counter})")

            self.env.reset()
            self.current_agent_path = [self.env.current_node]
            self.fig.canvas.draw_idle()

            self.episode_counter += 1
            self.correct_moves = 0
            self.total_moves = 0
            self.current_episode_reward = 0.0

        def _update_metrics_plot(self):
            """Update metrics plot without full redraw"""
            x = range(len(self.episode_rewards))
            self.metrics_lines['reward'].set_data(x, self.episode_rewards)
            self.metrics_lines['accuracy'].set_data(x, self.episode_accuracies)
            
            self.ax_metrics.relim()
            self.ax_metrics.autoscale_view()
            
            self.ax_metrics.draw_artist(self.metrics_lines['reward'])
            self.ax_metrics.draw_artist(self.metrics_lines['accuracy'])
            self.fig.canvas.blit(self.ax_metrics.bbox)

        def _is_correct_transition(self, node1, node2):
            """Check if transition exists in any global path segment"""
            current_segment = self.env.get_current_path_segment()
            try:
                idx = current_segment.index(node1)
                return current_segment[idx+1] == node2
            except (ValueError, IndexError):
                return False

        def _check_and_step_agent(self):
            """New method to check pause state before stepping"""
            if not self.paused or self.step_queued:
                self.step_queued = False
                self._step_agent()

        def on_key_press(self, event):
            """Enhanced key press handler with multiple commands"""
            if event.key == 'p':
                self._toggle_pause()
            elif event.key == 'n' and self.paused:
                self.step_queued = True
                self._check_and_step_agent()
            elif event.key == 't' and self.paused:
                self.training_enabled = not self.training_enabled
                status = "enabled" if self.training_enabled else "disabled"
                print(f"\nTraining {status}")
                self._update_pause_message()

        def _toggle_pause(self):
            """Toggle pause state with enhanced messaging"""
            self.paused = not self.paused
            if self.paused:
                print("\nExecution Paused")
                print("Controls:")
                print("- Press 'p' again to resume")
                print("- Press 'n' to step forward once")
                print("- Press 't' to toggle training on/off")
                self._display_pause_message()
            else:
                print("\nExecution Resumed")
                self._remove_pause_message()

        def _display_pause_message(self):
            """Enhanced pause message display"""
            if hasattr(self, 'pause_text') and self.pause_text:
                self.pause_text.remove()

            message = "PAUSED\n"
            message += "p: Resume | n: Step | t: Toggle Training"
            if not self.training_enabled:
                message += "\n(Training Disabled)"
                
            self.pause_text = self.ax.text(
                0.5, 0.5, message,
                transform=self.ax.transAxes,
                fontsize=16, color='red',
                ha='center', va='center',
                bbox=dict(facecolor='yellow', alpha=0.5, boxstyle='round,pad=1'),
                zorder=1000
            )
            self.fig.canvas.draw_idle()

        def _update_pause_message(self):
            """Update pause message if state changes while paused"""
            if self.paused:
                self._display_pause_message()

        def _remove_pause_message(self):
            """Remove the pause message from the plot"""
            if hasattr(self, 'pause_text') and self.pause_text:
                self.pause_text.remove()
                self.pause_text = None
                self.fig.canvas.draw_idle()
        
    class PathFollowingEnvironment:
        """
        Enhanced environment with refined reward structure:
        1) Tracks (crypto_idx, time_idx).
        2) Has 'valid_paths' to define which edges are correct.
        3) Rewards transitions within the same path highly.
        4) Rewards transitions across different paths moderately.
        5) Punishes transitions that deviate from all paths.
        6) Rewards for landing on green/positive nodes.
        7) Provides node-specific features to the agent.
        8) Lightly punishes landing on non-path connected red nodes.
        9) Enhanced state representation with additional features.
        """
        def __init__(self, structured_data, max_time=25):
            self.structured_data = structured_data
            self.max_time = max_time

            all_nodes = list(structured_data.keys())
            self.node2id = {node: i for i, node in enumerate(all_nodes)}
            self.id2node = {i: node for i, node in enumerate(all_nodes)}
            self.num_nodes = len(all_nodes)

            self.path_indices = self._index_paths(GLOBAL_PATHS)
            self.node_to_paths = self._map_nodes_to_paths(GLOBAL_PATHS)

            self.current_node = None
            self.current_path = []
            self.active_path_id = None
            self.in_position = False
            self.done = False
            self._compute_normalization_factors()

            self.node_embeddings = torch.eye(self.num_nodes)
            self.failure_counts = defaultdict(int)
            self.last_failed_node = None

        @property
        def state_dim(self):
            return self.num_nodes + 3 
        
        def get_current_path_segment(self):
            """Get full path segment from current position"""
            for path in GLOBAL_PATHS:
                if self.current_node in path:
                    idx = path.index(self.current_node)
                    return path[idx:]
            return []
        
        def _index_paths(self, paths):
            """Assign a unique index to each path."""
            valid_indices = {}
            for idx, path in enumerate(paths):
                if path and path[0][1] == 1:
                    valid_indices[idx] = path
            return valid_indices

        def _map_nodes_to_paths(self, paths):
            """Map each node to the list of paths it belongs to."""
            node_map = defaultdict(list)
            for idx, path in enumerate(paths):
                if path and path[0][1] == 1:
                    for node in path:
                        node_map[node].append(idx)
            return node_map

        def _compute_normalization_factors(self):
            """Compute normalization factors based on the dataset."""
            self.max_close_price = 0.0
            self.max_returns = 0.0
            self.max_volatility = 0.0
            self.max_volume = 0.0
            self.max_market_cap = 0.0
            self.max_vwap = 0.0
            self.max_candle_usd_volume = 0.0

            for node_data in self.structured_data.values():
                price_data = node_data.get('price_data', {})
                self.max_close_price = max(self.max_close_price, price_data.get('close', 0.0))
                self.max_returns = max(self.max_returns, price_data.get('returns', 0.0))
                self.max_volatility = max(self.max_volatility, price_data.get('volatility', 0.0))
                self.max_volume = max(self.max_volume, price_data.get('volume_profile', 0.0))
                self.max_market_cap = max(self.max_market_cap, price_data.get('market_cap', 0.0))
                self.max_vwap = max(self.max_vwap, price_data.get('vwap', 0.0))
                self.max_candle_usd_volume = max(self.max_candle_usd_volume, price_data.get('candle_usd_volume', 0.0))

        def reset(self):
            """Initialize with first node of a random valid path"""
            global AGENT_HISTORY
            AGENT_HISTORY.clear()
            
            all_nodes = list(self.structured_data.keys())
            if not GLOBAL_PATHS:
                raise ValueError("No global paths available")
            
            self.active_path = GLOBAL_PATHS[0]  
            self.current_node = self.active_path[0]
            self.current_path = [self.current_node]
            self.in_position = False
            self.done = False
            
            print(f"Reset to path: {self.active_path}")
            return self._get_state()

        def _calculate_next_position(self, agent_action, old_node, was_in_position):
            old_crypto, old_time = old_node
            new_time = old_time + 1

            if agent_action < NUM_CRYPTOS:  
                new_crypto = agent_action

            elif agent_action == EXIT:
                new_crypto = old_crypto

            elif agent_action == HOLD:
                new_crypto = old_crypto

            elif agent_action >= VERTICAL_START:
                new_time = old_time
                new_crypto = agent_action - VERTICAL_START

            else:
                new_crypto = old_crypto

            return (new_crypto, new_time)

        def step(self, agent_action: int):
            if self.done:
                return self._get_state(), 0.0, True

            global_path = GLOBAL_PATHS[0]
            old_node = self.current_node
            was_in_position = self.in_position

            if not was_in_position:
                if not (0 <= agent_action < NUM_CRYPTOS or agent_action == HOLD):
                    return self._get_state(), -3.0, False
            else:
                if agent_action < VERTICAL_START and agent_action not in [HOLD, EXIT]:
                    return self._get_state(), -3.0, False

            new_node = self._calculate_next_position(agent_action, old_node, was_in_position)

            if 0 <= agent_action < NUM_CRYPTOS:
                if was_in_position:
                    return self._get_state(), -5.0, False

                old_crypto = old_node[0]
                new_crypto = new_node[0]
                if new_crypto == old_crypto:
                    return self._get_state(), -4.0, False

            if new_node not in self.structured_data:
                return self._get_state(), -5.0, True

            self.current_node = new_node
            AGENT_HISTORY.append(new_node)
            if len(AGENT_HISTORY) > 10:
                AGENT_HISTORY.pop(0)

            next_state = self._get_state()
            reward = 0.0

            correct_actions = self.get_correct_actions(old_node, new_node, was_in_position)

            if agent_action in correct_actions:
                reward += 3.0

                if new_node in global_path:
                    old_idx = global_path.index(old_node) if old_node in global_path else -1
                    new_idx = global_path.index(new_node)

                    if old_idx != -1 and new_idx == old_idx + 1:
                        reward += 2.0
                    else:
                        reward -= 2.0
                else:
                    reward -= 3.0
            else:
                reward -= 1.0

            if agent_action >= VERTICAL_BASE and not was_in_position:
                self.in_position = True
            elif (0 <= agent_action < NUM_CRYPTOS) and not was_in_position:
                self.in_position = True
            elif agent_action == EXIT:
                self.in_position = False

            if new_node == global_path[-1] and not self.in_position:
                reward += 10.0
                self.done = True

            if new_node[1] >= self.max_time:
                self.done = True

            old_idx = global_path.index(old_node) if old_node in global_path else -1
            new_idx = global_path.index(new_node) if new_node in global_path else -1

            print(f"""
            Reward Breakdown:
            Action correctness: {'+3.0' if agent_action in correct_actions else '-1.0'}
            Path adherence: {build_path_adherence_string(new_node, old_idx, new_idx, global_path)}
            Total reward: {reward}
            """)
    
            return next_state, reward, self.done
        
        def get_valid_actions(self) -> List[int]:
            """Return valid actions based on current state."""
            if self.in_position:
                return list(range(VERTICAL_START, TOTAL_ACTIONS)) + [HOLD, EXIT]
            else:
                return list(range(NUM_CRYPTOS)) + [HOLD]
            
        def _find_and_set_valid_path(self, new_node):
            """Find the full valid path that includes the new node"""
            for path in self.valid_paths:
                if new_node in path:
                    idx = path.index(new_node)
                    self.current_valid_path = path[:idx+1]
                    return

        def _evaluate_transition(self, old_node, new_node):
            """
            Evaluate the type of transition:
            - 'same_path' if both nodes are in at least one common path.
            - 'different_path' if both nodes are in different paths.
            - 'invalid' otherwise.
            """
            old_paths = set(self.node_to_paths.get(old_node, []))
            new_paths = set(self.node_to_paths.get(new_node, []))

            common_paths = old_paths.intersection(new_paths)

            if common_paths:
                return "same_path"
            elif old_paths or new_paths:
                return "different_path"
            else:
                return "invalid"

        def _completed_valid_path(self):
            """Check if the current path is a complete valid path."""
            return self.current_path in self.path_indices.values()

        def _get_state(self):
            """Enhanced state representation with more context"""
            node_id = self.node2id[self.current_node]
            
            node_embedding = self.node_embeddings[node_id]
            
            in_position = torch.tensor([1.0] if self.in_position else [0.0])
            
            current_idx = -1
            path_progress = torch.zeros(1)
            if self.current_node in GLOBAL_PATHS[0]:
                current_idx = GLOBAL_PATHS[0].index(self.current_node)
                path_progress = torch.tensor([current_idx / len(GLOBAL_PATHS[0])])
            
            node_data = self.structured_data[self.current_node]
            is_ar = torch.tensor([1.0 if node_data['state'] == 'AR' else 0.0])
            
            state_vec = torch.cat((
                node_embedding,
                in_position,
                path_progress,
                is_ar
            ), dim=0)
            
            return state_vec

        def get_correct_actions(self, previous_node, current_node, in_position_before_action):
            correct = set()
            global_path = GLOBAL_PATHS[0] if GLOBAL_PATHS else []

            if not global_path or current_node not in global_path:
                return [EXIT] if in_position_before_action else []

            try:
                current_idx = global_path.index(current_node)
                next_node = global_path[current_idx + 1]

                if next_node[1] == current_node[1]:
                    if in_position_before_action:
                        vertical_action = VERTICAL_START + next_node[0]
                        correct.add(vertical_action)

                elif next_node[1] == current_node[1] + 1:
                    if next_node[0] == current_node[0]:
                        correct.add(HOLD)
                    else:
                        if not in_position_before_action:
                            correct.add(next_node[0])

                if current_node == global_path[-1]:
                    correct.add(EXIT)

            except IndexError:
                correct.add(EXIT)

            return sorted(list(correct))

    @dataclass
    class GraphContainer:
        graph: nx.Graph
        pos: Dict
        state_data: Dict
        horizontal_edges: List
        vertical_edges: List
        crypto_names: List
        structured_data: Dict
        metadata: Dict

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def main():
        set_seed()
        try:
            G, pos, node_colors, horizontal_edges, vertical_edges, state_data, crypto_names, structured_data = process_csv_data()

            print("\nBuilding paths...")
            path_builder = PathBuilder(structured_data)
            path_builder.build_paths()
            path_builder.print_paths()

            if not GLOBAL_PATHS:
                print("No valid paths found. Proceeding without paths...")

            print("\nInitializing agent & environment...")
            env = PathFollowingEnvironment(
                structured_data=structured_data,
                max_time=DEFAULT_NUM_ROWS
            )
            agent = create_agent(env)

            print("\nCreating visualization...")
            graph_plotter = GraphPlotter(
                structured_data=structured_data,
                pos=pos,
                node_colors=node_colors,
                state_data=state_data,
                path_builder=None
            )
            
            graph_plotter.fig.canvas.mpl_connect('draw_event', 
                lambda _: graph_plotter._optimized_draw())
                
            graph_plotter.plot_graph()
            graph_plotter.setup_agent(agent, env)

            fig = plt.gcf()
            fig.canvas.mpl_connect('key_press_event', graph_plotter.on_key_press)
            plt.ion()
            fig.canvas.manager.set_window_title('Graph Analysis (Press P to pause)')
            plt.show(block=True)

        except Exception as e:
            print(f"Error in main: {str(e)}")

    if __name__ == "__main__":
        main()
