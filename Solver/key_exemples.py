# !pip install torch matplotlib networkx
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from Solver import *
from repo_preliminary_instances import *

# Assuming these are available in your local environment
# from Solver import * # from repo_preliminary_instances import *

def generate_max_cut(n_nodes: int, p_edge: float = 0.5):
    G = nx.erdos_renyi_graph(n_nodes, p_edge, seed=42)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 10)
    
    W = nx.to_numpy_array(G)
    J = -torch.from_numpy(W).float()
    h = torch.zeros(n_nodes)
    return J, h, G

# def generate_profitable_cycles(n_nodes=3, lam=2):
#     # 1. Create a Fully Connected Directed Graph (Complete Digraph)
#     DG = nx.complete_graph(n_nodes, create_using=nx.DiGraph())
    
#     nodes = list(DG.nodes())
#     target_cycle_edges = []
    
#     # 2. Define the Gains
#     # We iterate through every edge in the complete graph
#     for u, v in DG.edges():
#         # Check if this edge is part of the unique profitable cycle (i -> i+1)
#         if v == (u + 1) % n_nodes:
#             DG[u][v]['gain'] = 1.2
#             target_cycle_edges.append((u, v))
#         else:
#             # All other pairwise connections
#             DG[u][v]['gain'] = 0.5

#     edges = list(DG.edges())
#     n_edges = len(edges)
    
#     # 3. Ising Mapping logic (Appendix A Formulation)
#     J = torch.zeros((n_edges, n_edges))
#     h = torch.zeros(n_edges)
    
#     def get_delta(edge_idx, vertex_k):
#         u, v = edges[edge_idx]
#         if v == vertex_k: return 1
#         if u == vertex_k: return -1
#         return 0

    # Fill J and h matrices
    # for i in range(n_edges):
    #     u_i, v_i = edges[i]
    #     gain_i = DG[u_i][v_i]['gain']
        
    #     # Linear field: 0.5 * log(c) - (lambda/2) * sum(delta^2)
    #     # Note: sum(delta^2) for any edge is always 2 (one -1, one 1)
    #     h[i] = 0.5 * np.log(gain_i) - (lam / 2.0) * 2.0
        
    #     for j in range(i + 1, n_edges):
    #         # Quadratic coupling: -(lambda/2) * sum(delta_i * delta_j)
    #         p_ij = sum(get_delta(i, v) * get_delta(j, v) for v in range(n_nodes))
    #         val = -(lam / 2.0) * p_ij
            
    #         J[i, j] = val
    #         J[j, i] = val
            
    # return J, h, (DG, edges, target_cycle_edges)

def visualize_max_cut(G, spins):
    pos = nx.spring_layout(G)
    colors = ['#ff9999' if s > 0 else '#99ff99' for s in spins]
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=800, edge_color='gray')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Max-Cut: Red vs Green Sets")
    plt.show()

# def visualize_cycles(DG, edges, spins, title="SBM Result: Profitable Cycles"):
#     plt.figure(figsize=(10, 7), facecolor='white')
#     pos = nx.circular_layout(DG)
    
#     selected_edges = [edges[i] for i, s in enumerate(spins) if s > 0]
#     non_selected_edges = [edges[i] for i, s in enumerate(spins) if s <= 0]
    
#     nx.draw_networkx_nodes(DG, pos, node_size=1000, node_color='#a1c9f4', 
#                            edgecolors='black', linewidths=1.5)
#     nx.draw_networkx_labels(DG, pos, font_size=12, font_weight='bold')

#     nx.draw_networkx_edges(DG, pos, edgelist=non_selected_edges, 
#                            edge_color='#e0e0e0', width=1.0, 
#                            arrowstyle='-|>', arrowsize=15, 
#                            connectionstyle='arc3, rad=0.1') 

#     nx.draw_networkx_edges(DG, pos, edgelist=selected_edges, 
#                            edge_color='#ff4b2b', width=3.5, 
#                            arrowstyle='-|>', arrowsize=25,
#                            connectionstyle='arc3, rad=0.1')

#     # Fix: Ensure we don't try to draw labels for an empty list if SBM failed
#     if selected_edges:
#         edge_labels = {edge: DG[edge[0]][edge[1]]['gain'] for edge in selected_edges}
#         nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, 
#                                      font_color='red', font_weight='bold', label_pos=0.3)

#     plt.title(title, fontsize=15, pad=20)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
def run_demo():
    # 2. Max-Cut
    print("--- 2. Max-Cut ---")
    J_mc, h_mc, G_mc = generate_max_cut(6)
    # Using default config for a simple graph
    res_mc = solve(J_mc, h_mc)
    true_e_mc, _ = brute_force_optimum(J_mc, h_mc)
    print(f"Optimal Energy: {true_e_mc:.3f} | SBM: {res_mc.energy:.3f}")
    visualize_max_cut(G_mc, res_mc.spins)

    # # # 4. Profitable Cycles
    # # print("--- 4. Profitable Cycles ---")
    # # # FIX: Unpack 3 values from the metadata tuple
    # # J_pc, h_pc, (DG_pc, edges_pc, target_pc) = generate_profitable_cycles()
    
    # # SBM tuning for constrained problems
    # pc_config = SBConfig(n_steps=2000, xi0=0.7, n_parallel=256,p_max=15)
    # res_pc = solve(J_pc, h_pc, config=pc_config)
    # print(J_pc)
    # print(h_pc)
    # print(f"SBM Energy: {res_pc.energy:.3f}")
    # print(res_pc.spins)
    # print(brute_force_optimum(J_pc,h_pc))
    # visualize_cycles(DG_pc, edges_pc, res_pc.spins)

if __name__ == "__main__":
    run_demo()