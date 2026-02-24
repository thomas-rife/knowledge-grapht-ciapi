# from typing import *
# from pgmpy.models.DiscreteBayesianNetwork import DiscreteBayesianNetwork
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import CausalInference
# import pandas as pd
# from knowledge_graph import CausalKnowledgeGraph

# nodes = ["t_1", "t_2", "t_3", "t_4"]
# edges = [
#     ("t_1", "t_2"),
#     ("t_1", "t_3"),
#     ("t_2", "t_3"),
#     ("t_3", "t_4"),
# ]

# # dummy student data
# s_2 = {"t_1": 40 / 50, "t_2": 30 / 35, "t_3": 21 / 30, "t_4": 0}
# s_3 = {"t_1": 27 / 50, "t_2": 27 / 35, "t_3": 17 / 30, "t_4": 0}
# s_4 = {"t_1": 26 / 50, "t_2": 20 / 35, "t_3": 13 / 30, "t_4": 0}
# s_1 = {"t_1": 37 / 50, "t_2": 28 / 35, "t_3": 18 / 30, "t_4": 0}

# test_model = CausalKnowledgeGraph(nodes = nodes, edges = edges, class_topic_progressions = [s_1, s_2, s_3, s_4])

# inference = CausalInference(test_model.model)

# test_query = inference.query(
#     variables=["t_2"],
#     do={"t_1": 1},
#     evidence={},
# ).values
# print(f"test query: {test_query}")

# # ---------------- REVIEW TEST ----------------

# print("\n--- REVIEW TEST ---")

# print(f"CPD t_3: {test_model.model.get_cpds("t_3")}")

# # Example student (partially mastered)
# student_progression = {
#     "t_1": 0.9,   # mastered
#     "t_2": 0.6,   # unmastered
#     "t_3": 0.5,   # unmastered
#     # omit t_4 to simulate unseen
# }

# review_result = test_model.determine_review_topics(
#     student_topic_progressions=student_progression
# )

# print("Review priorities:", review_result)

# # Basic sanity checks
# assert review_result is not None, "Returned None"
# assert isinstance(review_result, (dict, list)), "Unexpected return type"

# # If dict
# if isinstance(review_result, dict):
#     for topic, score in review_result.items():
#         print(f"{topic}: {score}")

# # If sorted list of tuples
# if isinstance(review_result, list):
#     for topic, score in review_result:
#         print(f"{topic}: {score}")

# print("--- END REVIEW TEST ---")

# /Users/thomasrife/Documents/knowledge grapht/knowledge-graph-api/test_model.py
from __future__ import annotations

import random
import networkx as nx
from typing import Dict, List, Tuple

from pgmpy.inference import CausalInference

from knowledge_graph import CausalKnowledgeGraph

# Larger, more realistic knowledge graph example
nodes: List[str] = [
    "Probability Theory",
    "Conditioning",
    "Law of Total Probability",
    "Distributions",
    "Bayesian Networks",
    "BN Exact Inference",
    "BN Approximate Inference",
    "Hidden Markov Models",
    "Forward Algorithm Filtering",
    "Particle Filtering",
    "Prior / Rejection Sampling",
    "Gibbs Sampling",
    "Supervised Learning",
    "Naive Bayes Classifiers",
    "Linear Perceptrons",
    "Parameter Estimation",
    "Logistic Regression",
    "Artificial Neural Networks",
]

# Edges provided as 1-indexed "i->j" references
edge_strs: List[str] = [
    "1->2",
    "1->3",
    "1->4",
    "2->5",
    "4->5",
    "3->5",
    "5->6",
    "5->7",
    "2->8",
    "3->8",
    "8->9",
    "8->10",
    "7->10",
    "7->11",
    "7->12",
    "1->13",
    "5->14",
    "13->14",
    "13->15",
    "16->15",
    "16->14",
    "15->17",
    "16->17",
    "17->18",
    "16->18",
]


def parse_edges(edge_specs: List[str], node_names: List[str]) -> List[Tuple[str, str]]:
    edges_out: List[Tuple[str, str]] = []
    for spec in edge_specs:
        left, right = spec.split("->")
        i = int(left) - 1
        j = int(right) - 1
        edges_out.append((node_names[i], node_names[j]))
    return edges_out


edges: List[Tuple[str, str]] = parse_edges(edge_strs, nodes)


def clamp(x: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, x))


def generate_synthetic_class_data(
    *,
    n_students: int,
    nodes: List[str],
    edges: List[Tuple[str, str]],
    missing_prob: float = 0.12,
    seed: int = 7,
) -> List[Dict[str, float]]:

    random.seed(seed)

    parents: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        parents[v].append(u)

    # Compute true topological order
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible as e:
        raise ValueError("Graph contains a cycle") from e

    def sample_accuracy(mastered: bool) -> float:
        if mastered:
            return random.uniform(0.78, 0.95)
        return random.uniform(0.30, 0.66)

    data: List[Dict[str, float]] = []

    for _ in range(n_students):
        mastered_state: Dict[str, int] = {}
        accs: Dict[str, float] = {}

        for node in topo:
            ps = parents[node]

            base = 0.35
            if "Inference" in node or "Sampling" in node or "Filtering" in node:
                base = 0.25
            if node in ("Probability Theory", "Distributions", "Supervised Learning"):
                base = 0.45

            if not ps:
                p_master = base
            else:
                parent_mean = sum(mastered_state[p] for p in ps) / len(ps)
                p_master = clamp(base + 0.45 * parent_mean)

            mastered = 1 if random.random() < p_master else 0
            mastered_state[node] = mastered

            if random.random() < missing_prob:
                continue

            accs[node] = sample_accuracy(bool(mastered))

        data.append(accs)

    return data


class_topic_progressions = generate_synthetic_class_data(
    n_students=80,
    nodes=nodes,
    edges=edges,
    missing_prob=0.10,
    seed=11,
)

test_model = CausalKnowledgeGraph(
    nodes=nodes,
    edges=edges,
    class_topic_progressions=class_topic_progressions,
)

inference = CausalInference(test_model.model)

print("\n--- MODEL BUILT ---")
print(f"Num nodes: {len(nodes)}")
print(f"Num edges: {len(edges)}")

for inspect_node in ["Bayesian Networks", "BN Approximate Inference", "Particle Filtering"]:
    print(f"\n--- CPD {inspect_node} ---")
    print(test_model.model.get_cpds(inspect_node))


student_progression = {
    "Probability Theory": 0.90,
    "Conditioning": 0.62,
    "Law of Total Probability": 0.64,
    "Distributions": 0.88,
    "Bayesian Networks": 0.61,
    "BN Exact Inference": 0.58,
    "BN Approximate Inference": 0.60,
    "Hidden Markov Models": 0.63,
    "Forward Algorithm Filtering": 0.59,
    "Particle Filtering": 0.55,
    "Gibbs Sampling": 0.57,
    "Supervised Learning": 0.92,
    "Naive Bayes Classifiers": 0.66,
    "Linear Perceptrons": 0.81,
    "Parameter Estimation": 0.89,
    "Logistic Regression": 0.63,
    "Artificial Neural Networks": 0.60,
}

print("\n--- REVIEW TEST ---")
review_result = test_model.determine_review_topics(
    student_topic_progressions=student_progression
)

print("Review priorities (raw):", review_result)

if isinstance(review_result, dict):
    ranked = sorted(review_result.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 recommended review topics:")
    for topic, score in ranked[:10]:
        print(f"  {topic}: {float(score):.4f}")