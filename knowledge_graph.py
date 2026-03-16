from typing import *
from pgmpy.models.DiscreteBayesianNetwork import DiscreteBayesianNetwork
from pgmpy.inference import CausalInference
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
import heapq
import networkx as nx

class CausalKnowledgeGraph:
    MASTERY_THRESHOLD = 0.8

    def __init__(
        self,
        *,
        nodes: Optional[list[str]] = None,
        edges: Optional[list[tuple[str, str]]] = None,
        class_topic_progressions: Optional[list[dict[str, float]]] = None,
        student_topic_progressions: Optional[dict[str, float]] = None,
    ):
        self.nodes: list[str] = nodes or []
        self.edges: list[tuple[str, str]] = edges or []
        self.class_topic_progressions: list[dict[str, float]] = class_topic_progressions or []
        self.student_topic_progressions: dict[str, float] = student_topic_progressions or {}

        self.model = DiscreteBayesianNetwork()
        self.model.add_nodes_from(self.nodes)

        # adding edges one-by-one so we can tell the user which edge(s) caused a cycle
        self.cycle: list[tuple[str, str]] = self.check_for_cycle()
        if self.cycle:
            raise ValueError(
                f"Cycle detected in the graph: {self.cycle}. Please check your edges."
            )

        self.initialize_cpts(
            data_table=self.process_raw_student_data(data=self.class_topic_progressions)
        )

        print("model created successfully!")

    def process_raw_student_data(self, *, data: list[dict[str, float]]) -> pd.DataFrame:
        """
        Process the raw student data into a format suitable for the creation of the
        Bayesian network CPTs

        This function takes a list of dictionaries representing student data and converts
        it into a pandas DataFrame. Each dictionary contains the overall scores for a topic
        across all lessons the student has completed as a percentage

        The percentages are then turned into binary values depending on whether the student's
        percentage is above or below a certain threhold. If the percentage is above the threshold,
        the value is set to 1 (indicating mastery of the topic), otherwise it is set to 0

        Parameters:
            data (list[dict[str, float]]): The list of dictionaries representing student data for
            a class

        Returns:
            DataFrame: A pandas DataFrame containing the processed student data, with each row
            representing a student and each column representing a topic

        Example:

                s_1 = {"t_1": 37 / 50, "t_2": 28 / 35, "t_3": 18 / 30, "t_4": 0}
                s_2 = {"t_1": 40 / 50, "t_2": 30 / 35, "t_3": 21 / 30, "t_4": 0}
                s_3 = {"t_1": 27 / 50, "t_2": 27 / 35, "t_3": 17 / 30, "t_4": 0}
                s_4 = {"t_1": 26 / 50, "t_2": 20 / 35, "t_3": 13 / 30, "t_4": 0}

                data_table: DataFrame = process_raw_student_data([s_1, s_2, s_3, s_4])
                print(data_table)

                        t_1  t_2  t_3  t_4
                s_1    1    1    0    0
                s_2    1    1    1    0
                s_3    0    1    0    0
                s_4    0    0    0    0
        """
        df = pd.DataFrame(data)

        # Ensure every graph node exists as a column, even if never observed.
        df = df.reindex(columns=self.nodes)

        df.index = [f"student_{i+1}" for i in range(len(df))]

        def to_binary_or_na(x):
            if pd.isna(x):
                return pd.NA
            return 1 if float(x) >= self.MASTERY_THRESHOLD else 0

        return df.map(to_binary_or_na)

    def initialize_cpts(
        self,
        *,
        data_table: pd.DataFrame,
        alpha: float = 1,
    ) -> None:
        """
        Initialize the Conditional Probability Tables (CPTs) for the Bayesian network.
        This function takes a pandas DataFrame containing the transformed student data
        and creates TabularCPD objects for each topic in the Bayesian network

        Parameters:

            data_table: DataFrame: The pandas DataFrame containing the transformed student data
            nodes: list[str]: The list of nodes (topics) in the Bayesian network
            model: DiscreteBayesianNetwork: The Bayesian network model to which the CPTs will be added
            alpha: float: The smoothing parameter for the CPTs (default is 1)

        Example:

                >>>
                data_table = process_raw_student_data([s_1, s_2, s_3, s_4])
                nodes = ["t_1", "t_2", "t_3", "t_4"]
                edges = [
                    ("t_1", "t_2"),
                    ("t_1", "t_3"),
                    ("t_2", "t_3"),
                    ("t_3", "t_4"),
                ]
                test_model = DiscreteBayesianNetwork()
                test_model.add_nodes_from(nodes)
                test_model.add_edges_from(edges)
                cpts = initialize_cpts(data_table, nodes, test_model)

                >>> CPTs for each topic in the Bayesian network:
                +--------+-----+
                | t_1(0) | 0.5 |
                +--------+-----+
                | t_1(1) | 0.5 |
                +--------+-----+

                >>>
                +--------+--------+--------+
                | t_1    | t_1(0) | t_1(1) |
                +--------+--------+--------+
                | t_2(0) | 0.5    | 0.25   |
                +--------+--------+--------+
                | t_2(1) | 0.5    | 0.75   |
                +--------+--------+--------+

                >>>
                +--------+--------------------+--------------------+--------+--------+
                | t_1    | t_1(0)             | t_1(0)             | t_1(1) | t_1(1) |
                +--------+--------------------+--------------------+--------+--------+
                | t_2    | t_2(0)             | t_2(1)             | t_2(0) | t_2(1) |
                +--------+--------------------+--------------------+--------+--------+
                | t_3(0) | 0.6666666666666666 | 0.6666666666666666 | 0.5    | 0.5    |
                +--------+--------------------+--------------------+--------+--------+
                | t_3(1) | 0.3333333333333333 | 0.3333333333333333 | 0.5    | 0.5    |
                +--------+--------------------+--------------------+--------+--------+

                >>>
                +--------+--------+--------------------+
                | t_3    | t_3(0) | t_3(1)             |
                +--------+--------+--------------------+
                | t_4(0) | 0.8    | 0.6666666666666666 |
                +--------+--------+--------------------+
                | t_4(1) | 0.2    | 0.3333333333333333 |
                +--------+--------+--------------------+

        """
        states: list[int] = [0, 1]  # 0 = not mastered, 1 = mastered

        print("nodes:", self.nodes)
        print("data_table columns:", list(data_table.columns))

        for node in self.nodes:
            parents: list[str] = self.model.get_parents(node)
            cols = [node, *parents]
            observed_table = data_table[cols].dropna()

            if parents:
                # Only use students who have observed values for node and all parents
                if observed_table.empty:
                    # No data: fall back to uniform CPT given all parent configurations
                    num_parent_configs = 2 ** len(parents)
                    cpt = TabularCPD(
                        variable=node,
                        variable_card=2,
                        values=[[0.5] * num_parent_configs, [0.5] * num_parent_configs],
                        evidence=parents,
                        evidence_card=[2] * len(parents),
                        state_names={node: states, **{p: states for p in parents}},
                    )
                    self.model.add_cpds(cpt)
                    continue

                joint_counts: pd.Series[int] = observed_table.groupby([node, *parents]).size()                
                joint_df: pd.DataFrame = joint_counts.reset_index(name="count")

                # this is needed b/c groupby does not return a count for combinations
                # that are not seen in the data (kinda troll of the library tbh)
                combinations = pd.MultiIndex.from_product(
                    [states] * (len(parents) + 1), names=[node, *parents]
                )
                child_given_parents: pd.DataFrame = (
                    joint_df.set_index([node, *parents])
                    .reindex(combinations, fill_value=0)
                    .reset_index()
                )

                # smoothing to avoid undefined probabilities
                child_given_parents["count"] += alpha
                child_given_parents["prob"] = child_given_parents.groupby(parents)[
                    "count"
                ].transform(lambda x: x / x.sum())

                node_values: list[str] = child_given_parents["prob"].values.tolist()
                cpt = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[
                        node_values[: (len(node_values) // 2)],
                        node_values[(len(node_values) // 2) :],
                    ],
                    evidence=parents,
                    evidence_card=[2] * len(parents),
                    state_names={node: states, **{p: states for p in parents}},
                )
                self.model.add_cpds(cpt)
            else:
                node_data: pd.Series[str] = data_table[node].dropna()
                if node_data.empty:
                    cpt = TabularCPD(
                        variable=node,
                        variable_card=2,
                        values=[[0.5], [0.5]],
                        state_names={node: states},
                    )
                    self.model.add_cpds(cpt)
                    continue
                value_counts: pd.Series[int] = node_data.value_counts()

                # a series with both states, filling missing ones with 0
                complete_counts = pd.Series(0, index=states)
                complete_counts.update(value_counts)

                smoothed_counts = complete_counts.add(alpha)
                probabilities = smoothed_counts / smoothed_counts.sum()

                cpt = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[probabilities[0]], [probabilities[1]]],
                    state_names={node: states},
                )
                self.model.add_cpds(cpt)

    def get_nodes(self) -> list[str]:
        """
        Get the nodes in the knowledge graph.

        Returns:
            list[str]: The nodes in the knowledge graph.
        """
        return self.nodes

    def get_edges(self) -> list[tuple[str, str]]:
        """
        Get the edges in the knowledge graph.

        Returns:
            list[tuple[str, str]]: The edges in the knowledge graph.
        """
        return self.edges

    def get_ancestor_topics(self, topic: str) -> set[str]:
        """
        Get the ancestors of the given topic in the knowledge graph.

        Parameters:
            topic (str): The topic for which to find the ancestors.

        Returns:
            set[str]: The ancestors of the given topic.
        """
      
        return set(self.model.get_ancestors(topic))

    def get_descendant_topics(self, topic: str) -> list[str]:
        return list(nx.descendants(self.model, topic))

    def check_for_cycle(self) -> list[tuple[str, str]]:
        """
        Check for cycles in the knowledge graph by attempting to add edges one-by-one.

        If we try to do add all at once, and there's a cycle, the model will throw a nasty
        exception on the first edge that causes the cycle. This is not user friendly, and
        we want to be able to tell the user which edge(s) caused the cycle.

        Returns:
            list[tuple[str, str]]: The list of edges that cause a cycle
        """
        edges_that_cause_cycle: list[tuple[str, str]] = []
        for edge in self.edges:
            # if edge[0] == edge[1]:
            #     raise ValueError(f"Cycle detected in the graph: {edge}")
            try:
                self.model.add_edge(edge[0], edge[1])
            except Exception as e:
                edges_that_cause_cycle.append(edge)
                print(f"Cycle detected: {e}")

        print(f"edges that cause cycle: {edges_that_cause_cycle}")
        return edges_that_cause_cycle

    def determine_review_topics(self, *, student_topic_progressions: dict[str, float], alpha: float = 1.0) -> dict[str, float]:

        valid_nodes = set(self.nodes)
        student_topic_progressions = {
            k: v for k, v in student_topic_progressions.items()
            if k in valid_nodes
        }

        inference = CausalInference(self.model)

        # Step 1: Build set of unmastered concepts C
        unmastered_nodes: set[str] = set()
        for node, mastery in student_topic_progressions.items():
            if mastery <= self.MASTERY_THRESHOLD:
                unmastered_nodes.add(node)

        print(f"Set of unmastered nodes: {unmastered_nodes}")
        print(f"student_topic_progressions: {student_topic_progressions}")

        # Step 2: Build evidence dictionary
        mastered_nodes: dict[str, int] = {}
        for node, mastery in student_topic_progressions.items():
            if mastery >= self.MASTERY_THRESHOLD:
                mastered_nodes[node] = 1

        print(f"List of mastered nodes: {mastered_nodes}")

        seen_nodes: set[str] = set(student_topic_progressions.keys())
        rev_priors: dict = {}
        for concept in unmastered_nodes:
            # Set of topic descendants from current node
            descendants: list[str] = self.get_descendant_topics(concept)

            # Returns the interesection of descendants that are also unmastered
            # If the descendant is not in the unmastered topics set, remove it from the descendants list
            # We want to keep only ones that are unmastered
            descendants = [d for d in descendants if d in seen_nodes and d in unmastered_nodes]

            print(f"This is the descendants list: {descendants}")

            # Compute descendant deltas
            deltas: list[float] = []
            for D in descendants:
                pre_review_proficiency = inference.query(
                    variables=[D],
                    evidence=mastered_nodes,
                ).values[1]

                print(f"pre review proficiency {pre_review_proficiency}")

                post_review_proficiency = inference.query(
                    variables=[D],
                    evidence=mastered_nodes,
                    do={concept: 1},
                ).values[1]

                print(f"post review proficiency {post_review_proficiency}")
                print(f"evidence list for queries: {mastered_nodes}")

                delta_importance = float(post_review_proficiency - pre_review_proficiency)
                deltas.append(delta_importance)

            k = len(descendants)
            avg_desc_delta = (sum(deltas) / k) if k > 0 else 0.0

            # Self improvement term
            # pre_self = P(concept=1 | evidence)
            # post_self under do(concept=1) is 1.0 by definition
            pre_self = inference.query(variables=[concept], evidence=mastered_nodes).values[1]
            delta_self = float(1.0 - pre_self)
            print(f"{concept} self delta: {delta_self}")

            # Weighting: self weight shrinks as descendant count grows
            # Leaf nodes (k=0): w_self=1, w_desc=0
            w_self = 1.0 / (k + 1)
            w_desc = 1.0 - w_self

            score = (w_desc * avg_desc_delta) + (w_self * delta_self)
            rev_priors[concept] = score

        return rev_priors

    def is_topic_mastered(self, topic: str) -> bool:
        return self.student_topic_progressions.get(topic, 0.0) >= self.MASTERY_THRESHOLD
