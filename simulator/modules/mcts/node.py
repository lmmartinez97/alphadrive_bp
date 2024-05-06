from typing import Dict, List

class Node(object):
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes:
      - visit_count (int): Number of times the node has been visited during the search.
      - prior (float): Prior probability of selecting the node.
      - value_sum (float): Sum of values encountered during the search.
      - children (Dict[int, int]): Dictionary of child nodes, where the key is the action and the value is the visitation frequency of the child node.
    """

    def __init__(self, prior: float) -> None:
        """
        Initializes a new Node with the given prior probability.

        Args:
            prior (float): Prior probability of selecting the node.
        """
        self.visit_count: int = 0
        self.state: List[float] = []
        self.prior: float = prior
        self.value_sum: float = 0
        self.children: Dict[int, 'Node'] = {}

    def expanded(self) -> bool:
        """
        Checks if the node has been expanded (has children).

        Returns:
            bool: True if the node has children, False otherwise.
        """
        return len(self.children) > 0
    
    def assign_state(self, state: List[float]) -> None:
        """
        Assigns the state to the node.

        Args:
            state (List[float]): The state to be assigned to the node.
        """
        self.state = state

    def value(self) -> float:
        """
        Returns the average value of the node.

        Returns:
            float: The average value of the node.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count