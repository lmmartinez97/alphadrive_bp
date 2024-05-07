from typing import Dict, List

class Node(object):
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes:
        - visit_count (int): Number of times the node has been visited during the search.
        - state (List[float]): State of the node.
        - value_sum (float): Sum of values encountered during the search.
        - children (Dict[int, int]): Dictionary of child nodes, where the key is the action and the value is the visitation frequency of the child node.
    """

    def __init__(self) -> None:
        """
        Initializes a new Node.
        """
        self.visit_count: int = 0
        self.state: List[float] = []
        self.value_sum: float = 0
        self.children: Dict[int, Node] = {'0': None, '1': None, '2': None}

    def assign_state(self, state: List[float]) -> None:
        """
        Assigns the state to the node.

        Args:
            state (List[float]): The state to be assigned to the node.
        """
        self.state = state

    def update_child(self, child_visited: int = None, value: float = None) -> None:
        """
        Updates the visit count and children information of the node.

        Args:
            child_visited (int): The child node that was visited.
            value (float): The value of the child node.
        """
        if child_visited is not None:
            if child_visited not in self.children.keys():
                raise ValueError("Child not found in children.")
            self.children[child_visited].value_sum += value
            self.children[child_visited].visit_count += 1
        else:
            raise ValueError("Child visted not provided.")

    def value(self) -> float:
        """
        Returns the average value of the node.

        Returns:
            float: The average value of the node.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def policy(self) -> Dict[int, float]:
        """
        Returns the policy of the node.

        Returns:
            Dict[int, float]: The policy of the node.
        """
        policy = {}
        for action, count in self.children.items():
            policy[action] = count / self.visit_count
        return policy