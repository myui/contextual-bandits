import random
from typing import List, Callable, Tuple

class ContextualBandit:
    """
    Contextual Bandit implementation with offline policy evaluation.

    Attributes:
        num_arms (int): Number of arms (actions) the bandit can choose from.
        num_features (int): Number of features in the context.
        learning_rate (float): Learning rate for updating the weights.
        weights (List[List[float]]): List of weight vectors, one for each arm.
    """

    def __init__(self, num_arms: int, num_features: int, learning_rate: float = 0.1):
        """
        Initializes the ContextualBandit with the specified number of arms and context features.

        Args:
            num_arms (int): Number of arms (actions).
            num_features (int): Number of features in the context.
            learning_rate (float): Learning rate for updating the weights. Default is 0.1.
        """
        self.num_arms = num_arms
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights: List[List[float]] = [self.initialize_weights(num_features) for _ in range(num_arms)]

    def initialize_weights(self, num_features: int) -> List[float]:
        """
        Initializes the weights for an arm.

        Args:
            num_features (int): Number of features in the context.

        Returns:
            List[float]: A list of randomly initialized weights.
        """
        return [random.random() for _ in range(num_features)]

    def compute_dot_product(self, weights: List[float], context: List[float]) -> float:
        """
        Computes the dot product between weights and context.

        Args:
            weights (List[float]): Weight vector for an arm.
            context (List[float]): Context vector.

        Returns:
            float: The dot product of the weights and context.
        """
        return sum(w * c for w, c in zip(weights, context))

    def choose_arm(self, context: List[float]) -> int:
        """
        Chooses an arm based on the current context.

        Args:
            context (List[float]): The current context vector.

        Returns:
            int: The index of the chosen arm.
        """
        values = [self.compute_dot_product(self.weights[arm], context) for arm in range(self.num_arms)]
        return values.index(max(values))

    def update_weights(self, chosen_arm: int, context: List[float], reward: float):
        """
        Updates the weights for the chosen arm based on the received reward.

        Args:
            chosen_arm (int): The index of the chosen arm.
            context (List[float]): The context vector.
            reward (float): The reward received for choosing the arm.
        """
        for i in range(self.num_features):
            dotp = self.compute_dot_product(self.weights[chosen_arm], context)
            self.weights[chosen_arm][i] += self.learning_rate * (reward - dotp) * context[i]
        self.post_update_weights(chosen_arm)


    def post_update_weights(self, chosen_arm: int):
        """
        Post update weights for the chosen arm.

        Args:
            chosen_arm (int): The index of the chosen arm.
        """
        pass

    def run(self, T: int, get_current_context: Callable[[], List[float]], get_reward: Callable[[int], float]):
        """
        Runs the contextual bandit algorithm for T rounds.

        Args:
            T (int): Number of rounds to run the algorithm.
            get_current_context (Callable[[], List[float]]): Function to get the current context.
            get_reward (Callable[[int], float]): Function to get the reward for a chosen arm.
        """
        for t in range(T):
            context = get_current_context()
            chosen_arm = self.choose_arm(context)
            reward = get_reward(chosen_arm)
            self.update_weights(chosen_arm, context, reward)

    def offline_policy_evaluation(self, historical_data: List[Tuple[List[float], int, float]]) -> float:
        """
        Evaluates the policy using offline data.

        Args:
            historical_data (List[Tuple[List[float], int, float]]): List of tuples (context, actual_arm, reward) representing historical interactions.

        Returns:
            float: The average reward of the policy on the historical data.
        """
        total_reward = 0
        num_samples = len(historical_data)

        for context, actual_arm, reward in historical_data:
            chosen_arm = self.choose_arm(context)
            if chosen_arm == actual_arm:
                total_reward += reward

        average_reward = total_reward / num_samples
        return average_reward
