import random

class ContextualBandit:
    """
    Contextual Bandit implementation with offline policy evaluation.

    Attributes:
        num_arms (int): Number of arms (actions) the bandit can choose from.
        num_features (int): Number of features in the context.
        learning_rate (float): Learning rate for updating the weights.
        weights (list): List of weight vectors, one for each arm.
    """

    def __init__(self, num_arms, num_features, learning_rate=0.1):
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
        self.weights = [self.initialize_weights(num_features) for _ in range(num_arms)]

    def initialize_weights(self, num_features):
        """
        Initializes the weights for an arm.

        Args:
            num_features (int): Number of features in the context.

        Returns:
            list: A list of randomly initialized weights.
        """
        return [random.random() for _ in range(num_features)]

    def compute_dot_product(self, weights, context):
        """
        Computes the dot product between weights and context.

        Args:
            weights (list): Weight vector for an arm.
            context (list): Context vector.

        Returns:
            float: The dot product of the weights and context.
        """
        return sum(w * c for w, c in zip(weights, context))

    def choose_arm(self, context):
        """
        Chooses an arm based on the current context.

        Args:
            context (list): The current context vector.

        Returns:
            int: The index of the chosen arm.
        """
        values = [self.compute_dot_product(self.weights[arm], context) for arm in range(self.num_arms)]
        return values.index(max(values))

    def update_weights(self, chosen_arm, context, reward):
        """
        Updates the weights for the chosen arm based on the received reward.

        Args:
            chosen_arm (int): The index of the chosen arm.
            context (list): The context vector.
            reward (float): The reward received for choosing the arm.
        """
        for i in range(self.num_features):
            self.weights[chosen_arm][i] += self.learning_rate * (reward - self.compute_dot_product(self.weights[chosen_arm], context)) * context[i]

    def run(self, T, get_current_context, get_reward):
        """
        Runs the contextual bandit algorithm for T rounds.

        Args:
            T (int): Number of rounds to run the algorithm.
            get_current_context (function): Function to get the current context.
            get_reward (function): Function to get the reward for a chosen arm.
        """
        for t in range(T):
            context = get_current_context()
            chosen_arm = self.choose_arm(context)
            reward = get_reward(chosen_arm)
            self.update_weights(chosen_arm, context, reward)

    def offline_policy_evaluation(self, historical_data):
        """
        Evaluates the policy using offline data.

        Args:
            historical_data (list): List of tuples (context, actual_arm, reward) representing historical interactions.

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


