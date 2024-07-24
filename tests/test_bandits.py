import pytest
import random

from knn_bandits.bandits import ContextualBandit

# Test cases
def test_initialize_weights():
    bandit = ContextualBandit(num_arms=5, num_features=10)
    weights = bandit.initialize_weights(10)
    assert len(weights) == 10
    assert all(0 <= w <= 1 for w in weights)

def test_choose_arm():
    bandit = ContextualBandit(num_arms=5, num_features=10)
    context = [random.random() for _ in range(10)]
    chosen_arm = bandit.choose_arm(context)
    assert 0 <= chosen_arm < 5

def test_update_weights():
    bandit = ContextualBandit(num_arms=5, num_features=10)
    context = [random.random() for _ in range(10)]
    chosen_arm = 0
    initial_weights = bandit.weights[chosen_arm][:]
    reward = 1.0
    bandit.update_weights(chosen_arm, context, reward)
    updated_weights = bandit.weights[chosen_arm]
    assert initial_weights != updated_weights

def test_offline_policy_evaluation():
    bandit = ContextualBandit(num_arms=5, num_features=10)
    historical_data = [
        ([random.random() for _ in range(10)], 0, 1.0),
        ([random.random() for _ in range(10)], 1, 0.5),
        ([random.random() for _ in range(10)], 2, 0.0),
        ([random.random() for _ in range(10)], 3, 1.0),
        ([random.random() for _ in range(10)], 4, 0.5)
    ]
    average_reward = bandit.offline_policy_evaluation(historical_data)
    assert 0 <= average_reward <= 1

def test_run():
    bandit = ContextualBandit(num_arms=5, num_features=10)

    def get_current_context() -> List[float]:
        return [random.random() for _ in range(10)]

    def get_reward(chosen_arm: int) -> float:
        return random.random()

    initial_weights = [list(arm_weights) for arm_weights in bandit.weights]
    bandit.run(T=10, get_current_context=get_current_context, get_reward=get_reward)
    updated_weights = bandit.weights
    assert initial_weights != updated_weights

if __name__ == "__main__":
    pytest.main()
