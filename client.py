"""Proximal Policy Optimization (PPO) Clipped Surrogate Engine.
100% Python Standard Library.
"""

class PPOClippedSurrogate:
    """Proximal Policy Optimization (PPO) clipped surrogate objective evaluator."""
    def __init__(self, clip_epsilon=0.2):
        self.epsilon = clip_epsilon

    def compute_surrogate_loss(self, old_probs, new_probs, advantages):
        losses = []
        clip_fractions = 0
        n = len(old_probs)

        for pi_old, pi_new, adv in zip(old_probs, new_probs, advantages):
            ratio = (pi_new / (pi_old + 1e-12))
            surr1 = ratio * adv
            clipped_ratio = max(1.0 - self.epsilon, min(1.0 + self.epsilon, ratio))
            surr2 = clipped_ratio * adv
            loss = min(surr1, surr2)
            losses.append(loss)
            if abs(ratio - 1.0) > self.epsilon:
                clip_fractions += 1

        mean_loss = sum(losses) / (n or 1)
        return {
            "mean_surrogate_objective": mean_loss,
            "clip_fraction": clip_fractions / (n or 1),
            "num_samples": n
        }
