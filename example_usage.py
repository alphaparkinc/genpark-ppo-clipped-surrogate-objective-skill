from client import PPOClippedSurrogate

ppo = PPOClippedSurrogate(clip_epsilon=0.2)
old_p = [0.5, 0.5, 0.5]
new_p = [0.8, 0.4, 0.5]
adv = [1.0, -1.0, 0.5]

res = ppo.compute_surrogate_loss(old_p, new_p, adv)
print(f"PPO Clipped Objective: {res['mean_surrogate_objective']:.4f}")
print(f"Clip Fraction: {res['clip_fraction']:.2f}")
