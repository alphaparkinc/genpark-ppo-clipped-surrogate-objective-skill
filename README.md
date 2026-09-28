# genpark-ppo-clipped-surrogate-objective-skill

Agent Skill implementing the **Proximal Policy Optimization (PPO) Clipped Surrogate Objective**, ensuring bounded policy parameter updates in RL and RLHF pipelines.

## Architectural Overview
```mermaid
flowchart TD
    OldPolicy["Old Policy Probabilities pi_old"] & NewPolicy["New Policy Probabilities pi_new"] --> Ratio["Probability Ratio r_t(theta) = pi_new / pi_old"]
    Ratio & Adv["Advantage Estimates A_t"] --> Unclipped["Unclipped Term: r_t * A_t"]
    Ratio --> Clip["Clip to [1 - eps, 1 + eps]"]
    Clip & Adv --> Clipped["Clipped Term: clip(r_t) * A_t"]
    Unclipped & Clipped --> Min["Min Bound: min(Unclipped, Clipped)"]
    Min --> Obj["PPO Surrogate Objective L^{CLIP}"]
```
