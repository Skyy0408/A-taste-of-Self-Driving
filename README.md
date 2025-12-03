# Robust Autonomous Driving


## I. AI Future Vision: Safe and Generalized Autonomous Decision-Making

**AI Capability: Context-Aware, Zero-Shot Generalization in Safety-Critical Scenarios**

The major limitation of current autonomous driving (AD) systems is their dependence on large, static datasets and their vulnerability to *distribution shift*. Rare but safety-critical events—such as a pedestrian or scooter suddenly cutting in—may push the model far outside its training distribution, causing unexpected and potentially dangerous behaviors (for example, steering off the road to avoid a collision and crashing into the guardrail instead).

Our future vision (20 years) is an AI that achieves *zero-shot generalization* in driving: the policy can safely and optimally navigate any unprecedented, high-stakes traffic situation (e.g., unexpected object movement, severe weather, chaotic construction zones) without requiring specific retraining or human oversight for that exact scenario. This capability ensures the vehicle's behavior is always *interpretable, verifiable, and constrained by physical laws and safety margins*, making AD systems trustworthy and scalable worldwide.

This vision is critical because it addresses the core societal bottleneck: the ability of AD systems to reliably handle "unknown unknowns" that lead to catastrophic failures, thus unlocking mass adoption.



## II. Ingredients for Realizing the Vision

To move from current data-driven approaches toward this highly generalized and safe AI, several essential ingredients are required.

Here's the markdown table:

| Ingredient | Role in the Future System |
| :--- | :--- |
| **Data: Multi-Modal & Temporal-Relational Graphs** | Current data is rasterized and localized. The future system requires large-scale, high-fidelity data that encodes *temporal relationships* between agents and environmental features. This necessitates graph-based representations of traffic flow, predicting intent rather than just position. |
| **Tools: Differentiable Physics and Model Predictive Control (MPC)** | The policy must be grounded in physics to ensure safety. This requires a differentiable *ego model* (like the Kinematic Bicycle Model detailed below) combined with a *differentiable cost function* (via cost masks) within an MPC framework. This enables end-to-end gradient-based optimization for planning that adheres to physical and safety constraints. |
| **Learning Setup: Self-Supervised Uncertainty Regularization** | To handle distribution shift, the policy must actively identify where its predictions are unreliable. This requires integrating a *stochastic world model* (trained via self-supervision on massive unlabeled data) and using *uncertainty regularization* to ensure the policy only acts confidently, or yields control when confidence is low. |
| **Hardware/Environment: High-Fidelity Simulator and Low-Latency Edge Compute** | Safe exploration is prohibitively expensive in the real world. Realizing this vision requires high-fidelity, randomized simulation environments for "trial-and-error," coupled with edge computing devices capable of executing complex MPC optimizations at low latency for real-time decision-making. |



## III. Machine Learning Type Analysis
The realization of the vision requires a Hybrid Learning Architecture combining *Supervised Learning (SL)*, *Unsupervised Learning (UL)*, and *Reinforcement Learning (RL)*.

Here's the markdown table:

| Learning Paradigm | Application | Details |
| :--- | :--- | :--- |
| **Supervised/Unsupervised Learning** | *World Model Training*: A deep neural network ($f_\theta^{\text{env}}$) must learn to predict the future state of the environment (lanes, other cars) given the current sensor input. This training is typically done via SL (predicting ground truth next frames) or UL (e.g., predicting occluded pixels or future state representations). | *Data Source:* Raw sensor data (Lidar, Camera, Maps).<br> *Target Signal:* The subsequent ground-truth sensor reading or feature representation. |
| **Reinforcement Learning** | *Policy Optimization*: The core task is finding an optimal sequence of actions ($a_t^{\text{self}}$) that minimizes a long-term cost function. This intrinsically involves learning from feedback (the cost function $C$) and interacting with a predicted environment over a horizon $T$, which is the definition of a model-based RL/MPC setup. | *Data Source:* The predicted state trajectory from the world model.<br> *Target Signal:* The cumulative cost function $J_t$, which provides the learning feedback signal. |

Existence of Feedback/Interaction: Yes. The policy interacts with the predicted environment model $f_\theta^{\text{env}}$ and the physics-based ego model $f^{\text{self}}$. The feedback signal is the cost $C$ (combining safety, progress, and smoothness), which the policy minimizes over a receding horizon.



## IV. Solvable Model Problem: DFM-KM MPC with Uncertainty Regularization

To take the first step toward the capability of zero-shot generalization under safety constraints, we design and implement a simplified model problem: *Differentiable Fixed-Horizon Model-Predictive Control with Kinematic Modeling (DFM-KM MPC) and Uncertainty Regularization*.


### A. Problem Design

**Goal:** Train a policy to safely navigate dense traffic by optimizing a future action sequence, where safety constraints are encoded via differentiable cost masks and robustness is enforced by penalizing model uncertainty.

| Component | Definition |
| :--- | :--- |
| Input | Current ego state $s_t$ (position, speed, heading) and a rasterized environment state (lanes, cars, off-road). |
| Output | An optimal sequence of ego actions $a_{t:t+T-1}^{\text{self}}$ over a finite horizon $T$. |
| Task Objective | Minimize the cumulative cost $J_t$, which is a weighted sum of interpretable physical, safety, and uncertainty terms. |
| Data Form | Simulated data or recorded trajectories are used to train the stochastic environment model $f_\theta^{\text{env}}$. The policy is optimized against the cost function. |


### B. Model and Method

#### 1. Kinematic Bicycle Ego Model

The ego dynamics $f^{\text{self}}$ are defined by a simple, differentiable kinematic bicycle model. Given control input $a_t^{\text{self}} = (a_{t,0}, a_{t,1})$ (longitudinal acceleration and turning command), the state updates over $\Delta t$ are:

$$
\begin{aligned}
x_{t+1} &= x_t + s_t u_t^x \Delta t, \\
y_{t+1} &= y_t + s_t u_t^y \Delta t, \\
s_{t+1} &= s_t + a_{t,0} \Delta t, \\
(u_{t+1}^x, u_{t+1}^y) &= \text{unit}\big[(u_t^x, u_t^y) + a_{t,1} \Delta t \,(u_t^y, -u_t^x)\big],
\end{aligned}
$$

This physics-based model ensures the ego dynamics are both interpretable and fully differentiable, propagating gradients only through the ego model during optimization.

#### 2. Differentiable Cost Mask Construction

Safety is encoded using pre-defined, differentiable cost masks $M^{\text{car}}$ (for proximity) and $M^{\text{side}}$ (for lanes/off-road) that align with the ego vehicle’s predicted pose and orientation. The safety margins ($d_x, d_y$) are defined based on vehicle speed ($s$), length ($l$), and width ($w$):

$$
d_x = 1.5 \cdot (\max(10, s) + l) + 1, \quad d_y = \frac{w}{2} + 3.7
$$

The trajectory cost $C$ is computed as a weighted sum of interpretable components:

$$
C = \alpha_{\text{lane}} C_{\text{lane}} + \alpha_{\text{offroad}} C_{\text{offroad}} + \alpha_{\text{proximity}} C_{\text{proximity}} + \alpha_{\text{destination}} C_{\text{destination}} + \alpha_{\text{jerk}} C_{\text{jerk}}.
$$

The safety terms ($C_{\text{lane}}, C_{\text{offroad}}, C_{\text{proximity}}$) are defined as pointwise inner products between the environment channels predicted by $f_\theta^{\text{env}}$ and the fixed, differentiable cost masks.

#### 3. Quantifying Uncertainty

To enhance robustness against distribution shift, a stochastic world model is trained, and dropout is applied during the forward pass ($z \sim \mathcal{B}(p_u)$) to yield an ensemble of predicted next states. The *model uncertainty cost* $U$ is defined as the sum of variances over these predictions:

$$
U(\hat{s}_{t+1})=\text{tr}[\text{Cov}[\{f_{\theta_k}(s_{1:t}, a_t,z_t)\}_{k=1}^K]]=\displaystyle\sum_{j=1}^d\text{Var}(\{f_{\theta_k}(s_{1:t},a_t,z_t)_j\}_{k=1}^K)
$$

The policy is then trained to minimize the total cost: $C_{\text{total}} = C_{\text{policy}} + \lambda U$. Minimizing $C_{\text{total}}$ encourages the policy to choose actions that achieve good driving behavior while also keeping the system within regions where the model is confident.


### C. Implementation and Results

In the DFM-KM MPC setting (as shown in Figure 1d of Sobal et al.), the optimization loop is:

1. At time $t$, optimize the action sequence $a_{t:t+T-1}^{\text{self}}$ to minimize the cumulative cost $J_t$.
2. The ego state is propagated purely by the kinematic model $f^{\text{self}}$.
3. The environment model $f_\theta^{\text{env}}$ predicts the surroundings.
4. Only the *first* action $a_t^{\text{self}}$ of the optimal sequence is executed (Receding-horizon control).

Empirical results from the foundational literature show that using longer rollouts and incorporating the uncertainty cost leads to a clear improvement in both mean travelled distance and success rate compared to baseline methods, particularly in dense and complex environments.

![Figure 6 from Model Predictive Policy Learning with Uncertaincy Regularization for Driving in Dense Traffic](Figures_of_README/Figure_6_from_Model_Predictive_Policy_Learning_with_Uncertainty_Regularization_for_Driving_in_Dense_Traffic.png)


### D. Discussion

#### Lessons Learned
The implementation of the DFM-KM MPC framework reveals the power of *separating learned and physical dynamics*. By confining the physics (ego dynamics) to a simple, differentiable kinematic model and assigning the complexity (predicting other agents) to a learned environment model, the system gains both interpretability and stability. The optimization is physically grounded, which is a major step toward safety verification.

#### Key Difficulty Revealed
This simplified model highlights the core difficulty of the larger problem: *Quantifying and reacting to Model Uncertainty in Real-Time*. While the uncertainty regularization term pushes the agent toward known territory, it does not provide a mechanism for generating a safe novel action when faced with a true unknown (a zero-shot scenario). The 20-year vision requires the AI not just to be confident, but to generate novel, safe responses, which moves beyond current stochastic models and into advanced symbolic reasoning and meta-learning techniques. The current DFM-KM MPC structure is a robust foundation, but the prediction horizon $T$ is still limited, meaning long-term safety is not guaranteed.

### E. Run DFM-KM MPC
According to this [repo](https://github.com/vladisai/pytorch-PPUU/tree/ICLR2022), by fixing some version incompatibilities, the DFM-KM MPC can be executed successfully. The detail is leave in Reproduction Guide. Its operating principle is to take control of one vehicle in a given frame and use the model to compute whether it can keep driving until the end of the time horizon (or any stopping point you define).
<video width="640" height="360" controls>
  <source src="Figures_of_README/0.mp4" type="video/mp4">
  你的瀏覽器不支援影片播放。
</video>
<video width="640" height="360" controls>
  <source src="Figures_of_README/1.mp4" type="video/mp4">
  你的瀏覽器不支援影片播放。
</video>

## References
* Henaff, P., Henaff, G., Dhiman, S., Chen, Y., Bogoian, B., Kim, E., & Liu, H. (2021). Model-Predictive Policy Learning with Uncertainty Regularization for Driving in Dense Traffic. Proceedings of the 9th International Conference on Learning Representations (ICLR). Retrieved from https://openreview.net/forum?id=HygQBn0cYm
* Sobal, V., Sanyal, S., Dhiman, S., Chen, Y., Liu, H., Bogoian, B., & Henaff, P. (2022). Separating the World and Ego Models for Self-Driving. arXiv preprint arXiv:2204.07184. Retrieved from https://arxiv.org/abs/2204.07184