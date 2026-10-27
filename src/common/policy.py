"""The agent's network: per-ward encoder with pooled context, and the critic.
"""

from typing import Callable, Tuple, Dict as TypingDict
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int, graph_filepath: str = None):
        super().__init__(observation_space, features_dim)

        self.num_nodes = observation_space.shape[0]
        node_dim = observation_space.shape[1]
        self.node_dim = node_dim

        hidden = 128

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, node_dim = observations.shape

        x = observations.reshape(batch_size * num_nodes, node_dim)
        h = self.node_encoder(x).reshape(batch_size, num_nodes, -1)

        ctx_mean = h.mean(dim=1, keepdim=True).expand(-1, num_nodes, -1)
        ctx_max = h.max(dim=1, keepdim=True).values.expand(-1, num_nodes, -1)

        combined = torch.cat([h, ctx_mean, ctx_max], dim=-1)

        out = self.combine(combined.reshape(batch_size * num_nodes, -1))
        return out.reshape(batch_size, num_nodes, self.features_dim)


class GNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, actions_per_node: int = None, **kwargs):
        num_nodes = observation_space.shape[0]
        derived = int(action_space.shape[0] // num_nodes)
        self.actions_per_node = actions_per_node if actions_per_node is not None else derived
        self.gnn_output_dim = kwargs["features_extractor_kwargs"]["features_dim"]
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, **kwargs)
        
    def _build_mlp_extractor(self) -> None:
        class _GNNMlpExtractor(nn.Module):
            def __init__(self, feature_dim: int):
                super().__init__()
                self.latent_dim_pi = feature_dim
                self.latent_dim_vf = feature_dim
                self.policy_net = nn.Identity()
                self.value_net = nn.Identity()

            def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                return self.policy_net(features), self.value_net(features)

            def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
                return self.policy_net(features)

            def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
                return self.value_net(features)

        self.mlp_extractor = _GNNMlpExtractor(self.gnn_output_dim)
        
    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)

        pi_dim = self.mlp_extractor.latent_dim_pi
        self.action_net = nn.Sequential(
            nn.Linear(pi_dim, pi_dim),
            nn.ReLU(),
            nn.Linear(pi_dim, self.actions_per_node),
        )

        vf_dim = self.mlp_extractor.latent_dim_vf
        self.value_net = nn.Sequential(
            nn.Linear(vf_dim * 2, vf_dim),
            nn.ReLU(),
            nn.Linear(vf_dim, vf_dim // 2),
            nn.ReLU(),
            nn.Linear(vf_dim // 2, 1),
        )

        if hasattr(self, 'log_std'):
            del self.log_std 

        self.global_log_std = nn.Parameter(torch.zeros(self.actions_per_node), requires_grad=True)

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1.0),
            **self.optimizer_kwargs,
        )
        
    def _get_masked_mean(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([features.mean(dim=1), features.max(dim=1).values], dim=-1)
        
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> DiagGaussianDistribution:
        mean_actions_per_node = self.action_net(latent_pi)
        flat_mean_actions = mean_actions_per_node.reshape(mean_actions_per_node.shape[0], -1)
        
        num_nodes = mean_actions_per_node.shape[1]
        expanded_log_std = self.global_log_std.repeat(num_nodes)
        
        return self.action_dist.proba_distribution(flat_mean_actions, expanded_log_std)
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        aggregated_vf = self._get_masked_mean(latent_vf)
        values = self.value_net(aggregated_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
        
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        aggregated_vf = self._get_masked_mean(latent_vf)
        values = self.value_net(aggregated_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy
        
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        aggregated_vf = self._get_masked_mean(latent_vf)
        return self.value_net(aggregated_vf)