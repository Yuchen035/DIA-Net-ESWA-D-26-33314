"""
损失函数模块（完全解耦）

包含：
- flow_matching: Flow Matching Loss
- load_balance: 负载均衡损失（MoE-LoRA）
- adversarial: 对抗损失计算（待实现）
- discriminator: 判别器网络（待实现）
- reconstruction: 重建损失（可选，待实现）
- color: 颜色余弦损失（可选，待实现）
"""

from .flow_matching import FlowMatchingLoss, compute_flow_matching_loss
from .load_balance import LoadBalanceLoss, AuxiliaryLoadBalanceLoss, compute_expert_utilization

# from .adversarial import AdversarialLoss  # TODO: 待实现
# from .discriminator import SingleRouteDiscriminator, DualRouteDiscriminator  # TODO: 待实现

__all__ = [
    'FlowMatchingLoss',
    'compute_flow_matching_loss',
    'LoadBalanceLoss',
    'AuxiliaryLoadBalanceLoss',
    'compute_expert_utilization',
    # 'AdversarialLoss',  # TODO: 待实现
    # 'SingleRouteDiscriminator',  # TODO: 待实现
    # 'DualRouteDiscriminator',  # TODO: 待实现
]
