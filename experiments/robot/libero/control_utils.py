import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple

def calculate_uncertainty_metrics(
    logits: List[torch.Tensor],  # List of tensors, each shape: (1, vocab_size)
    metrics: Optional[List[str]] = None,  # List of metric names to compute
    actual_tokens: Optional[List[int]] = None,  # List[int], length = len(logits)
    vocab_size: int = 32000,  # int, vocabulary size
    top_k: int = 10,  # int, k value for top-k probability mass
    return_sequence_level: bool = True,  # bool, whether to return sequence-level metrics
    return_token_level: bool = True  # bool, whether to return token-level metrics
) -> Dict[str, Union[List[float], float]]:
    """
    根据logits计算生成不确定性指标
    
    Args:
        logits: List of tensors, 每个tensor形状为(1, vocab_size)
        metrics: 要计算的指标列表，如果为None则计算所有指标
        actual_tokens: 实际生成的token ID列表，用于计算困惑度
        vocab_size: 词汇表大小
        top_k: Top-k概率质量计算中的k值
        return_sequence_level: 是否返回序列级别的汇总指标
        return_token_level: 是否返回token级别的指标
    
    Returns:
        Dict[str, Union[List[float], float]]: 包含各种不确定性指标的字典
        
        Token-level metrics (when return_token_level=True):
            'token_entropy': List[float]  # length = len(logits), 每个token的熵值
            'normalized_entropy': List[float]  # length = len(logits), 归一化熵值 [0,1]
            'max_probability': List[float]  # length = len(logits), 最大概率值 [0,1]
            'top_k_probability_mass': List[float]  # length = len(logits), top-k概率质量 [0,1]
            'probability_variance': List[float]  # length = len(logits), 概率分布方差
            'predictive_confidence': List[float]  # length = len(logits), 预测置信度
            'token_perplexity': List[float]  # length = len(logits), token困惑度
        
        Sequence-level metrics (when return_sequence_level=True):
            '{metric}_mean': float  # 对应token级指标的均值
            '{metric}_std': float   # 对应token级指标的标准差
            '{metric}_min': float   # 对应token级指标的最小值
            '{metric}_max': float   # 对应token级指标的最大值
            'entropy_variance': float  # 熵的方差，衡量不确定性的不确定性
            'min_confidence': float    # 最小置信度值
    """
    
    # 可用的指标列表
    available_metrics = [
        'token_entropy',           # 每个位置的熵值
        'normalized_entropy',      # 归一化的熵值
        'max_probability',         # 最大概率
        'top_k_probability_mass',  # top-k概率质量
        'probability_variance',    # 概率分布方差
        'predictive_confidence',   # 预测置信度
        'token_perplexity'        # token困惑度
    ]
    
    # 如果未指定metrics，则计算所有指标
    if metrics is None:
        metrics = available_metrics
    
    # 验证指标名称
    for metric in metrics:
        if metric not in available_metrics:
            raise ValueError(f"未知指标: {metric}. 可用指标: {available_metrics}")
    
    results: Dict[str, Union[List[float], float]] = {}
    
    # Token级别的指标计算
    if return_token_level:
        
        if 'token_entropy' in metrics:
            # Type: List[float], Shape: [len(logits)]
            # 值域: [0, log(vocab_size)], 越大表示不确定性越高
            entropies = []
            for logit in logits:
                probs = F.softmax(logit, dim=-1)  # shape: (1, vocab_size)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # shape: (1,)
                entropies.append(entropy.item())  # float
            results['token_entropy'] = entropies
        
        if 'normalized_entropy' in metrics:
            # Type: List[float], Shape: [len(logits)]
            # 值域: [0, 1], 归一化的熵值，1表示完全不确定
            if 'token_entropy' not in results:
                # 先计算token_entropy
                entropies = []
                for logit in logits:
                    probs = F.softmax(logit, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                    entropies.append(entropy.item())
            else:
                entropies = results['token_entropy']
            
            max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float)).item()  # float
            results['normalized_entropy'] = [h / max_entropy for h in entropies]
        
        if 'max_probability' in metrics:
            # Type: List[float], Shape: [len(logits)]
            # 值域: [0, 1], 最高概率token的概率，越大表示确定性越高
            max_probs = []
            for logit in logits:
                probs = F.softmax(logit, dim=-1)  # shape: (1, vocab_size)
                max_prob = torch.max(probs, dim=-1)[0]  # shape: (1,)
                max_probs.append(max_prob.item())  # float
            results['max_probability'] = max_probs
        
        if 'top_k_probability_mass' in metrics:
            # Type: List[float], Shape: [len(logits)]
            # 值域: [0, 1], 前k个最高概率token的概率总和
            top_k_masses = []
            for logit in logits:
                probs = F.softmax(logit, dim=-1)  # shape: (1, vocab_size)
                top_k_probs = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)[0]  # shape: (1, k)
                mass = torch.sum(top_k_probs, dim=-1)  # shape: (1,)
                top_k_masses.append(mass.item())  # float
            results['top_k_probability_mass'] = top_k_masses
        
        if 'probability_variance' in metrics:
            # Type: List[float], Shape: [len(logits)]
            # 值域: [0, +inf), 概率分布的方差，越大表示分布越分散
            variances = []
            for logit in logits:
                probs = F.softmax(logit, dim=-1)  # shape: (1, vocab_size)
                variance = torch.var(probs, dim=-1)  # shape: (1,)
                variances.append(variance.item())  # float
            results['probability_variance'] = variances
        
        if 'predictive_confidence' in metrics:
            # Type: List[float], Shape: [len(logits)]
            # 值域: [-1, 1], 基于Gini系数的置信度，越大表示置信度越高
            confidences = []
            for logit in logits:
                probs = F.softmax(logit, dim=-1)  # shape: (1, vocab_size)
                sorted_probs = torch.sort(probs, descending=True, dim=-1)[0]  # shape: (1, vocab_size)
                n = sorted_probs.size(-1)  # int
                indices = torch.arange(1, n + 1, dtype=torch.float, device=sorted_probs.device)  # shape: (vocab_size,)
                gini_coeff = torch.sum((2 * indices - n - 1) * sorted_probs) / (n * torch.sum(sorted_probs))  # shape: ()
                confidences.append(gini_coeff.item())  # float
            results['predictive_confidence'] = confidences
        
        if 'token_perplexity' in metrics:
            # Type: List[float], Shape: [len(logits)] or [len(actual_tokens)]
            # 值域: [1, +inf), token级困惑度，越大表示模型越不确定
            if actual_tokens is None:
                # 使用argmax作为"实际"token
                actual_tokens = [torch.argmax(logit, dim=-1).item() for logit in logits]  # List[int]
            
            perplexities = []
            for i, logit in enumerate(logits):
                probs = F.softmax(logit, dim=-1)  # shape: (1, vocab_size)
                if i < len(actual_tokens):
                    token_prob = probs[0, actual_tokens[i]]  # shape: ()
                    perplexity = 1 / (token_prob + 1e-8)  # 避免除零, shape: ()
                    perplexities.append(perplexity.item())  # float
            results['token_perplexity'] = perplexities
    
    # 序列级别的指标计算
    if return_sequence_level:
        
        # 计算各token级别指标的序列级汇总
        for metric in metrics:
            if metric in results:
                values = torch.tensor(results[metric])  # shape: (len(logits),)
                # Type: float, 对应指标在整个序列上的统计量
                results[f'{metric}_mean'] = values.mean().item()  # float: 均值
                results[f'{metric}_std'] = values.std().item()    # float: 标准差
                results[f'{metric}_min'] = values.min().item()    # float: 最小值
                results[f'{metric}_max'] = values.max().item()    # float: 最大值
        
        # 特殊的序列级指标
        if any(m in metrics for m in ['token_entropy', 'normalized_entropy']):
            # Type: float, 熵的方差（不确定性的不确定性）
            # 值域: [0, +inf), 衡量整个序列不确定性的变化程度
            if 'token_entropy' in results:
                entropy_tensor = torch.tensor(results['token_entropy'])  # shape: (len(logits),)
                results['entropy_variance'] = entropy_tensor.var().item()  # float
        
        if 'max_probability' in metrics and 'max_probability' in results:
            # Type: float, 最小置信度（最大不确定性位置）
            # 值域: [0, 1], 序列中置信度最低的token的置信度
            results['min_confidence'] = min(results['max_probability'])  # float
    
    return results


def entropy_units(logits: Tuple[torch.Tensor]) -> List[float]:
    """
    计算每8个logits为一组的序列entropy
    
    Args:
        logits: Tuple[torch.Tensor], 每个tensor形状为(1, vocab_size)
        num_act_units: int, 动作单元数量
        
    Returns:
        List[float]: 每8个logits一组的entropy值列表
    """
    # 确保logits长度可以被8整除
    assert len(logits) % 8 == 0, f"Logits length {len(logits)} must be divisible by 8"
    
    # 计算组数
    num_groups = len(logits) // 8
    
    # 存储每组的entropy
    group_entropies = []
    
    # 按8个一组处理logits
    for i in range(num_groups):
        group_logits = logits[i*8:(i+1)*8]  # 取出8个logits
        
        # 存储组内每个token的entropy
        token_entropies = []
        
        # 对每个logit计算entropy
        for logit in group_logits[-2]:
            # 计算概率分布
            probs = F.softmax(logit, dim=-1)  # shape: (1, vocab_size)
            
            # 计算单个token的entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
            token_entropies.append(entropy.item())
        
        # 计算组内平均entropy
        group_entropy = sum(token_entropies) / len(token_entropies)
        group_entropy = round(group_entropy, 4)
        group_entropies.append(group_entropy)
        
    return group_entropies

# 使用示例和类型说明
def example_usage():
    """展示函数的使用方法和返回值类型"""
    
    # 模拟logits数据
    # logits: List[torch.Tensor], length=5, each tensor shape=(1, 32000)
    logits = [torch.randn(1, 32000) for _ in range(5)]
    
    # 计算所有指标
    all_metrics = calculate_uncertainty_metrics(logits)
    # 返回类型: Dict[str, Union[List[float], float]]
    print("所有指标keys:", list(all_metrics.keys()))
    print("token_entropy type and length:", type(all_metrics['token_entropy']), len(all_metrics['token_entropy']))
    print("token_entropy_mean type:", type(all_metrics['token_entropy_mean']))
    
    # 只计算特定指标
    specific_metrics = calculate_uncertainty_metrics(
        logits, 
        metrics=['token_entropy'],  # List[str]
        return_sequence_level=True,  # bool
        return_token_level=True      # bool
    )
    # 返回: Dict[str, Union[List[float], float]]
    # - 'token_entropy': List[float], length=5
    # - 'max_probability': List[float], length=5  
    # - 'token_entropy_mean': float
    # - 'token_entropy_std': float
    # - 'max_probability_mean': float
    # - 'max_probability_std': float
    # - 等等...
    
    print("Token级token_entropy:", specific_metrics['token_entropy'])  # List[float]
    print("序列级token_entropy均值:", specific_metrics['token_entropy_mean'])  # float

    print(specific_metrics)

if __name__ == "__main__":
    example_usage()