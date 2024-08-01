import torch
from random import shuffle
import numpy as np

METHODS = {
    'agr-sum': 'agreement_sum',
    'agr-rand': 'agreement_rand',
    'pcgrad': 'pcgrad',
    'corgrad': 'correct_grad'
}


def get_method(method):
    if method in METHODS.keys():
        return globals()[METHODS[method]]
    else:
        raise ValueError


def agreement_sum(domain_grads, n_parts, alpha):
    """ Agr-Sum consensus strategy."""

    # Compute agreement mask
    agr_mask = agreement_mask(domain_grads)

    # Sum the components that have the same sign and zero those that do not
    new_grads = torch.stack(domain_grads).sum(0)
    new_grads *= agr_mask

    ###new
    change_rate = agr_mask.sum(0) / len(agr_mask)

    return new_grads, change_rate


def agreement_rand(domain_grads):
    """ Agr-Rand consensus strategy. """

    # Compute agreement mask
    agr_mask = agreement_mask(domain_grads)

    # Sum components with same sign
    new_grads = torch.stack(domain_grads).sum(0)
    new_grads *= agr_mask

    # Get sample for components that do not agree
    sample = torch.randn((~agr_mask).sum(), device=new_grads.device)
    scale = new_grads[agr_mask].abs().mean()
    # scale = new_grads.abs().mean()
    sample *= scale

    # Assign values to these components
    new_grads[~agr_mask] = sample

    return new_grads


def agreement_mask(domain_grads):
    """ Agreement mask. """

    grad_sign = torch.stack([torch.sign(g) for g in domain_grads])
    # True if all componentes agree, False if not
    agr_mask = torch.where(grad_sign.sum(0).abs() == len(domain_grads), 1, 0)
    change_rate = agr_mask.sum(0) / len(agr_mask)

    return agr_mask.bool()


def pcgrad(domain_grads, n_parts, alpha):
    """ Projecting conflicting gradients (PCGrad). """

    #new
    conflict_num = 0
    total_num = 0
    change_percentage = []

    task_order = list(range(len(domain_grads)))

    # Run tasks in random order
    shuffle(task_order)

    # Initialize task gradients
    grad_pc = [g.clone() for g in domain_grads]

    for i in task_order:

        # Run other tasks
        other_tasks = [j for j in task_order if j != i]

        for j in other_tasks:
            grad_j = domain_grads[j]

            # Compute inner product and check for conflicting gradients
            inner_prod = torch.dot(grad_pc[i], grad_j)
            
            total_num += 1

            if inner_prod < 0:
                # Sustract conflicting component
                tmp_big = grad_pc[i].view(-1).norm(2).item()
                grad_pc[i] -= inner_prod / (grad_j ** 2).sum() * grad_j
                tmp_small = (inner_prod / (grad_j ** 2).sum() * grad_j).view(-1).norm(2).item()
                sin_theta = tmp_small / tmp_big
                change_percentage.append(sin_theta)
                conflict_num += 1

    # Sum task gradients
    new_grads = torch.stack(grad_pc).sum(0)

    #new
    change_rate = conflict_num / total_num

    if len(change_percentage) != 0: 
        change_percentage_mean = sum(change_percentage)/len(change_percentage)
    else:
        change_percentage_mean = 0

    return new_grads, change_rate, change_percentage_mean

# 冲突+决定
def correct_grad(domain_grads, n_parts, alpha):
    ##交换domain_grads中的顺序，swap_domain_grads是一个双层列表，第一层是卷积层参数共有多少个卷积层，第二层是该层参数中包含了来自多个域的参数
    swap_domain_grads = [[row[i] for row in domain_grads] for i in range(len(domain_grads[0]))]
    conv_grads = []
    new_grads = []
    total_num = 0
    conflict_num = 0
    change_percentage = []

    for layer_index, layer_grads in enumerate(swap_domain_grads):
        conv_grads.append(layer_grads)
        domain_grad_chunks = []

        for domain_idx, domain_grad in enumerate(conv_grads[layer_index]):
            domain_grad_chunks.append(torch.chunk(domain_grad, n_parts))

        big_grads = []

        for i in range(len(domain_grad_chunks)):
            other_parts = [k for k in range(len(domain_grad_chunks)) if k != i]
            small_grads = []

            for j in range(len(domain_grad_chunks[i])):
                grad_i = domain_grad_chunks[i][j]
                
                for p in other_parts:
                    grad_j = domain_grad_chunks[p][j]
                    # 计算内积
                    if len(grad_i) == len(grad_j):
                        inner_prod = torch.dot(grad_i.flatten(), grad_j.flatten())
                        total_num += 1
                
                    # 如果内积小于0,说明存在冲突梯度，需要进行投影
                        if inner_prod < 0:
                            tmp_big = grad_i.view(-1).norm(2).item()
                            grad_i -= inner_prod / (grad_j ** 2).sum() * grad_j 
                            tmp_small = (inner_prod / (grad_j ** 2).sum() * grad_j).view(-1).norm(2).item()
                            sin_theta = tmp_small / tmp_big
                            change_percentage.append(sin_theta)
                            conflict_num += 1
                
                # small_grads存储分了n_part的grad
                small_grads.append(grad_i)

            #big_grads包含了一层三个源域的梯度
            big_grads.append(torch.cat(small_grads))

        norm_list = [big_grad.view(-1).norm(2).item() for big_grad in big_grads]
        mean_norm = np.mean(norm_list)
        std_norm = np.std(norm_list)
        outliers_index = [index for index, (big_grad, norm) in enumerate(zip(big_grads, norm_list)) if abs(norm - mean_norm) > 1 * std_norm ]
        except_big_grads = [big_grads[i] for i in range(len(big_grads)) if i not in outliers_index]
        mean_tensor = torch.mean(torch.stack(except_big_grads), dim = 0)
        
        for index in outliers_index:
            big_grads[index] = alpha * big_grads[index] + (1-alpha) * mean_tensor    

        new_grads.append(big_grads)
        change_rate = conflict_num / total_num
        if len(change_percentage) != 0: 
            change_percentage_mean = sum(change_percentage)/len(change_percentage)
        else:
            change_percentage_mean = 0

    return new_grads, change_rate, change_percentage_mean
