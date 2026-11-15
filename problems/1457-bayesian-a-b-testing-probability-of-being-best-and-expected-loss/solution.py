from math import lgamma, exp, log

def log_beta(x, y):
    return lgamma(x) + lgamma(y) - lgamma(x + y)

def posterior_params(successes, trials, alpha_prior=1.0, beta_prior=1.0):
    return (float(alpha_prior + successes), float(beta_prior + trials - successes))

def prob_b_beats_a(a_A, b_A, a_B, b_B):
    total = 0.0
    lb_A = log_beta(a_A, b_A)
    
    for i in range(int(a_B)):
        log_term = (
            log_beta(a_A + i, b_A + b_B)
            - log(b_B + i)
            - log_beta(1 + i, b_B)
            - lb_A
        )
        total += exp(log_term)
        
    return float(total)

def expected_loss(a_A, b_A, a_B, b_B):
    mean_A = a_A / (a_A + b_A)
    mean_B = a_B / (a_B + b_B)
    
    h_A_plus = prob_b_beats_a(a_A + 1, b_A, a_B, b_B)
    h_B_plus = prob_b_beats_a(a_A, b_A, a_B + 1, b_B)
    
    loss_A = mean_B * h_B_plus - mean_A * h_A_plus
    loss_B = mean_A * (1.0 - h_A_plus) - mean_B * (1.0 - h_B_plus)
    
    return (float(loss_A), float(loss_B))

def decide(a_A, b_A, a_B, b_B, threshold):
    loss_A, loss_B = expected_loss(a_A, b_A, a_B, b_B)
    
    if loss_A < loss_B and loss_A < threshold:
        return 'A'
    elif loss_B <= loss_A and loss_B < threshold:
        return 'B'
    else:
        return 'continue'