import numpy as np
from scipy.interpolate import LSQUnivariateSpline

def compute_knots(n):
    return max(1, n // 40)

def log_likelihood(m_hat, u_hat, n, RSS_hat):
   
    term_1 = np.log(m_hat)
    log_values = []
    for u in u_hat[:m_hat]:
        if u <= 0:
            log_values.append(-1e6)  
        else:
            log_values.append(np.log(u))
    term_2 = np.sum(log_values)
    term_4 = (m_hat+4) /0.5 * np.log(n)
    
    term_5 = 0.5 * n * np.log(RSS_hat / n)
    
    log_likelihood_value = term_1 + term_2 + term_4 + term_5
    
    return log_likelihood_value

def optimal_spline_fit(x, y):
    """
      Fits cubic splines with varying number of knots and selects the best model
      by minimizing the negative log-likelihood.
  
      Returns:
          min_L (float): Minimum negative log-likelihood
          best_m (int): Optimal number of intervals (knots + 1)
    """
    x, y = x[np.argsort(x)], y[np.argsort(x)]
    best_rss, best_m, L_list = float("inf"), 0, []
    num_knots = compute_knots(len(x))

    for m in range(1, num_knots + 1):
        knots = np.linspace(x.min(), x.max(), m + 1)[1:-1]
        try:
            spline = LSQUnivariateSpline(x, y, t=knots, k=3)
            rss = np.sum((y - spline(x))**2)

            if rss < best_rss:
                best_rss, best_m = rss, m

            uj = np.histogram(x, bins=np.r_[[x.min()], knots, [x.max()]])[0]
            L_list.append(log_likelihood(m, uj, len(x), rss))
        except:
            break

    return min(L_list), best_m

# ALGORITHM 2
def compute_L(x, y):
    
    def normalize(x):
        x = np.array(x)
        min_x = np.min(x)
        max_x = np.max(x)

        if min_x == max_x:
            return np.full(x.shape, 0)  
        else:
            return (x - min_x) / (max_x - min_x)
    # Normalize the data
    X = normalize(x)
    Y = normalize(y)
      
    # Step 2: Compute ηX and ηY
    pairwise_diff_X = np.abs(np.subtract.outer(X, X))
    non_zero_diff_X = pairwise_diff_X[pairwise_diff_X > 0]
    min_diff_X = np.min(non_zero_diff_X)

    pairwise_diff_Y = np.abs(np.subtract.outer(Y, Y))
    non_zero_diff_Y = pairwise_diff_Y[pairwise_diff_Y > 0]
    min_diff_Y = np.min(non_zero_diff_Y)

    # Step 3: Compute L(X) and L(Y)
    L_X = -(len(X) * np.log(min_diff_X))
    L_Y = -(len(Y) * np.log(min_diff_Y))

    return L_X, L_Y

# ALGORITHM 3
def compute_delta_x_to_y(x, y):
    L_x, L_y = compute_L(x, y)
    L_opt, best_m = optimal_spline_fit(x, y)
    return L_opt, best_m


# ALGORITHM 4
def infer_causal_direction(x, y):
    """
      Infers the causal direction between two variables using a spline-based
      minimum description length principle.
  
      Returns:
          direction (str): "->" if X causes Y, "<-" if Y causes X, "undecided" otherwise
          strength (float): Difference in model scores (delta_YX - delta_XY)
    """
    delta_xy, m_x = compute_delta_x_to_y(x, y)
    delta_yx, m_y = compute_delta_x_to_y(y, x)

    if delta_xy < delta_yx:
        direction = "->"  # X causes Y
    elif delta_yx < delta_xy:
        direction = "<-"  # Y causes X
    else:
        direction = "undecided"

    strength = delta_yx - delta_xy  
    return direction, strength, m_x, m_y
    
