functions {
  tuple(vector, vector) rev_cumsum_bounds (vector L, vector U) {
    int N = num_elements(L);
    vector[N] rev_cumsum_L;
    vector[N] rev_cumsum_U;
    rev_cumsum_L[N] = L[N];
    rev_cumsum_U[N] = U[N];
    for (i in 1:N - 1) {
      rev_cumsum_L[N - i] = rev_cumsum_L[N - i + 1] + L[N - i];
      rev_cumsum_U[N - i] = rev_cumsum_U[N - i + 1] + U[N - i];
    }
    return (rev_cumsum_L, rev_cumsum_U);
  } 

  vector isotropic_zerosum_bounded_jacobian(vector y, tuple(vector, vector) bounds, tuple(vector, vector) rev_cumsum) {
    int N = num_elements(bounds.1);
    vector[N] x;
    real R = 0.0;
    
    // 2. Sequential mapping
    for (k in 1:N - 1) {
      real sum_U_rem = rev_cumsum.2[k + 1];
      real sum_L_rem = rev_cumsum.1[k + 1];
      
      // Dynamic bounds based on remainder required
      real L_tilde = fmax(bounds.1[k], R - sum_U_rem);
      real U_tilde = fmin(bounds.2[k], R - sum_L_rem);
      
      // Isotropic Conditional Parameters
      real M_k = N - k + 1.0; 
      real mu_k = R / M_k;
      real sigma_k = sqrt((M_k - 1.0) / M_k);
      
      // Transform boundaries to standard normal space
      real alpha = (L_tilde - mu_k) / sigma_k;
      real beta  = (U_tilde - mu_k) / sigma_k;
      
      real v_k = Phi(alpha);
      real w_k = Phi(beta);
      real diff = w_k - v_k;
      
      // Map unconstrained y through Standard Normal CDF
      real u_k = Phi(y[k]);
      
      // Squeeze into the truncated window and map back
      real p_k = v_k + u_k * diff;
      real z_k = inv_Phi(p_k);
      
      x[k] = mu_k + sigma_k * z_k;
      
      // Add exact Log-Determinant of the Jacobian to target
      jacobian += log(sigma_k) - 0.5 * square(y[k]) + log(diff) + 0.5 * square(z_k);
      
      // Update remainder
      R -= x[k];
    }
    
    // 3. Final element is completely determined to guarantee sum to zero
    x[N] = R;
    
    return x;
  }

  /*
   * 1. The Core Isotropic Zero-Sum Bounded Transform
   * (Exactly as derived previously, operating on dynamic L and U)
   */
  vector isotropic_zerosum_bounded_lp(vector y, vector L, vector U) {
    int N = num_elements(L);
    vector[N] x;
    real R = 0.0;
    
    vector[N] rev_cumsum_L;
    vector[N] rev_cumsum_U;
    rev_cumsum_L[N] = L[N];
    rev_cumsum_U[N] = U[N];
    for (i in 1:(N - 1)) {
      rev_cumsum_L[N - i] = rev_cumsum_L[N - i + 1] + L[N - i];
      rev_cumsum_U[N - i] = rev_cumsum_U[N - i + 1] + U[N - i];
    }
    
    for (k in 1:(N - 1)) {
      real sum_U_rem = rev_cumsum_U[k + 1];
      real sum_L_rem = rev_cumsum_L[k + 1];
      
      real L_tilde = fmax(L[k], R - sum_U_rem);
      real U_tilde = fmin(U[k], R - sum_L_rem);
      
      real M_k = N - k + 1.0; 
      real mu_k = R / M_k;
      real sigma_k = sqrt((M_k - 1.0) / M_k);
      
      real alpha = (L_tilde - mu_k) / sigma_k;
      real beta  = (U_tilde - mu_k) / sigma_k;
      
      real v_k = Phi(alpha);
      real w_k = Phi(beta);
      real diff = w_k - v_k;
      
      real u_k = Phi(y[k]);
      real p_k = v_k + u_k * diff;
      real z_k = inv_Phi(p_k);
      
      x[k] = mu_k + sigma_k * z_k;
      
      // Isotropic Jacobian increment
      target += log(sigma_k) - 0.5 * square(y[k]) + log(diff) + 0.5 * square(z_k);
      
      R -= x[k];
    }
    
    x[N] = R;
    return x;
  }

  /*
   * 2. The 2-Loop Coupled Bounds Mapper
   * Derives bounds from the simplex and samples the zero-sum vector
   */
  vector simplex_to_isotropic_zerosum_lp(vector y, vector A, vector B) {
    int N = num_elements(A);
    vector[N] A_tilde;
    vector[N] B_tilde;
    
    real sum_A = sum(A);
    real sum_B = sum(B);
    
    // =========================================================
    // LOOP 1: Derive tight symbolic bounds on the simplex
    // =========================================================
    // Because sum(p) == 1, an element cannot be larger than 1 minus 
    // the minimum mass required by the rest of the elements.
    for (i in 1:N) {
      A_tilde[i] = fmax(A[i], 1.0 - (sum_B - B[i]));
      B_tilde[i] = fmin(B[i], 1.0 - (sum_A - A[i]));
    }
    
    real sum_log_A = sum(log(A_tilde));
    real sum_log_B = sum(log(B_tilde));
    
    vector[N] L_x;
    vector[N] U_x;
    
    // =========================================================
    // LOOP 2: Map the coupled bounds back to Zero-Sum Space
    // =========================================================
    // x_i = log(p_i) - 1/N * sum(log p)
    for (i in 1:N) {
      // To MINIMIZE x_i, we set p_i = A_tilde, and maximize the rest of the simplex to B_tilde
      real max_log_sum = sum_log_B - log(B_tilde[i]) + log(A_tilde[i]);
      L_x[i] = log(A_tilde[i]) - max_log_sum / N;
      
      // To MAXIMIZE x_i, we set p_i = B_tilde, and minimize the rest of the simplex to A_tilde
      real min_log_sum = sum_log_A - log(A_tilde[i]) + log(B_tilde[i]);
      U_x[i] = log(B_tilde[i]) - min_log_sum / N;
    }
    
    // 3. Sequentially sample taking all values into account
    return isotropic_zerosum_bounded_lp(y, L_x, U_x);
  }
}
}