functions {
  /*
   * 1. Core Isotropic Zero-Sum Bounded Transform
   * Maps unconstrained y -> bounded zero-sum x and applies the Jacobian
   */
  vector isotropic_zerosum_bounded_jacobian(vector y, vector L, vector U) {
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
      jacobian += log(sigma_k) - 0.5 * square(y[k]) + log(diff) + 0.5 * square(z_k);
      
      R -= x[k];
    }
    
    x[N] = R;
    return x;
  }

  /*
   * 2. The 2-Loop Simplex Bounding Pipeline
   * Takes raw y, derives symbolic bounds, maps to x, returns softmax(x)
   */
  vector bounded_simplex_from_zerosum_jacobian(vector y, vector A, vector B) {
    int N = num_elements(A);
    vector[N] A_tilde;
    vector[N] B_tilde;
    
    real sum_A = sum(A);
    real sum_B = sum(B);
    
    // =========================================================
    // LOOP 1: Derive tight symbolic bounds on the simplex
    // =========================================================
    // Account for the mass that MUST be absorbed by the rest of the elements
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
    for (i in 1:N) {
      // Minimum possible x_i: set p_i = A_tilde, maximize rest of simplex to B_tilde
      real max_log_sum = sum_log_B - log(B_tilde[i]) + log(A_tilde[i]);
      L_x[i] = log(A_tilde[i]) - max_log_sum / N;
      
      // Maximum possible x_i: set p_i = B_tilde, minimize rest of simplex to A_tilde
      real min_log_sum = sum_log_A - log(A_tilde[i]) + log(B_tilde[i]);
      U_x[i] = log(B_tilde[i]) - min_log_sum / N;
    }
    
    // =========================================================
    // 3. Sequential Sampling & Output Map
    // =========================================================
    vector[N] x = isotropic_zerosum_bounded_jacobian(y, L_x, U_x);
    
    return softmax(x);
  }
}
data {
  int<lower=2> N;
  vector<lower=0>[N] A; // Simplex lower bounds (must be > 0)
  vector<lower=0>[N] B; // Simplex upper bounds
}

parameters {
  vector[N - 1] y; 
}
transformed parameters {
    vector[N] p = bounded_simplex_from_zerosum_jacobian(y, A, B);
}

model {
  // 1. Standard normal prior gives us the exact isotropic geometry
  y ~ std_normal();
  
  // 3. Evaluate your model likelihood using p
  // e.g., target += multinomial_lpmf(counts | p);
}