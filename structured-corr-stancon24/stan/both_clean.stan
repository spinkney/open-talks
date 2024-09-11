functions {
  real lb_ub_lp (real y, real lb, real ub) {
    target += log(ub - lb) + log_inv_logit(y) + log1m_inv_logit(y);
    
    return lb + (ub - lb) * inv_logit(y);
  }
  
  matrix cholesky_corr_constrain_lp(int K, vector raw, int N_blocks,
                                    array[,] int res_index,
                                    array[] int res_id,
                                    array[,] int known_index,
                                    vector known_vals,
                                    vector lb, vector ub) {
    matrix[K, K] L = rep_matrix(0, K, K);
    int cnt = 1;
    int N_res = num_elements(res_id);
    int N_known = size(known_vals);
    vector[N_blocks] x_cache;
    vector[N_blocks] jac_cache;
    array[N_blocks] int res_id_cnt = ones_int_array(N_blocks);
    int res_row = 1;
    int known_row = 1;
    
    L[1, 1] = 1;
    if (known_index[known_row, 1] == 2) {
      L[2, 1] = known_vals[1];
      known_row += 1;
    } else {
      L[2, 1] = lb_ub_lp(raw[cnt], lb[cnt], ub[cnt]);
      if (res_index[res_row, 1] == 2) {
        x_cache[res_id[res_row]] = L[2, 1];
        jac_cache[res_id[res_row]] = 1;
        res_id_cnt[res_id[res_row]] += 1;
        res_row += 1;
      } 
      cnt += 1;
    }
    
    L[2, 2] = sqrt(1 - L[2, 1] ^ 2);
    
    for (i in 3 : K) {
      if (res_index[res_row, 1] == i && res_index[res_row, 2] == 1) {
     // restricted case
        if (res_id_cnt[res_id[res_row]] == 1) {
          L[i, 1] = lb_ub_lp(raw[cnt], lb[cnt], ub[cnt]);
          x_cache[res_id[res_row]] = L[i, 1];
          jac_cache[res_id[res_row]] = 1;
          res_id_cnt[res_id[res_row]] += 1;
          cnt += 1;
        } else {
          L[i, 1] = x_cache[res_id[res_row]];
        }
        res_row += 1;
      } 
      else if (known_index[known_row, 1] == i && known_index[known_row, 2] == 1) {
        // known case 
          L[i, 1] = known_vals[known_row];
         known_row += 1;
      } else {
        // base case
        L[i, 1] = lb_ub_lp(raw[cnt], lb[cnt], ub[cnt]);
        cnt += 1;
      } 
      
      L[i, 2] = sqrt(1 - L[i, 1] ^ 2);
      real l_ij_old = log1m(L[i, 1] ^ 2);
      for (j in 2 : i - 1) {
        real b1 = dot_product(L[j, 1 : (j - 1)], L[i, 1 : (j - 1)]);
        real stick_length = exp(0.5 * l_ij_old);
        real low = max({-stick_length, (lb[cnt] - b1) / L[j, j]});
        real up = min({stick_length, (ub[cnt] - b1) / L[j, j]});
        
        if (known_index[known_row, 1] == i && known_index[known_row, 2] == j) {
          L[i, j] = (known_vals[known_row] - b1) / L[j, j];
          known_row = known_row == N_known ? N_known : known_row + 1;
        } else if (res_index[res_row, 1] == i && res_index[res_row, 2] == j) {
          if (res_id_cnt[res_id[res_row]] == 1) {
            L[i, j] = lb_ub_lp(raw[cnt], low, up);
            x_cache[res_id[res_row]] = L[i, j] * L[j, j] + b1;
            jac_cache[res_id[res_row]] = log(L[j, j]);
            res_id_cnt[res_id[res_row]] += 1;
            cnt += 1;
          } else {
            L[i, j] = (x_cache[res_id[res_row]] - b1) / L[j, j];
            target += -log(L[j, j]);
          }
          res_row = res_row == N_res ? N_res : res_row + 1;
        } else {
          L[i, j] = lb_ub_lp(raw[cnt], low, up);
          cnt += 1;
        }
        l_ij_old = log_diff_exp(l_ij_old, 2 * log(abs(L[i, j])));
      }
      L[i, i] = exp(0.5 * l_ij_old);
    }
    return L;
  }
}
data {
  int<lower=2> K; // dimension of correlation matrix
  real<lower=0> eta;
  vector[(K * (K - 1)) %/% 2] lb;
  vector[(K * (K - 1)) %/% 2] ub;
  int<lower=0> free_params;
  int<lower=0> N_blocks;
  int<lower=0> N_res;
  array[N_res, 2] int res_index;
  array[N_res] int res_id;
  
  int<lower=0> N_known;
  array[N_known, 2] int known_index;
  vector[N_known] known_vals;
}
transformed data {
  int k_choose_2 = (K * (K - 1)) %/% 2;
}
parameters {
  vector[free_params] raw;
}
transformed parameters {
  matrix[K, K] L_Omega = cholesky_corr_constrain_lp(K, raw, N_blocks,
                                                    res_index, res_id,
                                                    known_index, known_vals,
                                                    lb,
                                                    ub);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
