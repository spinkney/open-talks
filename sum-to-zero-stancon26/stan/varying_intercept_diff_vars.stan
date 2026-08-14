data {
  int<lower=1> N;
  int<lower=2> J;
  array[N] int<lower=1, upper=J> group;
  vector[N] x;
}
transformed data {
  real mu_gamma = log(2.5);
  real s_gamma = 0.5;
}
parameters {
  real alpha_s2z;
  sum_to_zero_vector[J] a_s2z;
  real z_mean_a_cond;
  real gamma_s2z;
  sum_to_zero_vector[J] z_log_sd_s2z;
  real z_mean_b_cond;
  real lambda_omega;
  real<lower=0> sigma_x;
}
transformed parameters {
  real var_gamma = square(s_gamma);
  real var_mean_b = square(lambda_omega) / J;
  real var_gamma_s2z = var_gamma + var_mean_b;
  real mean_b_cond = var_mean_b / var_gamma_s2z
    * (gamma_s2z - mu_gamma);
  real mean_b = mean_b_cond + lambda_omega
      * sqrt(var_gamma / (J * var_gamma_s2z))
      * z_mean_b_cond;
  real gamma_recovered = gamma_s2z - mean_b;
  real<lower=0> alpha_prior_sd_recovered =
    exp(gamma_recovered);
  vector[J] b_s2z = lambda_omega * z_log_sd_s2z;
  vector[J] b_recovered = b_s2z + mean_b;
  vector<lower=0>[J] a_prior_sd =
    exp(gamma_s2z + b_s2z);
  vector[J] inv_var_a = inv(square(a_prior_sd));
  real inv_var_alpha =
    inv(square(alpha_prior_sd_recovered));
  real precision_mean_a = inv_var_alpha + sum(inv_var_a);
  real mean_a_cond = (alpha_s2z * inv_var_alpha
      - dot_product(a_s2z, inv_var_a)) / precision_mean_a;
  real sd_mean_a_cond = inv_sqrt(precision_mean_a);
  real mean_a_bayes = mean_a_cond + sd_mean_a_cond * z_mean_a_cond;
  real alpha_bayes_recovered = alpha_s2z - mean_a_bayes;
  vector[J] a_bayes_recovered = a_s2z + mean_a_bayes;
}
model {
  gamma_s2z ~ normal(mu_gamma, sqrt(var_gamma_s2z));
  z_log_sd_s2z ~ std_normal();
  z_mean_b_cond ~ std_normal();
  lambda_omega ~ normal(0, 0.25);

  alpha_s2z - mean_a_cond ~ normal(0, alpha_prior_sd_recovered);
  a_s2z + mean_a_cond ~ normal(0, a_prior_sd);
  target += 0.5 * log(2 * pi()) - 0.5 * log(precision_mean_a);
  z_mean_a_cond ~ std_normal();
  sigma_x ~ normal(0, 2);
  x ~ normal(alpha_s2z + a_s2z[group], sigma_x);
}
