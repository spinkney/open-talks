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
  real alpha_bayes;
  vector[J] a_bayes;
  real gamma;
  vector[J] z_log_sd;
  real lambda_omega;
  real<lower=0> sigma_x;
}
transformed parameters {
  real<lower=0> alpha_prior_sd = exp(gamma);
  vector[J] b = lambda_omega * z_log_sd;
  vector<lower=0>[J] a_prior_sd = exp(gamma + b);
}
model {
  gamma ~ normal(mu_gamma, s_gamma);
  z_log_sd ~ std_normal();
  lambda_omega ~ normal(0, 0.25);
  alpha_bayes ~ normal(0, alpha_prior_sd);
  a_bayes ~ normal(0, a_prior_sd);
  sigma_x ~ normal(0, 2);
  x ~ normal(alpha_bayes + a_bayes[group], sigma_x);
}
generated quantities {
  real mean_a = mean(a_bayes);
  real alpha_s2z_recovered = alpha_bayes + mean_a;
  vector[J] a_s2z_recovered = a_bayes - mean_a;
}
