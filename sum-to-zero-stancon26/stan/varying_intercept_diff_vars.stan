data {
  int<lower=1> N;
  int<lower=2> J;
  array[N] int<lower=1, upper=J> group;
  vector[N] x;
  real<lower=0> alpha_prior_sd;
  vector<lower=0>[J] a_prior_sd;
}
parameters {
  real alpha_s2z;
  sum_to_zero_vector[J] a_s2z;
  real mean_a_bayes;
  real<lower=0> sigma_x;
}
transformed parameters {
  real alpha_bayes_recovered = alpha_s2z - mean_a_bayes;
  vector[J] a_bayes_recovered = a_s2z + mean_a_bayes;
}
model {
  sigma_x ~ normal(0, 2);
  // Priors are placed in the intuitive Bayesian coordinates.
  alpha_bayes_recovered ~ normal(0, alpha_prior_sd);
  a_bayes_recovered ~ normal(0, a_prior_sd);
  // Exact change of variables; constant because J is fixed.
  target += 0.5 * log(J);
  x ~ normal(alpha_bayes_recovered + a_bayes_recovered[group], sigma_x);
}
