data {
  int<lower=1> N;
  int<lower=2> J;
  array[N] int<lower=1, upper=J> group;
  vector[N] x;
  real<lower=0> alpha_prior_sd;
  vector<lower=0>[J] a_prior_sd;
}
parameters {
  real alpha_bayes;
  vector[J] a_bayes;
  real<lower=0> sigma_x;
}
model {
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
