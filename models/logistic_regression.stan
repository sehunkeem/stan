data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] x;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real alpha;
  vector[K] beta;
}
model {
  y ~ bernoulli_logit(alpha + x * beta);
}