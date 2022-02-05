data {
  int<lower=0> Q;       // num noise lag
  int<lower=0> P;       // num autoregressive lag
  int<lower=3> T;       // num observations
  vector[T] y;          // observation at time t
}
parameters {
  real mu;              // mean
  real<lower=0> sigma;  // error scale
  vector[Q] theta;      // error coeff, lag -t
  array[P] real phi;

}
transformed parameters {
  vector[T] epsilon;    // error term at time t
  for (t in 1:T) {
    epsilon[t] = y[t] - mu;
    for (q in 1:min(t - 1, Q)) {
      epsilon[t] = epsilon[t] - theta[q] * epsilon[t - q];
    }
  }
}
model {
  vector[T] eta;
  mu ~ cauchy(0, 2.5);
  theta ~ cauchy(0, 2.5);
  sigma ~ cauchy(0, 2.5);
  for (t in 1:T) {
    eta[t] = mu;
    for (q in 1:min(t - 1, Q)) {
      eta[t] = eta[t] + theta[q] * epsilon[t - q];
    }
  }
  for (t in (P+1):T) {
    real mean = 0;
    for (p in 1:P) {
      mean += phi[p] * y[t-p];
    }
    y[t] ~ normal(mean + eta[t], sigma);
  }
}