data {
  int<lower=0> P;
  int<lower=0> T;
  array[T] real y;
}
parameters {
  real mu;
  array[P] real phi;
  real sigma;
}
model {
  for (t in (P+1):T) {
    real mean = mu;
    for (p in 1:P) {
      mean += phi[p] * y[t-p];
    }
    y[t] ~ normal(mean, sigma);
  }
}