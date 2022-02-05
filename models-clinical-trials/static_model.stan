data {
  int<lower=1> J; // Number of countries 
  int<lower=1> maxN; // max number of observations per country
  array[J] int<lower=1> N ; // number of observations per country
  array[J,maxN] int<lower=0> y; // Poisson observations (dummy-filled ragged array)
  array[J,maxN] real x; // Input  (dummy-filled ragged array)
  int<lower=1> n_tilde;
}

transformed data {
  int<lower=1> eta_start[J] ;
  int<lower=1> eta_end[J] ;
  eta_start[1] = 1 ;
  eta_end[1] = N[1] ;
  for(j in 2:J){
    eta_start[j] = eta_end[j-1] +1 ;
    eta_end[j] = eta_start[j] -1 + N[j] ;
}
}

parameters {
  matrix[J, maxN] lambda; // each lambda drawn from gamma(a,b);
}

model {    
    
  for (j in 1:J) {
      for (n in 1:N[j]){
      target += gamma_lpdf(lambda[j,n] | 2, 0.5);
    }
  }
    


    
  for (j in 1:J) {
      for (n in 1:N[j]){
      target += poisson_lpmf(y[j,n] | lambda[j,n]);
    }
    
}
}

generated quantities {  
  vector[eta_end[J]] log_lik;
  int<lower=0> count = 0;
  for (j in 1:J) {
    for (n in 1:N[j]) {
      count += 1;
      log_lik[count] =  poisson_lpmf(y[j,n] | lambda[j,n]);
  }      
  }
}