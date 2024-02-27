#---------------------------------------------------------------------
#  toygen.R
#  Function for simulation of toySWIFT (generative model)
#  (Version 3.0, January 3, 2023)
#  (c) Ralf Engbert & Maximilian M. Rabe, Universität Potsdam
#---------------------------------------------------------------------
toygen <- function(nu,r,mt,iota,eta,beta,kappa,gamma,lfreq) {
  # activation, saliency, probability
  NW = length(lfreq)     # number of words
  amax = 1 - beta*lfreq  # word frequency dependent maximum activation
  a = rep(0,NW)  # word activation
  s = rep(0,NW)  # word saliency
  p = rep(0,NW)  # selection probability
  # fixation duration 
  shape = 9                  # gamma density shape
  rate = shape/mt            # gamma density rate
  sigma = 1/(1+2*nu+nu^2)    # processing span normalization constant
  # other variables
  time = 0    # time
  k = 1       # fixated word
  traj = c()  # store trajectory
  
  # simulation loop
  while ( length(which(a<1))>0 & k<NW ) {
    
    # 1. Generate fixation duration
    if ( k>1 )  leftact = prod(1+kappa*(s[1:(k-1)]-10^eta)) #sum(s[1:(k-1)])#
    else leftact = 1
    tfix = rgamma(1,shape,(rate*(1+iota*a[k])/leftact))
    
    # 2. Update processing rates
    lambda = rep(0, NW)
    if(k-1 >= 1) lambda[k-1] = nu*sigma
    lambda[k] = sigma
    if(k+1 <= NW) lambda[k+1] = nu*sigma
    if(k+2 <= NW) lambda[k+2] = nu^2*sigma
    
    # 3. Evolve activations
    time = time + tfix
    a = a + r*lambda*tfix/1000
    idx = which(a>amax)
    a[idx] = amax[idx]
    
    # 4. Compute word saliencies 
    s = amax*sin(pi*a/amax) + 10^eta
    
    # 5. store trajectory
    traj = rbind(traj,c(time,k,tfix,s))
    
    # 6. Compute probability for target selection
    p = s/sum(s)      
    
    # 7. Select saccade target
    k = sample.int(n=NW, size=1, prob=p) 
    
  }
  if ( k>1 )  leftact = prod(1+kappa*(s[1:(k-1)]-10^eta)) #sum(s[1:(k-1)])#
  else leftact = 1
  tfix = rgamma(1,shape,(rate*(1+iota*a[k])/leftact))
  traj = rbind(traj,c(time,k,tfix,s))
  return(traj)
}