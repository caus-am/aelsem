simulData <- function(dataFileName,p,nObs,nInt,seed=123,prob=c(0.125,0.5,0.25,0.125),probConf=rbind(matrix(0,p,1),matrix(1,1,1)),numConf=0,sigmaConf=0.5,noiseSigma=0.1,measNoise=0.0,bottomUp=FALSE,logsigmaMean=0.0,logsigmaSd=0.0,EdoMean=-2.0,EdoSd=0.0,dotFileName='',weights=0,permute=FALSE)
{
# For reproducibility
set.seed(seed)

# Total number of variables (confounders + observed)
P <- numConf + p

# First, a random directed acyclic graph representing the true causal structure is built.
# B[i,j] = causal effect of Xi on Xj
B <- matrix(0,P,P)
# We start with numConf "global" confounders, i.e., unobserved variables that influence observed ones.
override_extY = FALSE
if( P >= 6 && P <= 7 ) {
  # Special case: generate extended Y-structures
  override_extY = TRUE
  obsStart = 0 # P - 4
  confStart = 4 # 0
  Narrows = P # happens to equal P
  if( weights == 1 ) {
    randomweights <- runif(Narrows,-sqrt(3),sqrt(3))
  } else if( weights == -1 ) {
    randomweights <- runif(Narrows,2./3.,4./3.) * (Narrows(Nchildren,1,0.5)*2-1)
  } else if( weights == -2 ) {
    randomweights <- runif(Narrows,0,2*sqrt(3))
  } else {
    randomweights <- rnorm(Narrows)
  }
  B[confStart + 1, obsStart + 1] <- randomweights[1]
  B[confStart + 1, obsStart + 3] <- randomweights[2]
  B[confStart + 2, obsStart + 2] <- randomweights[3]
  B[confStart + 2, obsStart + 3] <- randomweights[4]
  B[obsStart + 1, obsStart + 2] <- randomweights[5]
  if( P == 6 ) {
    # Z --> X
    B[obsStart + 4, obsStart + 1] <- randomweights[6]
  } else {
    # Z <-> X
    B[confStart + 3, obsStart + 1] <- randomweights[6]
    B[confStart + 3, obsStart + 4] <- randomweights[7]
  }
} else if( numConf > 0 ) {
  for( i in 1:numConf ) {
    Nchildren = sample(x=length(probConf),size=1,prob=probConf) - 1
    cat('Confounder',i,'has',Nchildren,'children\n')
    if( Nchildren > 0 ) {
      ch <- sample(x=p,size=Nchildren,replace=FALSE)
      if( weights == 1 ) {
        randomweights <- runif(Nchildren,-sqrt(3),sqrt(3))
      } else if( weights == -1 ) {
        randomweights <- runif(Nchildren,2./3.,4./3.) * (rbinom(Nchildren,1,0.5)*2-1)
      } else if( weights == -2 ) {
        randomweights <- runif(Nchildren,0,2*sqrt(3))
      } else {
        randomweights <- rnorm(Nchildren)
      }
      B[i, numConf + ch] <- randomweights
#      B[i, numConf + ch] <- matrix(1,1,Nchildren)
    }
  }
  # B[1:numConf,(1+numConf):(p+numConf)] <- matrix(rnorm(numConf*p),numConf,p) * (matrix(runif(numConf*p),numConf,p) <= 1.0)
}
if( !override_extY && bottomUp ) {
  # Then, for the observed variables, from bottom to top, for each node,
  # the number of children of that node is chosen randomly (with degree distribution prob),
  # and the children are selected out of all nodes with higher index at random.
  # Then, for each edge, the strength of the interaction is sampled from a Gaussian
  # distribution with mean zero.
  for( i in p:1 ) {
    Nchildren = sample(x=length(prob),size=1,prob=prob) - 1
    if( Nchildren > (p - i) )
      Nchildren <- p - i
    if( Nchildren > 0 ) {
      ch <- p + 1 - sample(x=p - i,size=Nchildren,replace=FALSE)
      if( weights == 1 ) {
        randomweights <- runif(Nchildren,-sqrt(3),sqrt(3))
      } else if( weights == -1 ) {
        randomweights <- runif(Nchildren,2./3.,4./3.) * (rbinom(Nchildren,1,0.5)*2-1)
      } else if( weights == -2 ) {
        randomweights <- runif(Nchildren,0,2*sqrt(3))
      } else {
        randomweights <- rnorm(Nchildren)
      }
      B[numConf + i, numConf + ch] <- randomweights
#      B[numConf + i,numConf + ch] <- matrix(1,1,Nchildren)
    }
  }
} else if (!override_extY) {
  # Then, for the observed variables, from top to bottom, for each node, 
  # the number of parents of that node is chosen randomly (with degree distribution prob), 
  # and the parents are selected out of all previous nodes at random.
  # Then, for each edge, the strength of the interaction
  # is sampled from a Gaussian distribution with mean zero.
  for( i in 1:p ) {
    Nparents = sample(x=length(prob),size=1,prob=prob) - 1
    if( Nparents > i - 1 )
      Nparents <- i - 1
    if( Nparents > 0 ) {
      pa <- sample(x=i-1,size=Nparents,replace=FALSE)
      if( weights == 1 ) {
        randomweights <- runif(Nparents,-sqrt(3),sqrt(3))
      } else if( weights == -1 ) {
        randomweights <- runif(Nparents,2./3.,4./3.) * (rbinom(Nparents,1,0.5)*2-1)
      } else if( weights == -2 ) {
        randomweights <- runif(Nparents,0,2*sqrt(3))
      } else {
        randomweights <- rnorm(Nparents)
      }
      B[numConf + pa,numConf + i] <- randomweights
#      B[numConf + pa,numConf + i] <- matrix(1,Nparents,1)
    }
  }
}

# The weights are rescaled so that in expectation, each variable will have a 
# stddev of sigmas[i], taking into account additive Gaussian noise for each variable. 
# sigmas are the expected standard deviations of X_i
# sigmas <- rep(1.0,times=P)                                                                # uniform scales (in expectation)
sigmas <- c(rep(sigmaConf,times=numConf),exp(rnorm(p,mean=logsigmaMean,sd=logsigmaSd)))     # obsered variables with log-Gaussian distribution
scales <- rep(1.0,times=P)
for( i in 1:P ) {
  if( i == 1 ) {
    scales[i] <- sigmas[i] / noiseSigma
  } else if( i == 2 ) {
    scales[i] <- sigmas[i] * (sum((sigmas[1] * B[1,i])^2) + noiseSigma^2)^-0.5
  } else {
    scales[i] <- sigmas[i] * (sum((sigmas[1:(i-1)] * B[1:(i-1),i])^2) + noiseSigma^2)^-0.5
  }
  B[,i] <- B[,i] * scales[i]
}

# Simulate observational data
E <- matrix(rnorm(nObs*P),nrow=nObs) %*% diag(scales * noiseSigma)
noiseCov <- diag((scales * noiseSigma)^2)
#X <- E / (eye(P) - B)   # SLOW
#obs <- t(solve(t(diag(1,P) - B),t(E)))   # SLOW
obs <- t(forwardsolve(t(diag(1,P) - B), t(E), transpose=FALSE))
# Only select observed variables
obs <- obs[,(numConf+1):(numConf+p)]

# Intervention data are simulated for a random subset of variables,
# sequentially setting each of them to a random value (normally distributed)
# and simulating a single corresponding data point.
intpos <- sample(p,size=nInt,replace=TRUE)
muts <- matrix(0,nInt,p)
if( nInt > 0 ) {
  for( j.idx in 1:nInt ) {
    j <- intpos[j.idx]
#    cat("intervention", j.idx, "out of", nInt, "\n")
    Edoj <- matrix(rnorm(P),nrow=1) %*% diag(scales * noiseSigma)
    Edoj[numConf + j] <- rnorm(1,mean=EdoMean,sd=EdoSd)
    B_doj <- B
    B_doj[,numConf + j] <- 0
  #  X_doj = E_doj / (eye(P) - B_doj);  # SLOW
  #  X_doj <- t(solve(t(diag(1,P) - B_doj),t(Edoj)))  # SLOW
    X_doj <- t(forwardsolve(t(diag(1,P) - B_doj), t(Edoj), transpose=FALSE))
    muts[j.idx,1:p] <- X_doj[(numConf + 1):(numConf + p)]
  }
}

# Add measurement noise
obs <- obs + matrix(rnorm(nObs*p,sd=measNoise),nrow=nObs)
if( nInt > 0 ) {
  muts <- muts + matrix(rnorm(nInt*p,sd=measNoise),nrow=nInt)
}

# Apply random permutation of variables, if requested
if( permute ) {
  perm <- sample(1:p)
  iperm <- 1:p
  iperm[perm] <- 1:p
  obs <- obs[,perm]
  int <- int[,perm]
  intpos <- iperm[intpos]
  B <- B[perm,perm]
  noiseCov <- noiseCov[perm,perm]
} else {
  perm <- 1:p
  iperm <- 1:p
}

# Make dot file, if requested
if( dotFileName != '' ) {
  sink(file=dotFileName)
  cat('digraph G {\n')
  for(i in 1:p) {
    for( j in 1:p) {
      if( B[i,j] != 0 ) {
        cat(i,'->',j,'\n')
      }
    }
  }
  cat('}\n')
  sink()
}

# Save / return simulated data
data <- list(obs=obs,int=muts,intpos=intpos,p=p,nObs=nObs,nInt=nInt,B=B,numConf=numConf,noiseCov=noiseCov,perm=perm,iperm=iperm)
if( dataFileName == '' )
  return(data)
else
  save(data,file=dataFileName)
}
