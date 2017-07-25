source('simulData.R')

args <- commandArgs(trailingOnly = TRUE)
expname <- args[1]
system(paste('mkdir -p',expname))

p <- strtoi(args[2])
howmany <- strtoi(args[3])
bottomUp <- strtoi(args[4])
nObs <- strtoi(args[5])
nInt <- strtoi(args[6])
noiseSigma <- as.numeric(args[7])
weights <- strtoi(args[8])

# different files
for( i in 0:(howmany-1) ) {
  cat('Simulating ',i,'...\n')
  simulData(dataFileName=paste(expname,'/simulData',i,'.Rdata',sep=''),p=p,nObs=nObs,nInt=nInt,seed=i,noiseSigma=noiseSigma,measNoise=0,weights=weights,bottomUp=bottomUp,logsigmaMean=0.0,logsigmaSd=0.0,EdoMean=-2.0,EdoSd=0.0)
}

