#---------------------------------------------------------------------
#  xtoygen.R
#  Generate simulated data using toySWIFT
#  (Version 3.0, January 3, 2023)
#  (c) Ralf Engbert & Maximilian M. Rabe, Universität Potsdam
#---------------------------------------------------------------------
library(ggplot2)
library(grid)
library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))
rm(list=ls())
source("toygen.R")
PDFOUT = F

# read text corpus
corpus = read.table("expdata/Rcorpus_PB2.dat",header=T)
corpus$freq = corpus$freq + 2
fmax = max(corpus$freq)
corpus$lfreq = log10(corpus$freq)/log10(fmax)

# simulation parameters
NSENT = 114   # number of sentences
NRUNS = 1    # number of runs 

# model parameters
nu = 0.25    # spatial decline in processing rate
r = 7      # overall processing rate
mt = 220    # mean fixation duration
iota = 0.4  # coupling of fixation duration and activation
eta = -2    # target noise
beta = 0.6  # word-frequency effect
kappa = 0.5   # inhibition by words left of fixation
gamma = 1.0   # targeting exponent
true = data.frame(nu,r,mt,iota,eta,beta,kappa,gamma)
#write.table(file="true.par",true,row.names=F,col.names=T)

# run toyswift function
data = c()
for ( run in 1:NRUNS ) {
  for ( sen in 1:NSENT ) {
    idx = which(corpus$sentID==sen)
    lfreq = corpus$lfreq[idx]
    #print(c(sen, lfreq))
    traj = toygen(nu,r,mt,iota,eta,beta,kappa,gamma,lfreq)
    traj = traj[,1:3]
    nfix = length(traj[,1])
    traj = cbind(rep(run,nfix),rep(sen,nfix),traj[,2:3])
    data = rbind(data,traj)
  }
}
out.df = data.frame(run=data[,1],sentID=data[,2],word=data[,3],fixdur=data[,4])
write.table(file="sim-prod-sal3.dat",out.df,row.names=F)




