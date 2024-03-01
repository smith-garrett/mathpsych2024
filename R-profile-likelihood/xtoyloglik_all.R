#---------------------------------------------------------------------
#  xtoyLOGLIK
#  (Version 2.0, February 20, 2024)
#  (c) Ralf Engbert & Maximilian M. Rabe, Universität Potsdam
#---------------------------------------------------------------------
library(ggplot2)
library(grid)
library(latex2exp)
library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))
rm(list=ls())
source("toyLL.R")
PDFOUT = T#F

# set output file
if ( PDFOUT ) pdf("Fig_loglik_corrected.pdf", width=6, height=8)#paper='a4r')
grid.newpage()
pushViewport(viewport(layout=grid.layout(7,3)))

# read text corpus
corpus = read.table("expdata/Rcorpus_PB2.dat",header=T)
corpus$freq = corpus$freq + 2
fmax = max(corpus$freq)
corpus$lfreq = log10(corpus$freq)/log10(fmax)

# load fixation sequences
data = read.table("sim-prod-sal3.dat",header=TRUE)
sID_list = unique(data$sentID)
Nruns = max(data$run)

# parameter variation
nu_vec = seq(from=0.05,to=0.95,by=0.01)
r_vec = seq(from=0.01,to=30.01,by=1)
mt_vec = seq(from=100,to=300,by=10)
iota_vec = seq(from=0,to=3,by=0.05)
eta_vec = seq(from=-3,to=-1,by=0.1)
beta_vec = seq(from=0,to=0.99,by=0.05) # PROBLEM: beta must be strictly less than 1
kappa_vec = seq(from=-0.5,to=2,by=0.05)

for ( par in 1:7 ) {
  # model parameters
  true = read.table("true.par",header=T)
  nu = true$nu        # spatial decline in processing rate
  r = true$r          # overall processing rate
  mt = true$mt        # mean fixation duration
  iota = true$iota    # coupling of fixation duration and activation
  eta = true$eta      # target noise
  beta = true$beta    # word-frequency effect
  kappa = true$kappa  # inhibition by words left of fixation
  
  # variation of one parameter
  parname = switch(par,"\\nu","r","\\mu_T","\\iota","\\eta","\\beta","\\kappa")
  parvec = switch(par,nu_vec,r_vec,mt_vec,iota_vec,eta_vec,beta_vec,kappa_vec)

  # computation of the loglik function
  llvec = c()
  for ( var in parvec ) {
    if ( par==1 )  nu = var
    if ( par==2 )  r = var
    if ( par==3 )  mt = var
    if ( par==4 )  iota = var
    if ( par==5 )  eta = var
    if ( par==6 )  beta = var
    if ( par==7 )  kappa = var
    
    # run toyloglik function
    loglik = c(0,0)
    NW = 0
    for ( run in 1:Nruns ) {
      for ( sent in sID_list ) {
        idx = which(corpus$sentID==sent)
        lfreq = corpus$lfreq[idx]
        idx = which(data$run==run & data$sentID==sent)
        fseq = cbind(data$word[idx],data$fixdur[idx])
        fixdur = fseq[,2]
        fixpos = fseq[,1]
        loglik = loglik + toyLL(lfreq,fixpos,fixdur,nu,r,mt,iota,eta,beta,kappa)
        NW = NW + length(lfreq)
      }
    }
    llvec = rbind(llvec,c(loglik,sum(loglik)))
  }
  llvec = llvec/NW
  
  # store results in data frame
  results.df = data.frame(parvec=parvec,loglik=llvec[,3],loglik_time=llvec[,1],
                          loglik_spat=llvec[,2])
  
  # plot results
  p1 = ggplot(results.df) + geom_line(aes(x=parvec,y=loglik),color="blue") + 
    scale_y_continuous(expression(loglik))
  if ( par==1 )  p1 = p1 + scale_x_continuous(expression(nu)) + geom_vline(xintercept=true$nu,linetype='dashed')
  if ( par==2 )  p1 = p1 + scale_x_continuous(expression(r)) + geom_vline(xintercept=true$r,linetype='dashed')
  if ( par==3 )  p1 = p1 + scale_x_continuous(expression(mu[T])) + geom_vline(xintercept=true$mt,linetype='dashed')
  if ( par==4 )  p1 = p1 + scale_x_continuous(expression(iota)) + geom_vline(xintercept=true$iota,linetype='dashed')
  if ( par==5 )  p1 = p1 + scale_x_continuous(expression(eta)) + geom_vline(xintercept=true$eta,linetype='dashed')
  if ( par==6 )  p1 = p1 + scale_x_continuous(expression(beta)) + geom_vline(xintercept=true$beta,linetype='dashed')
  if ( par==7 )  p1 = p1 + scale_x_continuous(expression(kappa)) + geom_vline(xintercept=true$kappa,linetype='dashed')
  print(p1,vp=viewport(layout.pos.row=par,layout.pos.col=1)) 
  
  p2 = ggplot(results.df) + geom_line(aes(x=parvec,y=loglik_time),color="red") +
    scale_y_continuous(expression(loglik^temp))
  if ( par==1 )  p2 = p2 + scale_x_continuous(expression(nu)) + geom_vline(xintercept=true$nu,linetype='dashed')
  if ( par==2 )  p2 = p2 + scale_x_continuous(expression(r)) + geom_vline(xintercept=true$r,linetype='dashed')
  if ( par==3 )  p2 = p2 + scale_x_continuous(expression(mu[T])) + geom_vline(xintercept=true$mt,linetype='dashed')
  if ( par==4 )  p2 = p2 + scale_x_continuous(expression(iota)) + geom_vline(xintercept=true$iota,linetype='dashed')
  if ( par==5 )  p2 = p2 + scale_x_continuous(expression(eta)) + geom_vline(xintercept=true$eta,linetype='dashed')
  if ( par==6 )  p2 = p2 + scale_x_continuous(expression(beta)) + geom_vline(xintercept=true$beta,linetype='dashed')
  if ( par==7 )  p2 = p2 + scale_x_continuous(expression(kappa)) + geom_vline(xintercept=true$kappa,linetype='dashed')
  print(p2,vp=viewport(layout.pos.row=par,layout.pos.col=2)) 
  
  p3 = ggplot(results.df) + geom_line(aes(x=parvec,y=loglik_spat),color="green") +
    scale_y_continuous(expression(loglik^spat))
  if ( par==1 )  p3 = p3 + scale_x_continuous(expression(nu)) + geom_vline(xintercept=true$nu,linetype='dashed')
  if ( par==2 )  p3 = p3 + scale_x_continuous(expression(r)) + geom_vline(xintercept=true$r,linetype='dashed')
  if ( par==3 )  p3 = p3 + scale_x_continuous(expression(mu[T])) + geom_vline(xintercept=true$mt,linetype='dashed')
  if ( par==4 )  p3 = p3 + scale_x_continuous(expression(iota)) + geom_vline(xintercept=true$iota,linetype='dashed')
  if ( par==5 )  p3 = p3 + scale_x_continuous(expression(eta)) + geom_vline(xintercept=true$eta,linetype='dashed')
  if ( par==6 )  p3 = p3 + scale_x_continuous(expression(beta)) + geom_vline(xintercept=true$beta,linetype='dashed')
  if ( par==7 )  p3 = p3 + scale_x_continuous(expression(kappa)) + geom_vline(xintercept=true$kappa,linetype='dashed')
  print(p3,vp=viewport(layout.pos.row=par,layout.pos.col=3)) 
}
if ( PDFOUT )  dev.off()
