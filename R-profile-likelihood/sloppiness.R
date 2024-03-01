
#nll <- function(nu, r, mt, iota, eta, beta, kappa) {
nll <- function(ps) {
    ll <- 0.0
    for (i in unique(data$sentID)) {
        ll <- ll + sum(toyLL(corpus[corpus$sentID == i,]$lfreq, data[data$sentID == i,]$word, data[data$sentID == i,]$fixdur, ps[1], ps[2], ps[3], ps[4], ps[5], ps[6], ps[7]))
    }
    return(-ll)
}

#inits <- c(0.25, 7.0, 220.0, 0.4, -2.0, 0.2, 0.1)
#inits <- c(0.385299, 8.12581, 205.279, 1.48644, -4.95495, 0.565822, 1.96575)
inits <- c(0.385299, 8.12581, 205.279, 0.001, -4.95495, 0.565822, 0.001)
nll(inits)

mle <- optim(par=inits, fn=nll, method="CG", hessian=T)#, control=c(maxit=1000, reltol=0.0))
mle2 <- nlm(nll, inits, hessian=T, gradtol=1e-8, steptol=1e-8, print.level=2)
mle
mle2
# We optimized the negative LL, so the Hessian here *is* the observed information matrix
#H <- optimHess(mle$par, nll)
Idecomp <- eigen(mle2$hessian)
(min(abs(Idecomp$values)) / max(abs(Idecomp$values))) < 0.001
