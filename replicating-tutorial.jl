### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 0defa484-7311-438e-8292-45d697844c7b
begin
	using PlutoUI
	TableOfContents(title="Identifiability & Sloppiness")
end

# ╔═╡ 6ab91324-f6cf-42e4-8f8d-0502fa717ba4
begin
	using Distributions, Random
	using Plots, StatsPlots
	using SpecialFunctions
	using StatsBase
	using DataFrames
	using Turing, Optim, DynamicPPL
	using ForwardDiff, LinearAlgebra
	using OrderedCollections
	#using MinimallyDisruptiveCurves, OrdinaryDiffEq
	#Random.seed!(54321)
end

# ╔═╡ 1a62d194-bf7c-11ee-33ac-a52260216072
md"""
# Replicating Ralf & Max's tutorial
"""

# ╔═╡ fba9d036-d8c4-491e-8b67-624c1031862d
md"""
# Setting up for simulating data

By simulating data from the model, we know that the likelihood is exactly the data generating process. So what we learn about identifiability here is the *best-case* scenario. We should not expect the identifiability/sloppiness of the model to turn out this good when we use real data, because we know that our model is a major simplification (possibly incorrect) of the actual data generating process (=cognition).
"""

# ╔═╡ 19295fd8-55ad-4c3a-b931-eadefa61ed9c
md"""
## Simulating word frequency data

Word frequencies tend to have a roughly Zipfian/ζ distribution.
"""

# ╔═╡ 41a44fe5-c3a4-4264-b387-fd1e581950a3
function zetaprobs(s, xs)
	ptemp = xs.^(-s) ./ zeta(s, 1)
	return ptemp ./ sum(ptemp)
end

# ╔═╡ 9f489ab7-80c9-4cf0-baa7-5c31e07c9cda
plot(zetaprobs(1.5, 1:10), title="Prob. mass function for word frequencies")

# ╔═╡ 4f884ba9-8aaf-451f-9768-f2d88626944b
rzipf = rand(Categorical(zetaprobs(1.5, 1:100)), 1000000)

# ╔═╡ 481fad8c-2be2-4e90-9cbc-d9f0252c4361
histogram(rzipf)#, ylims=(0.01, 1e7), yscale=:log10, xscale=:log10, xlims=(1, 100))

# ╔═╡ 798b1894-86d3-4098-ae25-52c2bac85197
cts = countmap(rzipf)

# ╔═╡ 8f7c3384-0b62-4b4a-8192-b68497cb44be
mx = maximum(values(cts))

# ╔═╡ 795c4e72-bcd4-43c9-8278-b536af9df641
nlogfreqs = Dict(k => cts[k] / mx for k in keys(cts))

# ╔═╡ 7e538a57-372c-48f1-9113-4aebf07d08b3
extrema(values(nlogfreqs))

# ╔═╡ d1ca84fe-2570-4a91-8b77-71b1959c12f7
md"""
## Setting up the model to generate data

The versions here closely follow the current ToySWIFT implementation.
"""

# ╔═╡ 647d9fd8-230b-4604-bf27-1c0af38e01ee
function update_proc_rates!(λ, currword, ν, σ)
	nwords = length(λ)
	λ .= zero(eltype(λ))
	(currword-1 >= 1) && (λ[currword-1] = ν * σ)
	λ[currword] = σ
	(currword+1 <= nwords) && (λ[currword+1] = ν * σ)
	(currword+2 <= nwords) && (λ[currword+2] = ν^2 * σ)
	nothing
end

# ╔═╡ af95b3ce-049d-42e7-ac5f-9b884b17254a
function update_activations!(activations, r, λ, fixdur, maxact)
	activations .+= r .* λ .* fixdur .* 0.001
	idx = activations .>= maxact
	activations[idx] .= maxact[idx]
	nothing
end

# ╔═╡ 6cb6113f-9648-4034-bcef-ae8e4abcbf30
function update_saliencies!(activations, saliencies, maxact, minsal)
	saliencies .= (maxact .* sinpi.(activations ./ maxact)) .+ minsal
	nothing
end

# ╔═╡ 155db03c-099d-443d-a622-fdd6302500f4
function update_probabilities!(probs, sal, γ)
	salexp = sal.^γ
	probs .= salexp ./ sum(salexp)
	nothing
end

# ╔═╡ 2e3d51d4-d419-4cea-ab6e-f879aaabdbf9
"""
	simsent(nwords, ps, worddict, sentnr = missing)

Simulates a single sentence with `nwords` using parameters `ps` and a dictionary containing word indices and their normalized log frequencies. `sentnr` is an optional sentence ID to save with the generated data.
"""
function simsent(nwords, ps, worddict, sentnr = missing)
	σ = inv(1 + 2*ps.ν + ps.ν^2)
	shapeparam = 9.0
	rate0 = shapeparam / ps.μ
	minsal = 10^-3.0 # ps.η
	wordids = rand(keys(worddict), nwords)
	#maxact = 1 .- params.β .* rand(nwords)
	maxact = 1 .- ps.β .* [nlogfreqs[k] for k in wordids]
	activations = zeros(nwords)
	saliencies = zeros(nwords)
	probabilities = zeros(nwords)
	λ = zeros(nwords)
	currword = 1
	# Preallocating a place to save scanpath
	scanpath = Array{NamedTuple}(undef, 0)
	sizehint!(scanpath, nwords)

	while any(activations .< maxact) && currword ≠ nwords
		# Generate new fixation duration
		#fixdur = rand(LogNormal(log(ps.μ), ps.sd))
		#fixdur = rand(Gamma(shapeparam, inv(rate0 * (1 + ps.ι*activations[currword]))))
		fixdur = rand(Gamma(shapeparam, inv(rate0)))# + eps()))
		
		# Updating
		update_proc_rates!(λ, currword, ps.ν, σ)
		update_activations!(activations, ps.r, λ, fixdur, maxact)
		update_saliencies!(activations, saliencies, maxact, minsal)
		update_probabilities!(probabilities, saliencies, 1.0)

		# Saving
		push!(scanpath, (sentnr = sentnr, sentlength = nwords, fixatedword = currword, wordid = wordids[currword], fixationduration = fixdur, wordids = wordids))
	
		# Make a saccade
		currword = rand(Categorical(probabilities))
	end

	# Last fixation
	#fixdur = rand(LogNormal(log(ps.μ), ps.sd))
	#fixdur = rand(Gamma(shapeparam, inv(rate0 * (1 + ps.ι*activations[currword]))))
	fixdur = rand(Gamma(shapeparam, inv(rate0)))# + eps()))
	push!(scanpath, (sentnr = sentnr, sentlength = nwords, fixatedword = currword, wordid = wordids[currword], fixationduration = fixdur, wordids = wordids))
	
	return scanpath
end

# ╔═╡ 27931fb1-ac52-4260-9a34-5c72b06a447f
# True parameter settings for generating data
p0 = (μ = 220.0, β = 0.6, ι = 0.4, sd = 0.5, ν = 0.25, r = 7.0)

# ╔═╡ b6b6c999-94fc-4415-ab2c-84c6c7e34785
simsent(10, p0, nlogfreqs, 1)

# ╔═╡ 7331008d-e7bf-44b0-97bf-c3105fe234c7
"""
Simulate an experiment.
"""
function simexp(nwordrange, nsents, worddict, ps)
	# Preallocating space to save data
	expdata = scanpath = Array{NamedTuple}(undef, 0)
	sizehint!(scanpath, maximum(nwordrange) * nsents)
	for i in 1:nsents
		nwords = rand(nwordrange)
		push!(expdata, simsent(nwords, ps, worddict, i)...)
	end
	return DataFrame(expdata)
end

# ╔═╡ 368a123d-3d50-4eaa-bb07-01a9ebf32c25
md"""
Generating a bunch of data: Notice that a lot of observations (500) are needed to get even mediocre MLEs for a non-hierarchical (= complete pooling) model. 1000 sentences is better.
"""

# ╔═╡ 7535e3bf-be16-43bc-9005-bd88b24c6722
begin
	sentlens = [10]  # 4:30
	nsents = 20 #5000
	data = simexp(sentlens, nsents, nlogfreqs, p0)
end

# ╔═╡ 21b563bd-70b3-4c63-9237-181c856369dd
md"""
## Setting up the likelihood function using Turing.jl

It's automatically differentiable now, which is a minor miracle that I'm pretty proud of.

Could also use Stan. 🤷
"""

# ╔═╡ 31add289-a8b9-4bd3-b961-16d560755721
@model function basemodel(nlogfreqs, data, fixatedwords, fixationdurations, ::Type{T} = Float64) where {T}
	# Priors
	μ ~ LogNormal(log(0.2), 0.75) #LogNormal(log(200), 0.75)
	sd = 0.5 * one(T) # ~ Exponential() # 
	ν ~ Beta(1.5, 1.5) # Exponential() # 0.6 * one(T)
	η = -3.0#~ truncated(Normal(-3, 1), -Inf, 0) # -3.0 * one(T) # 
	β ~ Beta(1.5, 1.5)
	r ~ LogNormal(log(5), 0.75) #one(T) #
	γ = 1.0 #~ truncated(Normal(1, 0.5), 0, Inf) # = one(T)
	#ι ~ Exponential() #truncated(Normal(1, 0.5), -0.5, Inf) # zero(T)
	#κ ~ Exponential() #Exponential(5.0) # zero(T)
	
	σ = inv(1 + 2*ν + ν^2)
	minsal = 10^η
	shapeparam = 9.0
	rate0 = shapeparam / μ

	#Threads.@threads for pi in groupby(data, [:subject, :trial_id])
	Threads.@threads for pi in data
		nwords = pi[1, :sentlength]
		nfix = nrow(pi)
		#itemnr = pi.sentenceid[1]
		#maxact = maxactivations[corpus.itemidseq .== itemnr]
		maxact = 1 .- β .* [nlogfreqs[k] for k in pi[1, :wordids]]
		
		activations = zeros(T, nwords)
		saliencies = zeros(T, nwords)
		probabilities = zeros(T, nwords)
		λ = zeros(T, nwords)
		
		for row in eachrow(@view pi[1:end-1, :])
			rowidx = parentindices(row)[1]
			currword = row.fixatedword
			# Update proc. rates
			update_proc_rates!(λ, currword, ν, σ)
			update_activations!(activations, r, λ, row.fixationduration, maxact)
			update_saliencies!(activations, saliencies, maxact, minsal)
			update_probabilities!(probabilities, saliencies, γ)
	
			# Spatial LL
			if !isprobvec(probabilities)
				Turing.@addlogprob! -Inf
			else
				fixatedwords[rowidx+1] ~ Categorical(probabilities)
			end
	
			# Temporal LL
			nextword = fixatedwords[rowidx+1]
			#fixationdurations[rowidx+1] ~ LogNormal(log(μ), sd)
			#currrate = inv(rate0 * (1 + ι*activations[nextword]))
			currrate = inv(rate0)
			if currrate <= zero(currrate)# + 0.0001
				#currrate += eps(typeof(currrate))
			#end
				Turing.@addlogprob! -Inf
			#	println(currrate)
			else
				fixationdurations[rowidx+1] ~ Gamma(shapeparam, currrate)
			end
		end
	end
end

# ╔═╡ 16c96964-b4d6-4e4c-98b4-1050cd6a3193
md"""
- Separate temp. & spatial LL terms
- Fix γ = 1, κ only relevant for parsing, but would need to refomulate
- Calculate LL per word: divide by number of words
- Prob. need effect on temp. LL to be able to detect (probably)
  - Don't necessarily need to fit all params at once
- If can find syn. proc. that affects temporal likelihood, then that can form basis fo future work.
- Set up on GitUP
- Mo. in two wks. 04.03. 10:00
"""

# ╔═╡ f84eefbd-b42d-495c-9e59-487db1208241
gdata = groupby(data, :sentnr);

# ╔═╡ 4938efe4-7f03-46b1-8087-bd2e94363922
mod = basemodel(nlogfreqs, gdata, data.fixatedword, data.fixationduration);

# ╔═╡ 06301bab-8245-44ad-90b9-8a1310190f06
md"# Basic MLE stuff"

# ╔═╡ bfe53851-3423-4c7a-8e84-79412ed92a72
mle0 = optimize(mod, MLE(), LBFGS(alphaguess=0.01); autodiff = :forward)

# ╔═╡ 28b6d8d4-b047-472e-8892-bc31365ac57f
paramnames = DynamicPPL.syms(DynamicPPL.VarInfo(mod))

# ╔═╡ 5b64b6e8-5f8a-4f59-9271-9378ba237aeb
[(p, mle0.values.array[i], p0[p]) for (i, p) in enumerate(paramnames)]

# ╔═╡ 33e1c158-caf2-48cd-9852-1c304ec0ab15
md"""
At first glance, the MLE estimates don't look terrible, at least for some parameters. ι and κ seem fishy.
"""

# ╔═╡ fc6b5dbe-f855-4e6e-b0f6-d26328cbc10a
# kinda meaningless b/c based on normal approximation of numerical approximation of Hessian
try
	coeftable(mle0)
catch DomainError
	@warn "Bad matrix"
end

# ╔═╡ 6877bc24-c236-4737-922e-9c427d08b204
md"""
## Bias and coefficient of variation

Estimates can be biased when parameters are unidentifiable. A high CV indicates a lot of uncertainty relative to the parameter's magnitude: hard to tell signal from noise.
"""

# ╔═╡ 4a392913-714a-43fb-ae74-c5e17b3f2647
# Some of these are quite biased...
relbias = [(p, abs(mle0.values.array[i] - p0[p]) / abs(p0[p])) for (i, p) in enumerate(paramnames)]

# ╔═╡ 36e45590-40b6-4de6-b2d2-7eab5135824d
md"""
# Creating profile likelihoods
"""

# ╔═╡ fbd630d0-a434-4e0a-ac74-de74b1d4ec1e
function profll(mod, var, vals, inits)
	mles = zeros(size(vals, 1))
	prevmle = zeros(length(inits))
	for (i, v) in enumerate(vals)
		if i == 1
			omle = optimize(fix(mod, (; var => v)), MLE(), inits, LBFGS(alphaguess=0.01); autodiff=:forward)
			mles[i] = omle.lp
			prevmle = omle.values.array
		else
			omle = optimize(fix(mod, (; var => v)), MLE(), prevmle, LBFGS(alphaguess=0.01); autodiff=:forward)
			mles[i] = omle.lp
			prevmle = omle.values.array
		end
	end
	return mles
end

# ╔═╡ 180af191-1fbd-410e-84aa-97485241f961
function allprofs(mod, vals, inits)
	@assert size(vals, 1) == length(inits) "Number of ranges must match number of variables"
	vars = DynamicPPL.syms(DynamicPPL.VarInfo(mod))
	mles = OrderedDict(v => Float64[] for v in vars)
	for (i, v) in enumerate(vars)
		#@show("Currently working on $v")
		mles[v] = profll(mod, v, vals[i], inits[eachindex(inits) .∉ [i]])
	end
	return mles
end

# ╔═╡ 14d9a0ff-a785-46c7-a04e-f969abab078e
paramnames

# ╔═╡ 659f89b3-4799-4623-9ca5-7b19871d6d57
md"""
# Current issue: Gamma likelihood

possibly rescale to be on second scale instead of ms?
"""

# ╔═╡ 351428ab-a719-4b53-aa11-cee69f5ba187
begin
	nvals = 40
	vals = [range(150, 350, nvals),#range(0.15, 0.35, nvals), #
	range(0.01, 1, 10+nvals), range(0, 0.999, nvals), range(0.01, 30, 10+nvals)]#, range(0.0, 2, nvals)]
	lls = allprofs(mod, vals, mle0.values.array)
end

# ╔═╡ 9c37b9aa-4d58-4c40-a2cb-dc5c5a0fc9d6
llsnorm = OrderedDict(k => v ./ nrow(data) for (k, v) in lls)

# ╔═╡ 69d1353a-0bf1-4937-ab7a-8bcbd820ca46
#function plotprofs(lls, vals, cutoff = nothing, mles = nothing, trues = nothing)
function plotprofs(lls, vals, mles = nothing, trues = nothing)
	p = plot(layout=(ceil(Int, length(lls) // 2), 2), margin=1*Plots.mm, size=(680, 800))
	for (i, v) in enumerate(keys(lls))
		plot!(p[i], vals[i], lls[v], xlabel=String(v), label=nothing)
		#p = plot(vals[i], lls[v], xlabel=String(v), label=nothing)
		if mles != nothing
			vline!(p[i], [mles[i]], label="MLE")
		end
		if trues != nothing
			vline!(p[i], [trues[v]], label="True value")
		end
	end
	p
end

# ╔═╡ 850ac1af-db1f-4b34-b9d8-e1c3bf48bc81
#cutoff = mle0.lp - 0.5 * quantile(Chisq(8), 0.95)
cutoff = mle0.lp ./ nrow(data) - 0.5 * quantile(Chisq(8), 0.95)

# ╔═╡ 622c0eed-324a-4e8c-ba19-f85fd8040446
[v for (v, x) in lls if any(x .>= cutoff)]
#[v for (v, x) in lls if any(x .>= log(1/15))] # Fisher's suggestion, Pawitan 2001

# ╔═╡ 4b3798d6-3616-4351-8996-89f2279b3831
#vals[3][lls[:η] .>= cutoff]

# ╔═╡ 1d872b00-3025-45c5-ad9e-2dd765af418b
profllplot = plotprofs(lls, vals, mle0.values.array, p0)

# ╔═╡ 1c9c2f9f-e02a-4d4b-b184-2530931916d6
p0

# ╔═╡ 3b44d760-9868-47eb-adf0-35b1387b9e96
vals

# ╔═╡ 2b071d65-441d-4f71-841d-76074be9a2b5
begin
	mus = [loglikelihood(fix(mod, (; ν = p0.ν, β = p0.β, ι = p0.ι, r = p0.r)), (; μ = m)) for m in vals[1]]# ./ nrow(data)
	nus = [loglikelihood(fix(mod, (; μ = p0.μ, β = p0.β, ι = p0.ι, r = p0.r)), (; ν = m)) for m in vals[2]]# ./ nrow(data)
	betas = [loglikelihood(fix(mod, (; ν = p0.ν, μ = p0.μ, ι = p0.ι, r = p0.r)), (; β = m)) for m in vals[3]]# ./ nrow(data)
	rs = [loglikelihood(fix(mod, (; ν = p0.ν, β = p0.β, ι = p0.ι, μ = p0.μ)), (; r = m)) for m in vals[4]]# ./ nrow(data)
	#iotas = [loglikelihood(fix(mod, (; ν = p0.ν, β = p0.β, μ = p0.μ, r = p0.r)), (; ι = m)) for m in vals[5]]# ./ nrow(data);
end

# ╔═╡ f391ea1b-3a64-46e1-9b3b-107f403923bd
begin
	pp = plot(wsize=(680, 800), layout=@layout [x x; x x; x _])#, wsize=(680, 800))
	plot!(pp[1], vals[1] ,mus, label="Old way")
	plot!(pp[1], vals[1], lls[:μ], label="Correct", xlabel="μ")
	vline!(pp[1], [p0.μ], label="True value")
	vline!(pp[1], [mle0.values.array[1]], label="MLE")
	
	plot!(pp[2], vals[2] ,nus, legend=nothing)
	plot!(pp[2], vals[2], lls[:ν], legend=nothing, xlabel="ν")
	vline!(pp[2], [p0.ν], label="True value")
	vline!(pp[2], [mle0.values.array[2]], label="MLE")
	
	plot!(pp[3], vals[3] ,betas, legend=nothing)
	plot!(pp[3], vals[3], lls[:β], legend=nothing, xlabel="β")
	vline!(pp[3], [p0.β], label="True value")
	vline!(pp[3], [mle0.values.array[3]], label="MLE")
	
	plot!(pp[4], vals[4] ,rs, legend=nothing)
	plot!(pp[4], vals[4], lls[:r], legend=nothing, xlabel="r")
	vline!(pp[4], [p0.r], label="True value")
	vline!(pp[4], [mle0.values.array[4]], label="MLE")
	#=
	plot!(pp[5], vals[5] ,iotas, legend=nothing)
	plot!(pp[5], vals[5], lls[:ι], legend=nothing, xlabel="ι")
	vline!(pp[5], [p0.ι], label="True value")
	vline!(pp[5], [mle0.values.array[5]], label="MLE")
	=#
end

# ╔═╡ 0d51f279-918c-47d8-868c-a297870b1f34
md"""
What do we see in the likelihood profiles? First, it seems like there are relatively clear peaks with finite confidence intervals for most parameters. There were *many* warnings about the converges of the optimiation, though, so we should not put too much trust on the profiles.

In the slopiness analyses, we will see that the likelihood profiles are not painting the whole picture. What seem like isolated peaks are actually multi-parameter ridges of approximately equal likelihood. This will make parameter fitting difficult.
"""

# ╔═╡ 3dc7a293-477b-4732-9139-800b72bedb5e
md"""
# Sloppiness

Sloppiness is indicated by a wide spread in the eigenvalues of the Hessian of the likelihood or the Fisher information matrix. In addition, the eigenvalues exhibit a more or less uniform distribution in their orders of magnitude.

I'll focus on the Hessian of the log likelihood, but the Fisher information matrix provides similar results (below).

Now we look at the Hessian matrix, which should ideally have a rank of 8 and and be positive definite if we are to have hope for this model.

## Analyses based on the Hessian
"""

# ╔═╡ aefd44db-7881-4093-b738-33b0611a8eca
"""
Returns the Hessian of the likelihood.
"""
function get_hessian(m)#, θ=nothing)
	linked = DynamicPPL.istrans(m.f.varinfo)
	if linked
        Turing.Setfield.@set! m.f.varinfo = DynamicPPL.invlink!!(m.f.varinfo, m.f.model)
    end
	# Calculate the Hessian.
    varnames = StatsBase.coefnames(m)
	# To do it on the deviance scale:
    #H = ForwardDiff.hessian(x -> -2 * m.f(x), m.values.array[:, 1])
	H = ForwardDiff.hessian(m.f, m.values.array[:, 1])
	# Link it back if we invlinked it.
    if linked
        Turing.Setfield.@set! m.f.varinfo = DynamicPPL.link!!(m.f.varinfo, m.f.model)
    end
	return H
end

# ╔═╡ e0849631-8ced-413d-8930-8508a2460e3b
H = get_hessian(mle0)

# ╔═╡ 51f3c3e4-001e-4bf6-bce0-e0c3450be7f6
round.(H; digits=3)

# ╔═╡ e7ea41bf-8fc3-4d34-950e-86ca9076a470
rank(H), cond(H), rank(H, rtol=0.001), isposdef(H)

# ╔═╡ 08cdba3b-57ca-49d4-aa44-3f29289efdd3
md"""
For very small tolerances, the Hessian is full rank, but the condition number is huge, which means it is actually ill-conditioned. When we set a larger tolerance for the rank, we see that it is not actually full rank. Also, the Hessian isn't positive definite, so we can expect problems.

Now let's look at the eigendecomposition of the Hessian.
"""

# ╔═╡ ee8b1508-2e8b-48b4-bdf6-e041365440ec
evals, evecs = eigen(H)

# ╔═╡ b24c109c-6e05-461a-bd75-b40f7cf9c7d4
# Is it sloppy?
(minimum(evals) / maximum(evals)) < 0.001

# ╔═╡ c78882f3-dbb4-4bc9-a50f-2e76d2f2d7b7
scatter(ones(8), log10.(abs.(evals)), xlims=(0.5, 1.5), xticks=nothing, ylabel="Order of magnitude of eigenvalues", legend=false, marker=:rect, markersize=5)

# ╔═╡ 699e035f-b40f-4c2e-8de9-cca4e5ea00f0
md"""
The model is sloppy by the heuristic that the eigenvalues span at least three orders of magnitude. They are also approximately uniformly distributed across orders of magnitude.

What this indicates is that there are ridges in the likelihood landscape where huge changes to parameter values have no effect on the likelihood/model output. Put another way, highest-likelihood hyper-ellipsoids are highly anisotropic. This can be seen by looking at the eigevectors themselves, which encode which (combinations of) parameters that "move" together.

The smallest eigenvalue corresponds to the "sloppiest" eigenvector, here, the first column. In this eigendirection, large changes in the parameter(s) leads to very small changes in the model output. The largest eigenvalue corresponds to the stiffest eigenvector, where small changes in that eigendirection lead to large changes in the model output.
"""

# ╔═╡ b3ed302a-3ed9-41cc-a9fd-58a5a0c8ae96
round.(evecs; digits=1) .+ 0

# ╔═╡ ec891f2b-c9fb-4eb7-a404-ca3bdbac4ca6
md"""
Thinking about practical identifiability, when there are really small eigenvalues, this means that there will be (combinations of) parameters that have large standard errors given this data set, making them hard to identify.
"""

# ╔═╡ c9654f02-f149-497c-b8f1-7196e3f42ab0
# Approx. standard errors of each of the eigenparameters
plot(sqrt.(abs.(evals)).^-1, yscale=:log10, legend=false, xticks=1:8, xlabel="Eigenparameter index", ylabel="Order of magnitude of SE of eigenparameter", marker=:o)

# ╔═╡ 5e5770ef-2741-4e8d-88c2-e6d75a299f88
md"""
We can also calculate the observed information matrix, which is a sample-based approximation of the (expected) Fisher information matrx below.
"""

# ╔═╡ 7dac6ba9-c635-40ae-8159-dab062147747
# Observed information matrix
# How much info does one variable give about another?
oim = -H

# ╔═╡ ce36e74e-9a00-4720-a801-69010ec196aa
# CVs ⪆ 1, too much variability for this sample size
# Note: using the absolute value might be weird
try
	cv = [(p, abs(diag(oim)[i])^-0.5 / abs(mle0.values.array[i])) for (i, p) in enumerate(paramnames)]
catch
	@warn "Bad matrix"
end

# ╔═╡ 24b19308-770c-4ec0-bbf0-49daf746d148
# Asymptotic VCov
#round.(inv(bunchkaufman(Hermitian(oim))); digits=3)
round.(inv(oim); digits=3)

# ╔═╡ f1e9a8a2-eda5-40a3-b990-4686c07057fd
isposdef(oim), cond(oim)

# ╔═╡ 0bdb769e-37b7-4ff8-9fbc-16ed043758e5
paramnames

# ╔═╡ 41cd5857-e991-409f-b6d2-528d4e120cf3
fevals, fevecs = eigen(oim)# - 1 * I)

# ╔═╡ 4e69d83d-0e3a-4b09-a26e-29a498227bfa
round.(fevecs, digits=2)

# ╔═╡ 254c5889-7ed5-4958-8b7a-d671cfb9e937
md"""
This is interesting: Right now, the smallest magnitude eigenvalue is associated with the eigenvector that points along a hyperplane that basically just includes μ, ι, and κ. The eigenvalue is small along this direction meaning that big moves along the hyperplane only have a small effect on the Hessian. The next two eigenpairs tell basically the same story.

The largest-magnitude eigenvalue is associated with the last eigenvector. This eigenvector has the biggest movement along ν: A small change leads to a big change in the overall Hessian ∴ optimization.

While sloppiness isn't exactly equivalent to structural or practical non-identifiability, it is related: as [@rama2017delineating] note, the inverse of the negative of the Hessian is a lower bound on the covariance matrix of the parameters, so elements in that matrix that are large suggest where there might be difficulty identifying parameters.

Here, it's also important to note the following: Above, in the likelihood profiles, μ, e.g., *seems* well identified. It has a clear peak. This is backed up by the observed information matrix, which has a not tiny value for μ with itself: $(round(oim[1, 1]; digits=3)). But when we look at the sloppiness analysis, we see that the peak is trügerisch. There is actually a whole manifold of μ-ι-κ pairings that all give the approximately the same likelihood.
"""

# ╔═╡ aa4d7850-4d36-4aa7-84f2-98da00fb4847
md"""
## Expected Fisher information

Need to calculate the average of the Hessian over all individual data points. Here, I consider a data point a whole sentence and estimate the Hessian using the outer product of the Jacobian with itself for each sentence. This evaluated at the MLE, following [@brown2003statistical].

This gives similar results to the Hessian at the MLE.
"""

# ╔═╡ 3af274c3-9f47-4ebc-8394-962664621d95
paramnames

# ╔═╡ 430a6859-050b-41dd-8322-abb95f8f17af
size(mle0.values.array)

# ╔═╡ d0cb5728-f408-4552-bdc6-f2023569b372
function efim(gdata, mle)
	I = zeros(size(mle, 1), size(mle, 1))
	for i in 1:length(gdata)
		mod = basemodel(nlogfreqs, gdata[i:i], data.fixatedword, data.fixationduration)
		∇𝑙 = ForwardDiff.gradient(x -> loglikelihood(mod, (μ = x[1], ν = x[2], β = x[3], r = x[4])), mle0.values.array) # , ι = x[5]
		I .+= ∇𝑙 * ∇𝑙'
	end
	return I ./ length(gdata)
end

# ╔═╡ d9489f67-e538-4e78-abe1-7d7890157538
fim = efim(gdata, mle0.values.array)

# ╔═╡ 568014d5-9d25-408f-9f8f-cc95d9ec1e92
# Still ill-conditioned
cond(fim), rank(fim, rtol=0.01)

# ╔═╡ 040132cf-31c5-47a7-8af9-43986fcac947
Ivals, Ivecs = eigen(fim)

# ╔═╡ 569f198c-03ad-4b48-9a6a-d8266e5ce64d
# Is it sloppy?
minimum(Ivals) / maximum(Ivals) < 0.001

# ╔═╡ c416d1b9-f9c7-45e2-91e3-c3c752c3a8c5
round.(Ivecs ./ sum(abs.(Ivecs), dims=1); digits=2)

# ╔═╡ 6a03a2cb-addc-47fd-a5a9-fefcad0f10b9
paramnames

# ╔═╡ 49db9667-63dd-4e68-8435-3e307701f432
nullspace(fim; rtol=0.001)

# ╔═╡ 8acfe19f-6e96-4b27-a3d9-155e949d3dc1
isposdef(fim)

# ╔═╡ 02c99f24-7847-45c6-83ff-773575b103f2
md"""
# Finding eliminable parameter combinations
"""

# ╔═╡ c020cc25-58ad-4e87-b825-a264ae76fe23
md"""
## Trying out MBAM

...Or at least a half-assed version of it...
1. Start at θ = MLE
2. Find smallest eigenvalue λ of Hessian evaluated at θ
3. Set θ → θ + eigenvector associated with λ
4. Repeat from 2 until λ very, very small


It looks like this really naïve implementation is not working super great. Probably need to incorporate all of the geodesic stuff...

Actually, it seems fine. It is a really naïve implementation, but just following the smallest eigenvector does do something non-stupid, it seems.

**Note:** Now that the estimation procedure for the actual Fisher information matrix is working, it's possible it more properly implement the MBAM, but that will have to happen later.
"""

# ╔═╡ 2cf8d8fc-383a-4684-b846-bc78911f0ae9
function mbam(m, niter=20)
	linked = DynamicPPL.istrans(m.f.varinfo)
	if linked
        Turing.Setfield.@set! m.f.varinfo = DynamicPPL.invlink!!(m.f.varinfo, m.f.model)
    end
	vhist = zeros(niter, size(evecs, 1))
	θ = copy(m.values.array)
	for i in 1:niter
		#J = ForwardDiff.gradient(m.f, θ)
		H = ForwardDiff.hessian(m.f, θ)
		#H = ForwardDiff.hessian(x -> -2 * m.f(x), θ)
		evals, evecs = eigen(H)
		λ = minimum(evals) # evals[3]
		println(λ)
		v = evecs[:, argmin(evals)] # 3]#
		θ .+= 0.01 .* v # 0.0001 .*
		#θ .-= inv(H) * J
		vhist[i, :] = v
	end
	# Link it back if we invlinked it.
    if linked
        Turing.Setfield.@set! m.f.varinfo = DynamicPPL.link!!(m.f.varinfo, m.f.model)
    end
	return vhist
end

# ╔═╡ 446a5091-5118-4595-93a2-61a1682b2d7f
#v = mbam(mle0, 20);

# ╔═╡ b7faa27b-c7bb-4c17-ae4f-f85fc718eed3
anim = @animate for i ∈ 1:size(v, 1)
    plot(v[i, :], label="$i / $(size(v, 1))", ylims=(-1, 1), xticks=(1:8, paramnames))
end

# ╔═╡ c28473b8-4e96-4944-884b-bb9cb023307a
gif(anim, fps=10)#, loop=0)

# ╔═╡ 2db58b76-4593-480d-8d86-2c8b47c855b8
md"""
## Trying out minimally disruptive curves

This is a method for finding potentially nonlinear dependencies between the model parameters [@raman2017delineating].
"""

# ╔═╡ ad0919ac-24c1-4226-8d37-c4f5c6353e16
paramnames

# ╔═╡ 10f401ee-d629-451c-977f-1395071a40cc
# ╠═╡ disabled = true
#=╠═╡
function loss(ps)
	return -2 * loglikelihood(mod, (μ = ps[1], ν = ps[2], η = ps[3], β = ps[4], r = ps[5], γ = ps[6], ι = ps[7], κ = ps[8]))
end
  ╠═╡ =#

# ╔═╡ 9b272218-ff5c-4c76-ada9-2d8d7d5db965
mle0.lp

# ╔═╡ b28803a9-15f9-4c3b-9992-83a4c30f6d6b
#=╠═╡
loss(mle0.values.array) * -0.5
  ╠═╡ =#

# ╔═╡ 90e9fb47-b239-47fd-b895-0833092aa79b
function loss(ps, g)
	g[:] = ForwardDiff.gradient(loss, ps)
	return loss(ps)
end

# ╔═╡ ced3459e-7171-4364-bdb5-c7b75b5156c4
pinit = NamedTuple{filter(!=(:sd), keys(p0))}(p0)

# ╔═╡ cf4b44d6-39a0-4215-b14b-1ea829c89fd8
#=╠═╡
loss(collect(values(pinit)), zeros(8))
  ╠═╡ =#

# ╔═╡ 4b375efd-6bb4-4123-a148-66f7f9cdff75
#=╠═╡
H2 = ForwardDiff.hessian(loss, mle0.values.array)
  ╠═╡ =#

# ╔═╡ a8c30639-e396-41b4-b201-86149154a339
#=╠═╡
_, h2e = eigen(H2)
  ╠═╡ =#

# ╔═╡ 186555bf-8371-4fd6-8685-01d1f4194aef
#=╠═╡
round.(h2e; digits=3)
  ╠═╡ =#

# ╔═╡ 9a56b94c-1176-44e1-9ad1-ca25ebb324cd
paramnames

# ╔═╡ 0e0a49fa-327f-439c-bdd7-42cea19f89d8
#=╠═╡
initdir = h2e[:, 2]
  ╠═╡ =#

# ╔═╡ 88561bd2-4cd9-412e-95ab-8958446cc5a8
span = (-100.0, 100.0)

# ╔═╡ 03087d13-d5a7-450c-b702-659b5c6585b8
mmtm = 100.0

# ╔═╡ bb64c497-b222-4810-8d76-e33b150731b4
#=╠═╡
eprob = MDCProblem(loss, mle0.values.array, initdir, mmtm, span);
  ╠═╡ =#

# ╔═╡ 69643664-8a81-438d-8a16-8ad68816d8ee
#=╠═╡
mdc = evolve(eprob, OrdinaryDiffEq.Tsit5)
  ╠═╡ =#

# ╔═╡ a5efafc4-d7a4-4a73-aabc-16db3f647f70
#=╠═╡
plot(mdc; pnames = collect(String.(paramnames)))
#plot([1 2 3], [2 4 6])
  ╠═╡ =#

# ╔═╡ 447f0ddd-143c-4805-99dc-0ecf120e6ba9
#=╠═╡
begin
	cost_vec = [loss(el) for el in eachcol(trajectory(mdc))]
	plot(distances(mdc), log.(cost_vec), ylabel="log(cost)", xlabel="distance", title="cost over MD curve");
end
  ╠═╡ =#

# ╔═╡ 2a3ffcc0-d851-404c-9f08-90336759afed
#=╠═╡
cost_vec, distances(mdc)
  ╠═╡ =#

# ╔═╡ 2532de3c-77f7-4236-bee3-e5fbe1b86053
md"""
## Results of minimally distruptive curves

The smalles eigendirections affect μ, ι, and κ. Given the equations of the model, this isn't too suprising.
"""

# ╔═╡ b7c890c3-75aa-419c-9e11-c229fa291cfd
md"""
# Conclusions

**All of these discussion points should be taken with a grain of salt! The optimization procedures they're based on rarely converge properly due to the terribly ill-conditioned-ness of the problem!**

**Also, different true parameter settings highlight different aspects of the non-identifiability/sloppiness.**

## MLE

Maybe surprisingly, we were able to get warning-free MLEs. But the lack of warnings is hiding an ill-conditioned situation.

The relative biases of the parameters are also interesting. Most were less than 3%, but that's still kinda high given that there $(nrow(data)) fixations in the data set.

The coefficients of variation[^1] were much larger than 1, with the exception of μ and ι. Interestingly, ι had relatively high bias (~80%), but one of lowest CVs. Maybe the correlations with other parameters constrained it a bit.

Something interesting is that many parameters have MLEs close to the true value. But mostly these are *still* not identifiable due to the limitations caused by their CIs. This is truly weird.

## Profile likelihood

The profile likelihoods make a similar point: Most of the parameters have at least onesidedly infinite CIs.
- r and κ are structurally non-identifiable.
- ι and ν have profile LL maxima at the edge of the possible range.
- μ, η, β, and γ have somewhat clear LL peaks that seem to be above the cutoff.
  - η and γ have flat profiles, though, and are nearly perfectly correlated (≈ 1)

It's also important to note that optimization often did not converge when calculating the profile likelihoods. That suggests that there are no easily identifiable maxima when holding particular parameter values constant. This seems especially clear for r and κ.

## Sloppiness

Depending on how you calculate it, the Hessian can be of lower rank than the number of parameters. In addition, its condition number is practically infinite, and the ratio of the smallest to largest eigenvalues is larger than 0.001; this suggests that the model is sloppy, meaning there is a weak link between parameter settings and outputs.

When the number of degrees of freedom in the data is smaller than the number of (effective) parameters, sloppiness results [@transtrum2010why]. The data we are using is effectively two-dimensional: (fixated word, fixation duration), while there are eight parameters. The eigenvalue analysis suggest there are 3-4 parameter (-s/combinations) that govern the models predictions, which is still larger than the number of data dimensions, so sloppiness seems inevitable.

Maybe a more correct way of putting it is this: When different data points can be approximated/interpolated by different parameter settings in the model. This means that the *effective* dimensionality of the model is reduced. And when the *actual* dimensionality (number of parameters) is greater than the effective number of parameter, we will have sloppiness.

## Effect of data set size

When there is not enough data, basically all of the parameters become practically non-identifiable: 50 sentences, e.g., is not enough to reliably identify any of the parameters. Even 500 sentences is not really enough. This doesn't bode well for fitting to experimental data.

[^1]: When standard errors can be calculated. This often fails b/c the various matrices are ill-conditioned/non-invertable.
"""

# ╔═╡ cf923955-0ca1-467a-a1b7-8227bc9fd6a7
md"""
# Bayesian alternative, similar results?

The analyses below show similar results to the likelihood-based results above. Instead of calculating the numerical Hessian of the likelihood, we calculate the variance-covariance matrix of the posterior parameter estimates, invert that, and then do the same sloppy/stiff analysis.

There seem to be a lot of different, overlapping colinearities in the posterior.

This seems to be at least a little different from what the likelihood-only analysis showed. ⇒ Effect of prior? The priors are pretty tight, so that might be dominating the posterior...

One interesting thing about sloppy models is that *individual* parameters are hard to estimate, but combinations can be easy to estimate. This might be why the posteriors for most parameters look alright; they mutually constrain each other to a reasonable range.
"""

# ╔═╡ 4617b5ad-ffe9-4900-89a1-9df1d2c33850
# ╠═╡ disabled = true
#=╠═╡
#post = sample(mod, NUTS(), 100)
post = sample(mod, Emcee(20), 1000)
  ╠═╡ =#

# ╔═╡ bd93ed14-3e7f-40a3-b49c-ade4e9bb49c4
#=╠═╡
describe(post)
  ╠═╡ =#

# ╔═╡ 96c1f4cf-31cd-49fb-80ce-685754d4b8e4
#=╠═╡
BH = inv(cov(reshape(permutedims(post[501:end, 1:8, :].value.data, (3, 2, 1)), (10000, 8))))
#BH = inv(cov(dropdims(post.value.data[:, 1:8, :]; dims=3)))
  ╠═╡ =#

# ╔═╡ b274cc04-8a0c-4d0a-88f2-1f8fccb1aaf1
#=╠═╡
peval, pevec = eigen(BH)
  ╠═╡ =#

# ╔═╡ 08c7003a-8a7a-4409-8c3d-377df46ff1e5
#=╠═╡
round.(pevec; digits=1)
  ╠═╡ =#

# ╔═╡ dbdcefa2-8e8c-4672-8390-f9174c55c991
paramnames

# ╔═╡ c5978128-40d1-4fc6-adce-df98878a8b90
p0

# ╔═╡ a072df1c-fe6b-4250-81e5-a6312b2dfac8
#=╠═╡
#UpperTriangular(cor(dropdims(post.value.data[:, 1:8, :]; dims=3)))
UpperTriangular(cor(reshape(permutedims(post[501:end, 1:8, :].value.data, (3, 2, 1)), (10000, 8))))
  ╠═╡ =#

# ╔═╡ e10168a4-c967-4750-8487-5c8bbdd4807b
#=╠═╡
plot(post)
  ╠═╡ =#

# ╔═╡ dbfa8a0b-62ae-4a66-9aa2-100a7939db5b
#=╠═╡
begin
	#pp = plot(size=(800, 800), layout=3)#@layout([x x; x _]))
	p1 = marginalkde(post[:μ][:], post[:ι][:], ylabel="ι")
	p2 = marginalkde(post[:μ][:], post[:κ][:], xlabel="μ", ylabel="κ")
	p3 = marginalkde(post[:κ][:], post[:ι][:], xlabel="κ")
	plot(p1, p3, p2)
end
  ╠═╡ =#

# ╔═╡ 2f6041c2-06ea-41ea-8582-65361da47d63
md"""
- Bader & Meng design for long-distance dependencies
  - Boiled-down self-avoiding walk idea for syntax-driven regressions
- Mtg. w/ Ralf to simplify SWIFT as much as possible to minimize sloppiness

---

- Abs.: present as method from sys.bio.: Goal is spell out methods
- Can make slides for retreat, Beamer or Keynote for march PI retreat

---
- Platform for model comparison
- Std. data sets, cf. scene viewing lit.
- Similar to benchmark datasets in compling
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
DynamicPPL = "366bfd00-2699-11ea-058f-f148b4cae6d8"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
OrderedCollections = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
DataFrames = "~1.6.1"
Distributions = "~0.25.107"
DynamicPPL = "~0.24.4"
ForwardDiff = "~0.10.36"
Optim = "~1.7.8"
OrderedCollections = "~1.6.3"
Plots = "~1.39.0"
PlutoUI = "~0.7.55"
SpecialFunctions = "~2.3.1"
StatsBase = "~0.34.2"
StatsPlots = "~0.15.6"
Turing = "~0.30.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "2ad80c6c117df53cc74200730236f96b1c458be2"

[[deps.ADTypes]]
git-tree-sha1 = "41c37aa88889c171f1300ceac1313c06e891d245"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.6"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "FillArrays", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "63ae0647e8db221d63256820d1e346216c65ac66"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "5.0.0"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "DensityInterface", "Random", "Setfield", "SparseArrays"]
git-tree-sha1 = "917ad8da4becae82028aba80b7e25197f0c76dd1"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.7.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Test"]
git-tree-sha1 = "cb96992f1bec110ad211b7e410e57ddf7944c16f"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.35"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "ProgressMeter", "Random", "Requires", "Setfield", "SimpleUnPack", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "dfa0e3508fc3df81d28624b328f3b937c1df8bc2"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.6.1"

    [deps.AdvancedHMC.extensions]
    AdvancedHMCCUDAExt = "CUDA"
    AdvancedHMCMCMCChainsExt = "MCMCChains"
    AdvancedHMCOrdinaryDiffEqExt = "OrdinaryDiffEq"

    [deps.AdvancedHMC.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "FillArrays", "LinearAlgebra", "LogDensityProblems", "Random", "Requires"]
git-tree-sha1 = "1cc336be36fef7df68473a7d0d60ebba25958b9e"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.8.0"
weakdeps = ["DiffResults", "ForwardDiff", "MCMCChains", "StructArrays"]

    [deps.AdvancedMH.extensions]
    AdvancedMHForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AdvancedMHMCMCChainsExt = "MCMCChains"
    AdvancedMHStructArraysExt = "StructArrays"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Random", "Random123", "Requires", "StatsFuns"]
git-tree-sha1 = "672f7ce648e06f93fceefde463c5855d77b6915a"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.5.4"
weakdeps = ["Libtask"]

    [deps.AdvancedPS.extensions]
    AdvancedPSLibtaskExt = "Libtask"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "1f919a9c59cf3dfc68b64c22c453a2e356fca473"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.2.4"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "bbec08a37f8722786d87bedf84eae19c020c4efa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRules", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "199dc2c4151db557549a0ad8888ce1a60337ff42"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.13.8"

    [deps.Bijectors.extensions]
    BijectorsDistributionsADExt = "DistributionsAD"
    BijectorsForwardDiffExt = "ForwardDiff"
    BijectorsLazyArraysExt = "LazyArrays"
    BijectorsReverseDiffExt = "ReverseDiff"
    BijectorsTrackerExt = "Tracker"
    BijectorsZygoteExt = "Zygote"

    [deps.Bijectors.weakdeps]
    DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "237cb6c8e662b7648f8ccdb336e0d8faedfc24b8"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.58.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "c1deebd76f7a443d527fc0430d5758b8b2112ed8"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.19.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"
weakdeps = ["InverseFunctions"]

    [deps.ChangesOfVariables.extensions]
    ChangesOfVariablesInverseFunctionsExt = "InverseFunctions"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "75bd5b6fc5089df449b5d35fa501c846c9b6549b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.12.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"
weakdeps = ["ChainRulesCore", "DensityInterface", "Test"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "Distributions", "FillArrays", "LinearAlgebra", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "d61f08c7bd15c5ab215fd7a2eb61c1ae15d8ff5e"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.53"

    [deps.DistributionsAD.extensions]
    DistributionsADForwardDiffExt = "ForwardDiff"
    DistributionsADLazyArraysExt = "LazyArrays"
    DistributionsADReverseDiffExt = "ReverseDiff"
    DistributionsADTrackerExt = "Tracker"

    [deps.DistributionsAD.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "Compat", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "LogDensityProblems", "MacroTools", "OrderedCollections", "Random", "Requires", "Setfield", "Test"]
git-tree-sha1 = "26c17df3ef86eec24851f00bfec25c734d6f0ad9"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.24.4"

    [deps.DynamicPPL.extensions]
    DynamicPPLChainRulesCoreExt = ["ChainRulesCore"]
    DynamicPPLEnzymeCoreExt = ["EnzymeCore"]
    DynamicPPLMCMCChainsExt = ["MCMCChains"]
    DynamicPPLZygoteRulesExt = ["ZygoteRules"]

    [deps.DynamicPPL.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "e611b7fdfbfb5b18d5e98776c30daede41b44542"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "2.0.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "73d1214fec245096717847c62d389a5d2ac86504"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.22.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9a68d75d466ccc1218d0552a8e1631151c569545"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "abbbb9ec3afd783a7cbd82ef01dcd088ea051398"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5fdf2fe6724d8caabf43b557b84ce53f3b7e2f6b"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.0.2+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "68772f49f54b479fa88ace904f6127f0a3bb2e46"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.12"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "4e0cb2f5aad44dcfdc91088e85dee4ecb22c791c"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.16"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "cb4619f7353fc62a1a22ffa3d7ed9791cfb47ad8"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.4.2"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "98eaee04d96d973e79c25d49167668c5c8fb50e2"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.27+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LRUCache]]
git-tree-sha1 = "5930ef949f30a9a947c69ef6b069c0b1aa27619d"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "345a40c746404dd9cb1bbc368715856838ab96f2"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.8.6"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "f9a11237204bc137617194d79d813069838fcf61"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.1"

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems", "Requires", "SimpleUnPack"]
git-tree-sha1 = "9c50732cd0f188766b6217ed6a2ebbdaf9890029"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.7.0"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADADTypesExt = "ADTypes"
    LogDensityProblemsADEnzymeExt = "Enzyme"
    LogDensityProblemsADFiniteDifferencesExt = "FiniteDifferences"
    LogDensityProblemsADForwardDiffBenchmarkToolsExt = ["BenchmarkTools", "ForwardDiff"]
    LogDensityProblemsADForwardDiffExt = "ForwardDiff"
    LogDensityProblemsADReverseDiffExt = "ReverseDiff"
    LogDensityProblemsADTrackerExt = "Tracker"
    LogDensityProblemsADZygoteExt = "Zygote"

    [deps.LogDensityProblemsAD.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"
weakdeps = ["ChainRulesCore", "ChangesOfVariables", "InverseFunctions"]

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "3b1ae6bcb0a94ed7760e72cd3524794f613658d2"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "6.0.4"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "6ea46c36b86320593d2017da3c28c79165167ef4"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.8"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "14bd8088cf7cd1676aa83a57004f8d23d43cd81e"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.9.5"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "900a11b3a2b02e36b25cb55a80777d4a4670f0f6"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.10"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "6d42eca6c3a27dc79172d6d947ead136d88751bb"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.10.0"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded64ff6d4fdd1cb68dfcbb818c69e144a5b2e4c"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.16"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "6a731f2b5c03157418a20c12195eb4b74c8f8621"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.13.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "01f85d9269b13fedc61e63cc72ee2213565f7a72"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.8"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "34205b1204cc83c43cd9cfe53ffbd3b310f6e8c5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.1"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "68723afdb616445c6caaef6255067a8339f91325"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.55"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "c860e84651f58ce240dd79e5d9e055d55234c35a"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.2"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "d7087c013e8a496ff396bae843b1e16d9a30ede8"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.38.10"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Roots]]
deps = ["Accessors", "ChainRulesCore", "CommonSolve", "Printf"]
git-tree-sha1 = "af540898b1e6ca7aa6ba7213c05052809c6c522a"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.1.0"

    [deps.Roots.extensions]
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "6aacc5eefe8415f47b3e34214c1d79d2674a0ba2"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.12"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "ArrayInterface", "ChainRulesCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FillArrays", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces", "ZygoteRules"]
git-tree-sha1 = "916b8a94c0d61fa5f7c5295649d3746afb866aff"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.98.1"

    [deps.SciMLBase.extensions]
    ZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "Lazy", "LinearAlgebra", "Setfield", "SparseArrays", "StaticArraysCore", "Tricks"]
git-tree-sha1 = "51ae235ff058a64815e0a2c34b1db7578a06813d"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.7"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "f68dd04d131d9a8a8eb836173ee8f105c360b0c5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.1"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "1b0b1205a56dc288b71b1961d48e351520702e24"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.17"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymbolicIndexingInterface]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "f8ab052bfcbdb9b48fad2c80c873aa0d0344dfe5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.2.2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5c942be30a85ac75d14e9e527b55504031e1bbd3"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.31"
weakdeps = ["PDMats"]

    [deps.Tracker.extensions]
    TrackerPDMatsExt = "PDMats"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "3064e780dbb8a9296ebb3af8f440f787bb5332af"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.80"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.Turing]]
deps = ["ADTypes", "AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "Setfield", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "e236871c3efed84b17187ea517183b408eaaf807"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.30.2"

    [deps.Turing.extensions]
    TuringDynamicHMCExt = "DynamicHMC"
    TuringOptimExt = "Optim"

    [deps.Turing.weakdeps]
    DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "801cbe47eae69adc50f36c3caec4758d2650741b"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.2+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "93284c28274d9e75218a416c65ec49d0e0fcdf3d"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.40+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─0defa484-7311-438e-8292-45d697844c7b
# ╠═1a62d194-bf7c-11ee-33ac-a52260216072
# ╠═6ab91324-f6cf-42e4-8f8d-0502fa717ba4
# ╟─fba9d036-d8c4-491e-8b67-624c1031862d
# ╟─19295fd8-55ad-4c3a-b931-eadefa61ed9c
# ╠═41a44fe5-c3a4-4264-b387-fd1e581950a3
# ╟─9f489ab7-80c9-4cf0-baa7-5c31e07c9cda
# ╠═4f884ba9-8aaf-451f-9768-f2d88626944b
# ╠═481fad8c-2be2-4e90-9cbc-d9f0252c4361
# ╠═798b1894-86d3-4098-ae25-52c2bac85197
# ╠═8f7c3384-0b62-4b4a-8192-b68497cb44be
# ╠═795c4e72-bcd4-43c9-8278-b536af9df641
# ╠═7e538a57-372c-48f1-9113-4aebf07d08b3
# ╟─d1ca84fe-2570-4a91-8b77-71b1959c12f7
# ╠═647d9fd8-230b-4604-bf27-1c0af38e01ee
# ╠═af95b3ce-049d-42e7-ac5f-9b884b17254a
# ╠═6cb6113f-9648-4034-bcef-ae8e4abcbf30
# ╠═155db03c-099d-443d-a622-fdd6302500f4
# ╠═2e3d51d4-d419-4cea-ab6e-f879aaabdbf9
# ╠═27931fb1-ac52-4260-9a34-5c72b06a447f
# ╠═b6b6c999-94fc-4415-ab2c-84c6c7e34785
# ╠═7331008d-e7bf-44b0-97bf-c3105fe234c7
# ╠═368a123d-3d50-4eaa-bb07-01a9ebf32c25
# ╠═7535e3bf-be16-43bc-9005-bd88b24c6722
# ╟─21b563bd-70b3-4c63-9237-181c856369dd
# ╠═31add289-a8b9-4bd3-b961-16d560755721
# ╠═16c96964-b4d6-4e4c-98b4-1050cd6a3193
# ╠═f84eefbd-b42d-495c-9e59-487db1208241
# ╠═4938efe4-7f03-46b1-8087-bd2e94363922
# ╠═06301bab-8245-44ad-90b9-8a1310190f06
# ╠═bfe53851-3423-4c7a-8e84-79412ed92a72
# ╠═28b6d8d4-b047-472e-8892-bc31365ac57f
# ╠═5b64b6e8-5f8a-4f59-9271-9378ba237aeb
# ╟─33e1c158-caf2-48cd-9852-1c304ec0ab15
# ╠═fc6b5dbe-f855-4e6e-b0f6-d26328cbc10a
# ╟─6877bc24-c236-4737-922e-9c427d08b204
# ╠═4a392913-714a-43fb-ae74-c5e17b3f2647
# ╠═ce36e74e-9a00-4720-a801-69010ec196aa
# ╟─36e45590-40b6-4de6-b2d2-7eab5135824d
# ╠═fbd630d0-a434-4e0a-ac74-de74b1d4ec1e
# ╠═180af191-1fbd-410e-84aa-97485241f961
# ╠═14d9a0ff-a785-46c7-a04e-f969abab078e
# ╠═659f89b3-4799-4623-9ca5-7b19871d6d57
# ╠═351428ab-a719-4b53-aa11-cee69f5ba187
# ╠═9c37b9aa-4d58-4c40-a2cb-dc5c5a0fc9d6
# ╠═69d1353a-0bf1-4937-ab7a-8bcbd820ca46
# ╠═850ac1af-db1f-4b34-b9d8-e1c3bf48bc81
# ╠═622c0eed-324a-4e8c-ba19-f85fd8040446
# ╠═4b3798d6-3616-4351-8996-89f2279b3831
# ╠═1d872b00-3025-45c5-ad9e-2dd765af418b
# ╠═1c9c2f9f-e02a-4d4b-b184-2530931916d6
# ╠═3b44d760-9868-47eb-adf0-35b1387b9e96
# ╠═2b071d65-441d-4f71-841d-76074be9a2b5
# ╠═f391ea1b-3a64-46e1-9b3b-107f403923bd
# ╟─0d51f279-918c-47d8-868c-a297870b1f34
# ╟─3dc7a293-477b-4732-9139-800b72bedb5e
# ╠═aefd44db-7881-4093-b738-33b0611a8eca
# ╠═e0849631-8ced-413d-8930-8508a2460e3b
# ╠═51f3c3e4-001e-4bf6-bce0-e0c3450be7f6
# ╠═e7ea41bf-8fc3-4d34-950e-86ca9076a470
# ╟─08cdba3b-57ca-49d4-aa44-3f29289efdd3
# ╠═ee8b1508-2e8b-48b4-bdf6-e041365440ec
# ╠═b24c109c-6e05-461a-bd75-b40f7cf9c7d4
# ╠═c78882f3-dbb4-4bc9-a50f-2e76d2f2d7b7
# ╟─699e035f-b40f-4c2e-8de9-cca4e5ea00f0
# ╠═b3ed302a-3ed9-41cc-a9fd-58a5a0c8ae96
# ╟─ec891f2b-c9fb-4eb7-a404-ca3bdbac4ca6
# ╠═c9654f02-f149-497c-b8f1-7196e3f42ab0
# ╟─5e5770ef-2741-4e8d-88c2-e6d75a299f88
# ╠═7dac6ba9-c635-40ae-8159-dab062147747
# ╠═24b19308-770c-4ec0-bbf0-49daf746d148
# ╠═f1e9a8a2-eda5-40a3-b990-4686c07057fd
# ╠═0bdb769e-37b7-4ff8-9fbc-16ed043758e5
# ╠═41cd5857-e991-409f-b6d2-528d4e120cf3
# ╠═4e69d83d-0e3a-4b09-a26e-29a498227bfa
# ╟─254c5889-7ed5-4958-8b7a-d671cfb9e937
# ╟─aa4d7850-4d36-4aa7-84f2-98da00fb4847
# ╠═3af274c3-9f47-4ebc-8394-962664621d95
# ╠═430a6859-050b-41dd-8322-abb95f8f17af
# ╠═d0cb5728-f408-4552-bdc6-f2023569b372
# ╠═d9489f67-e538-4e78-abe1-7d7890157538
# ╠═568014d5-9d25-408f-9f8f-cc95d9ec1e92
# ╠═040132cf-31c5-47a7-8af9-43986fcac947
# ╠═569f198c-03ad-4b48-9a6a-d8266e5ce64d
# ╠═c416d1b9-f9c7-45e2-91e3-c3c752c3a8c5
# ╠═6a03a2cb-addc-47fd-a5a9-fefcad0f10b9
# ╠═49db9667-63dd-4e68-8435-3e307701f432
# ╠═8acfe19f-6e96-4b27-a3d9-155e949d3dc1
# ╟─02c99f24-7847-45c6-83ff-773575b103f2
# ╟─c020cc25-58ad-4e87-b825-a264ae76fe23
# ╠═2cf8d8fc-383a-4684-b846-bc78911f0ae9
# ╠═446a5091-5118-4595-93a2-61a1682b2d7f
# ╠═b7faa27b-c7bb-4c17-ae4f-f85fc718eed3
# ╠═c28473b8-4e96-4944-884b-bb9cb023307a
# ╟─2db58b76-4593-480d-8d86-2c8b47c855b8
# ╠═ad0919ac-24c1-4226-8d37-c4f5c6353e16
# ╠═10f401ee-d629-451c-977f-1395071a40cc
# ╠═9b272218-ff5c-4c76-ada9-2d8d7d5db965
# ╠═b28803a9-15f9-4c3b-9992-83a4c30f6d6b
# ╠═90e9fb47-b239-47fd-b895-0833092aa79b
# ╠═ced3459e-7171-4364-bdb5-c7b75b5156c4
# ╠═cf4b44d6-39a0-4215-b14b-1ea829c89fd8
# ╠═4b375efd-6bb4-4123-a148-66f7f9cdff75
# ╠═a8c30639-e396-41b4-b201-86149154a339
# ╠═186555bf-8371-4fd6-8685-01d1f4194aef
# ╠═9a56b94c-1176-44e1-9ad1-ca25ebb324cd
# ╠═0e0a49fa-327f-439c-bdd7-42cea19f89d8
# ╠═88561bd2-4cd9-412e-95ab-8958446cc5a8
# ╠═03087d13-d5a7-450c-b702-659b5c6585b8
# ╠═bb64c497-b222-4810-8d76-e33b150731b4
# ╠═69643664-8a81-438d-8a16-8ad68816d8ee
# ╠═a5efafc4-d7a4-4a73-aabc-16db3f647f70
# ╠═447f0ddd-143c-4805-99dc-0ecf120e6ba9
# ╠═2a3ffcc0-d851-404c-9f08-90336759afed
# ╟─2532de3c-77f7-4236-bee3-e5fbe1b86053
# ╟─b7c890c3-75aa-419c-9e11-c229fa291cfd
# ╟─cf923955-0ca1-467a-a1b7-8227bc9fd6a7
# ╠═4617b5ad-ffe9-4900-89a1-9df1d2c33850
# ╠═bd93ed14-3e7f-40a3-b49c-ade4e9bb49c4
# ╠═96c1f4cf-31cd-49fb-80ce-685754d4b8e4
# ╠═b274cc04-8a0c-4d0a-88f2-1f8fccb1aaf1
# ╠═08c7003a-8a7a-4409-8c3d-377df46ff1e5
# ╠═dbdcefa2-8e8c-4672-8390-f9174c55c991
# ╠═c5978128-40d1-4fc6-adce-df98878a8b90
# ╠═a072df1c-fe6b-4250-81e5-a6312b2dfac8
# ╠═e10168a4-c967-4750-8487-5c8bbdd4807b
# ╠═dbfa8a0b-62ae-4a66-9aa2-100a7939db5b
# ╠═2f6041c2-06ea-41ea-8582-65361da47d63
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
