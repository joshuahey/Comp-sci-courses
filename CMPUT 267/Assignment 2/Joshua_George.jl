### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ a7b272bc-e8f5-11eb-38c6-8f61fea9941c
using CSV, DataFrames, StatsPlots, PlutoUI, Random, Statistics

# ╔═╡ 131466aa-d30d-4f75-a99f-13f47e1c7956
using LinearAlgebra: dot, norm, norm1, norm2

# ╔═╡ 7f8a82ea-1690-490c-a8bc-3c1f9556af2e
using Distributions: Uniform

# ╔═╡ 6fef79fe-aa3c-497b-92d3-6ddd87b6c26d
md"""
# Setup

This section loads and installs all the packages. You should be setup already from assignment 1, but if not please read and follow the `instructions.md` for further details.
"""

# ╔═╡ f3393367-4ded-4bf0-b030-13828cbaad82
plotly() # In this notebook we use the plotly backend for Plots. 

# ╔═╡ 5924a9a5-88c7-4751-af77-4a217dfdc15f
md"""
### !!!IMPORTANT!!!

Insert your details below. You should see a green checkmark.
"""

# ╔═╡ 9d38b22c-5893-4587-9c33-82d846fd0728
student = (name="Joshua George", email="jjgeorge@ualberta.ca", ccid="jjgeorge", idnumber=1665548)

# ╔═╡ 72d6b53f-f20d-4c05-abd4-d6fb4c997795
let
	def_student = (name="NAME as in eclass", email="UofA Email", ccid="CCID", idnumber=0)
	if length(keys(def_student) ∩ keys(student)) != length(keys(def_student))
		md"You don't have all the right entries! Make sure you have `name`, `email`, `ccid`, `idnumber`. ❌"
	elseif any(getfield(def_student, k) == getfield(student, k) for k in keys(def_student))
		md"You haven't filled in all your details! ❌"
	elseif !all(typeof(getfield(def_student, k)) === typeof(getfield(student, k)) for k in keys(def_student))
		md"Your types seem to be off: `name::String`, `email::String`, `ccid::String`, `idnumber::Int`"
	else
		md"Welcome $(student.name)! ✅"
	end
end

# ╔═╡ c8ceaf7a-d740-4123-8b26-e652dd17a86e
md"""
# Distance Metrics

Here we are defining some convenience functions for commonly used distance metrics.

"""

# ╔═╡ b3edc600-6670-4ae6-af7a-337e0600390e
begin
	RMSE(x̂, x) = sqrt(mean(abs2.(x̂ .- x))) # abs2 is equivalent to squaring, but faster and better numerically. 
	l2_error(x̂, x) = norm2(x̂ .- x)
	l1_error(x̂, x) = norm1(x̂ .- x)
end

# ╔═╡ ac180a6e-009f-4db2-bdfc-a9011dc5eb7b
md"""
# Abstract type Regressor

This is the basic Regressor interface. For the methods below we will be specializing the `predict(reg::Regressor, x::Number)`, and `epoch!(reg::Regressor, args...)` functions. Notice the `!` character at the end of epoch, as discussed earlier this this is a commonly used naming practice throughout the Julia language to indicate a function which modifies its arguments.


"""

# ╔═╡ 999ce1f2-fd11-4584-9c2e-bb79585d11f7
"""
	Regressor

Abstract Type for regression algorithms. Interface includes `predict` and an `epoch!`. In this notebook, we will only be using single variate regression.
- `predict(reg::Regressor, X::Number)`: return a prediction of the target given the feature `x`.
- `epoch!(reg::Regressor, X::AbstractVector, Y::AbstractVector)`: trains using the features `X` and regression targets `Y`.
"""
abstract type Regressor end # assume linear regression

# ╔═╡ 80f4168d-c5a4-41ef-a542-4e6d7772aa80
predict(reg::Regressor, x::Number) = Nothing

# ╔═╡ 65505c23-5096-4621-b3d4-d8d5ed339ad5
predict(reg::Regressor, X::AbstractVector) = [predict(reg, x) for x in X]

# ╔═╡ 67a8db8a-365f-4912-a058-d61648ac096e
epoch!(reg::Regressor, X::AbstractVector, Y::AbstractVector) = nothing

# ╔═╡ 7e083787-7a16-4064-97d9-8e2fca6ed686
md"""
# Baselines

In this section we will define the:
- `MeanRegressor`: Predict the mean of the training set.
- `RandomRegressor`: Predict `b*x` where `b` is sampled from a random normal distribution.
- `RangeRegressor`: Predict randomly in the range defined by the training set.

All the following baselines assume one dimension
"""

# ╔═╡ 5d5fe2d8-98ea-4ee1-b2e2-354eefaf8f77
md"""
## MeanRegressor
"""

# ╔═╡ eb152f9f-908f-4e5b-9642-f7314eb66e09
begin
	"""
		MeanRegressor()
		
	Predicts the mean value of the regression targets passed in through `epoch!`.
	"""
	mutable struct MeanRegressor <: Regressor
		μ::Float64
	end
	MeanRegressor() = MeanRegressor(0.0)
	predict(reg::MeanRegressor, x::Number) = reg.μ
	epoch!(reg::MeanRegressor, X::AbstractVector, Y::AbstractVector) = reg.μ = mean(Y)
end

# ╔═╡ a7b7bdaf-f4fd-40fb-a704-a584a93920f2
md"""
## RandomRegressor
"""

# ╔═╡ 0a90ceec-fc50-41fb-bf8b-e96ed677a5c3
begin
	"""
		RandomRegressor
	
	Predicts `b*x` where `b` is sambled from a normal distribution.
	"""
	struct RandomRegressor <: Regressor # random weights
		b::Float64
	end
	RandomRegressor() = RandomRegressor(randn())
	predict(reg::RandomRegressor, x::Number) = reg.b*x
end

# ╔═╡ 7771ac7c-e265-4e45-bfbe-7508124f266b
md"""
## RangeRegressor
"""

# ╔═╡ 2d5aa32d-6d57-4a8c-b8ae-c49d4d8c2e2d
begin
	"""
		RangeRegressor
	
	Predicts a value randomly from the range defined by `[minimum(Y), maximum(Y)]` as set in `epoch!`. Defaults to a unit normal distribution.
	"""
	mutable struct RangeRegressor <: Regressor
		min_value::Float64
		max_value::Float64
	end
	RangeRegressor() = RangeRegressor(0.0, 1.0)
	
	predict(reg::RangeRegressor, x::Number) = 
		rand(Uniform(reg.min_value, reg.max_value))
	predict(reg::RangeRegressor, x::AbstractVector) = 
		rand(Uniform(reg.min_value, reg.max_value), length(x))
	function epoch!(reg::RangeRegressor, X::AbstractVector, Y::AbstractVector) 
		reg.min_value = minimum(Y)
		reg.max_value = maximum(Y)
	end
end

# ╔═╡ e8f3981d-545c-4e35-9066-69fa4f78dbce
md"""
# Gradient Descent Regressors: Q3 a,b,c

In this section you will be implementing two gradient descent regressors, assuming a gaussian hypothesis class.  First we will create a gaussian regressor, and then use this to build our two new GD regressors. You can test your algorithms in the [experiment section](#experiment)

All the Gaussian Regressors will have data:
- `b::Float64` which is the parameter we are learning.
"""

# ╔═╡ 6ef4d625-fdf6-4f11-81f1-b242b2195e8b
abstract type GaussianRegressor <: Regressor end

# ╔═╡ 579da3a3-0e4e-4ba2-8f44-2657229994e3
predict(reg::GaussianRegressor, x::Float64) = reg.b * x

# ╔═╡ 67274c5b-2ee3-4865-a5e1-799db543d0c7
predict(reg::GaussianRegressor, X::Vector{Float64}) = reg.b .* X

# ╔═╡ 57c4dbf9-b7ba-44e9-8120-0e1a85896976
function probability(reg::GaussianRegressor, x, y)
end

# ╔═╡ b0160dc0-35af-4750-9ac6-d9018ce89ea9
begin
	mutable struct StochasticRegressor <: GaussianRegressor
		b::Float64
		η::Float64
	end
	StochasticRegressor(η::Float64) = StochasticRegressor(0.0, 0.01)
end

# ╔═╡ adedf31d-945f-491a-912a-bef9b03f6665
# Hint: Checkout the function randperm
function epoch!(reg::StochasticRegressor, 
		         X::AbstractVector{Float64}, 
		         Y::AbstractVector{Float64})
	for i=1:length(X)
        for j=1:length(Y)
            if i==j
                reg.b=reg.b-reg.η*(X[i]*reg.b-Y[j])*X[i]
            end
        end
    end
    reg.b
end
	


# ╔═╡ 4d422513-34e3-4d04-8ff1-c5165d953342
begin
	mutable struct BatchRegressor <: GaussianRegressor
		b::Float64
		η::Float64
		n::Union{Int, Nothing}
	end
	BatchRegressor(η, n=nothing) = BatchRegressor(0.0, η, n)
end

# ╔═╡ e5d78429-55f4-4a1a-831f-dbcb53c6a0f6
function epoch!(reg::BatchRegressor,
		        X::AbstractVector{Float64},
		        Y::AbstractVector{Float64})

if reg.n==nothing
	reg.n=length(X)
    end

for j in 1:(length(X)÷reg.n)
    size=j*reg.n
    sum1=0
        for i in (size-reg.n)+1:size
            sum1=sum1+(X[i]*reg.b-Y[i])*X[i]
        end
    reg.b=reg.b-(reg.η)*sum1/reg.n
	end
    print(reg.b)
end



# ╔═╡ 2aa8bd30-4b65-4e89-9dd5-5333efbbda3f
md"""
# Stepsize Heuristic: Q3 d
"""

# ╔═╡ 51bc41b4-b27f-4e60-8dba-70783c60c1c2
begin
	mutable struct StochasticHeuristicRegressor <: GaussianRegressor
		b::Float64
	end
	StochasticHeuristicRegressor() = 	
		StochasticHeuristicRegressor(0.0)
end

# ╔═╡ 00fcafa5-308c-4895-ad52-961772348125
# Hint: Checkout the function randperm
function epoch!(reg::StochasticHeuristicRegressor, 
		         X::AbstractVector{Float64}, 
		         Y::AbstractVector{Float64})
	for i=1:length(X)
        reg.b=reg.b-((X[i]*reg.b-Y[i])*X[i])/(1+abs(((X[i]*reg.b-Y[i])*X[i])))
    end
    reg.b
end
	

# ╔═╡ 7d6027af-6bfe-4cca-8b81-3e8fb12a6f79
begin
	mutable struct BatchHeuristicRegressor <: GaussianRegressor
		b::Float64
		n::Union{Int, Nothing}
	end
	BatchHeuristicRegressor(n=nothing) = BatchHeuristicRegressor(0.0, n)
end

# ╔═╡ 297b3bd1-29cc-4a96-bf2c-6305f8d375e4
function epoch!(reg::BatchHeuristicRegressor,
		        X::AbstractVector{Float64},
		        Y::AbstractVector{Float64})
if reg.n==nothing
    reg.n=length(X)
    end
    
for j in 1:(length(X)÷reg.n)
    size=j*reg.n
    sum1=0
    for i in (size-reg.n)+1:size
        sum1=sum1+(X[i]*reg.b-Y[i])*X[i]
    end
    reg.b=reg.b-sum1/((1+abs(sum1/reg.n))*reg.n)
    end
    print(reg.b)
end

# ╔═╡ add25a22-7423-4077-b0b0-f71eed2d2e20
md"""
### Stochastic Regressor $(let; sgr = StochasticRegressor(0.0, 0.1); epoch!(sgr, [1.0, 1.0], [1.0, 1.0]); predict(sgr, 1.0) == 0.19 ? "✅" : "❌"; end)

The stochastic regressor will be implemented via the stochastic gradient rule

```math
\begin{align*}
b^{t}_{i+1} = b^{t}_i - \eta (x_i b^{t}_i - y_i) x_i.
\end{align*}
```

Where $$b^{t}_{N+1}  = b^{t+1}$$, and each epoch iterates over the entire dataset in a random order.
"""

# ╔═╡ 9b69b182-ee8d-485e-ad6b-1f868874652c
let
	X, Y = [1.0, 2.0, 3.0], [1.0, 0.2, 0.1]
	bgr = BatchRegressor(0.1*3)
	epoch!(bgr, X, Y)
	batch_test = predict(bgr, 1.0) == 0.17 ? "✅" : "❌"
	
	X, Y = [1.0, 1.0, 1.0, 1.0,  1.0, 1.0], [0.32, 0.32, 0.32, 0.32, 0.32, 0.32]
	mbgr = BatchRegressor(0.1*2, 2)
	epoch!(mbgr, X, Y)
	mb_test = predict(mbgr, 1.0) == 0.15616 ? "✅" : "❌"
	
	md"""
	### Batch Regressor $(batch_test)

	The Minibatch regressor will be implemented via the gradient rule for a minibatch `j` with indicies for a batch defined by the set  $$\mathcal{I}$$

	```math
	\begin{align*}
	g_t^j &= \sum_{i\in \mathcal{I}_j} (x_i b_t - y_i) x_i \\
	b_{t+1} &= b_t - \eta g_t.
	\end{align*}
	```

	Your implementation should handle Batch Gradient Descent when the batch size is not specified. The minibatch regressor $(mb_test) can also be implemented through this interface using the same struct and `epoch!` function. 
	"""
end

# ╔═╡ 7b1d7d43-8613-46dc-9e92-00b91bfd5943
let
	sgr = StochasticHeuristicRegressor(0.0)
	epoch!(sgr, [1.0, 1.0], [1.0, 1.0])
	sgr_check = predict(sgr, 1.0) * 6 == 5.0 ? "✅" : "❌"
md"""
### Stochastic Regressor with heuristic $(sgr_check)
"""
end

# ╔═╡ 375f1a57-50f4-49ba-b089-a38f5315bd81
let
	X, Y = [1.0, 2.0, 3.0], [1.0, 0.2, 0.1]
	bgr = BatchHeuristicRegressor()
	epoch!(bgr, X, Y)
	batch_test = predict(bgr, 1.0) ≈ 0.36170212765 ? "✅" : "❌"
	
	X, Y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.32, 0.32, 0.32, 0.32, 0.32, 0.32]
	mbgr = BatchHeuristicRegressor(2)
	epoch!(mbgr, X, Y)
	mb_test = predict(mbgr, 1.0) ≈ 0.319968983713 ? "✅" : "❌"
	
	md"""
	### Batch Regressor with heuristic
	- Full Batch: $(batch_test)
	- Minibatch: $(mb_test)
	"""
end

# ╔═╡ 5aba1286-42de-4241-ba74-565385043f0b
md"""
# Data

Next we will be looking at the `height_weight.csv` dataset found in the data directory. This dataset provides three features `[sex, height, weight]`. In the following regression task we will be using `height` to predict `weight`, ignoring the `sex` feature.

The next few cells:
- Loads the dataset
- Plots distributions for the `height` and `weight` features seperated by `sex`
- Standardize the set so both `height` and `weight` conform to a standard normal.
- Defines `splitdataframe` which will be used to split the dataframe into training and testing sets.
"""

# ╔═╡ fdca1cde-48ba-4542-b762-b470ee80b1d3
# Read the data from the file in "data/height_weight.csv". DO NOT CHANGE THIS VALUE!
df_height_weight = DataFrame(CSV.File(joinpath(@__DIR__, "data/height_weight.csv"), 
							 header=["sex", "height", "weight"]));

# ╔═╡ 210f224f-a6aa-4ba7-b560-84ddc18823bf
try 
	identity(df_height_weight)
	md"Successfully loaded dataset ✅"
catch
	Markdown.parse("""Please place datset at `$(joinpath(@__DIR__, "data/height_weight.csv"))`""")
end
	

# ╔═╡ ce1bc6dc-70db-4ad7-b70c-f5c7f6c1129e
df_hw_norm = let
	df = copy(df_height_weight)
	σ_height = sqrt(var(df[!, :height]))
	μ_height = mean(df[!, :height])

	
	df[:, :height] .= (df[!, :height] .- μ_height) ./ σ_height
	
	σ_weight = sqrt(var(df[!, :weight]))
	μ_weight = mean(df[!, :weight])
	df[:, :weight] .= (df[!, :weight] .- μ_weight) ./ σ_weight
	
	df
end

# ╔═╡ 79dc3541-a75a-4f50-ab0c-2f6522a32eba
plt_hw = let
	df = df_height_weight # For convenience in the bellow code
	nothing
	plt1 = plot(xlabel="Sex", ylabel="Height", legend=nothing)
	@df df violin!(:sex, :height, linewidth=0)
	@df df boxplot!(:sex, :height, fillalpha=0.6)
	
	plt2 = plot(xlabel="Sex", ylabel="Weight", legend=nothing)
	@df df violin!(:sex, :weight, linewidth=0)
	@df df boxplot!(:sex, :weight, fillalpha=0.6)
	
	plot(plt1, plt2)
end

# ╔═╡ 6f4005da-57ff-4161-b4a5-437a1c072dd9
md"""
#### Plot data $(plt_hw isa Plots.Plot ? "✅" : "❌")
Plot a boxplot and violin plot of the height and weight. This can be with the classes `male` and `female` combined or with them separate. 
"""

# ╔═╡ 4c764fe7-e407-4ed9-9f5a-e2740b9863f6
"""
	splitdataframe(split_to_X_Y::Function, df::DataFrame, test_perc; shuffle = false)
	splitdataframe(df::DataFrame, test_perc; shuffle = false)

Splits a dataframe into test and train sets. Optionally takes a function as the first parameter to split the dataframe into X and Y components for training. This defaults to the `identity` function.
"""
function splitdataframe(split_to_X_Y::Function, df::DataFrame, test_perc; 
		                shuffle = false)
	#= shuffle dataframe. 
	This is innefficient as it makes an entire new dataframe, 
	but fine for the small dataset we have in this notebook. 
	Consider shuffling inplace before calling this function.
	=#
	
	df_shuffle = if shuffle == true
		df[randperm(nrow(df)), :]
	else
		df
	end
	
	# Get train size with percentage of test data.
	train_size = Int(round(size(df,1) * (1 - test_perc)))
	
	dftrain = df_shuffle[1:train_size, :]
	dftest = df_shuffle[(train_size+1):end, :]
	
	split_to_X_Y(dftrain), split_to_X_Y(dftest)
end

# ╔═╡ 790b9f50-7a3b-480c-92f4-33edd967b73d
splitdataframe(df::DataFrame, test_perc; kwargs...) = 
	splitdataframe(identity, df, test_perc; kwargs...)

# ╔═╡ 97dc155f-ef8a-4e55-960c-07fd0f5a6dde
let
	#=
		A do block creates an anonymous function and passes this to the first parameter of the function the do block is decorating.
	=#
	trainset, testset =
		splitdataframe(df_hw_norm, 0.1; shuffle=true) do df
			(X=df[!, :height], Y=df[!, :weight]) # create namedtuple from dataframes
		end
end

# ╔═╡ 168ef51d-dbcd-44c9-8d2e-33e98471962d
md"""
# Training the Models

The following functions are defined as utilities to train and evaluate our models. While hidden below, you can expand these blocks to uncover what is happening. `run_experiment!` is the main function used below in "**Using and Analyzing you Algorithms**".
"""

# ╔═╡ ae23765a-33fd-4a6b-a18c-3081c4382b14
function evaluate(err_func::Function, reg::Regressor, X, Y)
	err_func(predict(reg, X), Y)
end

# ╔═╡ 30177ca7-f9a5-4b97-8967-ffc154a509b0
begin
	evaluate_l1(reg::Regressor, X, Y) = evaluate(l1_error, reg, X, Y)
	evaluate_l2(reg::Regressor, X, Y) = evaluate(l2_error, reg, X, Y)
	evaluate_l∞(reg::Regressor, X, Y) = evaluate(reg, X, Y) do ŷ, y
		maximum(abs.(ŷ .- y))
	end

end

# ╔═╡ 532ed065-1f61-4eb0-9b4f-a6c236ab334d
function train!(ev_func::Function, reg::Regressor, X, Y, num_epochs)
	
	start_error = ev_func(reg, X, Y)
	ret = zeros(typeof(start_error), num_epochs+1)
	ret[1] = start_error
	
	for epoch in 1:num_epochs
		epoch!(reg, X, Y)
		ret[epoch+1] = ev_func(reg, X, Y)
	end
	
	ret
end

# ╔═╡ fbb14b46-51a2-4041-860b-8259f85ef2b7
train!(reg::Regressor, X, Y, num_epochs) = train!(evaluate_l2, reg::Regressor, X, Y, num_epochs)

# ╔═╡ 74b7458e-c654-4b85-9a31-ea575e0aa548
function run_experiment!(reg::Regressor, 
		               	 trainset, 
		                 testset, 
		                 num_epochs)
	
	train_err = train!(reg, trainset.X, trainset.Y, num_epochs)
	test_err = evaluate_l2(reg, testset.X, testset.Y)

	(regressor=reg, train_error=train_err, test_error=test_err)
end

# ╔═╡ 071b7886-ac26-4999-b20d-d63848331ebe
function run_experiment(regressors::Dict{String, Function}, 
						num_runs, 
						num_epochs; 
						seed=10392)

	ret = Dict{String, Any}()
	for (k, reg_init) in regressors
		ret[k] = map(1:num_runs) do r
			Random.seed!(seed+r)
			trainset, testset = splitdataframe(df_hw_norm, 0.1; shuffle=true) do df
				(X=df[!, :height], Y=df[!, :weight]) # create named tuple from DF
			end
			run_experiment!(reg_init(), trainset, testset, num_epochs)
		end
	end

	ret
end

# ╔═╡ 6f2a0d1e-e76b-4326-a350-af49a8fd30e6
html"""<h1 id="experiment"> Using and Analyzing your Algorithms </h1>"""

# ╔═╡ 76d0599d-b893-4cb2-b94d-0e787fd39a74
begin
	
	__s_η = 0.01
	__b_η = 0.01
	__mb_η = 0.01
	__mb_n = 100
	
	Markdown.parse("""

	In this section we will be running and analyzing a small experiment. The goal is to get familiar with analyzing data, plotting learning curves, and comparing different methods. Below we've provided a start with the baselines. Add new initilizors for a Batch update `(η = $(__b_η))`, a Minibatch update `(η = $(__mb_η), n = $(__mb_n))`, and a Stochastic update `(η = $(__s_η))`. Also add their heuristic counterparts. 
		
	As a point of reference: running 
	```julia
	results = run_experiment(regressor_init, 10, 30)
	```
	in the cell below takes roughly `8 seconds` on my machine.

	""")
end

# ╔═╡ 468b1077-6cf4-45d4-a4a6-41b134a6d3d7
regressor_init = Dict(
	"Mean"=>()->MeanRegressor(),
	"Random"=>()->RandomRegressor(),
	"Range"=>()->RangeRegressor(),
	# use the keys "Batch", "Stochastic", and "Minibatch".
	"Stochastic"=>()->StochasticRegressor(0.01),
	"Batch"=>()->BatchRegressor(0.01),
    "Minibatch"=>()->BatchRegressor(0.01, 100),
    "StochasticHeuristic"=>()->StochasticHeuristicRegressor(),
    "BatchHeuristic"=>()->BatchHeuristicRegressor(),
    "MinibatchHeuristic"=>()->BatchHeuristicRegressor(100)


)

# ╔═╡ ed20ad2c-87ab-4833-ae78-0c1906aa90a6
results = run_experiment(regressor_init, 10, 30)

# ╔═╡ 42ab7e2a-d390-4528-80cd-0e30a1eb2133
begin
	__results_checks = Dict{String, Union{Vector{Bool}, Bool}}(
		"Stochastic"=>false,
		"Minibatch"=>[false, false],
		"Batch"=>false
	)
	
	
	a = """
	Experiment ran for:
		
	"""
	for k in ["Mean", "Random", "Range", "Stochastic", "Batch", "Minibatch", "StochasticHeuristic", "BatchHeuristic", "MinibatchHeuristic"]
		
		if k in keys(results)
			if k == "Batch"
				a *= "- ✅ `$(k)`: "
				__results_checks[k] = results[k][1].regressor.η == __b_η
				a *= """with stepsize= `$(results[k][1].regressor.η)` $(results[k][1].regressor.η == __b_η ? "✅" : "❌" )"""
			elseif k == "Stochastic"
				a *= "- ✅ `$(k)`: "
				__results_checks[k] = results[k][1].regressor.η == __s_η
				a *= """with stepsize= `$(results[k][1].regressor.η)` $(results[k][1].regressor.η == __s_η ? "✅" : "❌" )"""
			elseif k == "Minibatch"
				a *= "- ✅ `$(k)`: "
				__results_checks[k] = [results[k][1].regressor.η == __mb_η, 	
					   				   results[k][1].regressor.n == __mb_n]
				a *= """with stepsize= `$(results[k][1].regressor.η)` $(results[k][1].regressor.η == __mb_η ? "✅" : "❌" ) and batch size = `$(results[k][1].regressor.n)` $(results[k][1].regressor.n == __mb_n ? "✅" : "❌" )"""
			elseif k == "MinibatchHeuristic"
				a *= "- ✅ `$(k)`: "	
				__results_checks[k] = [results[k][1].regressor.n == __mb_n]
				a *= """with batch size = `$(results[k][1].regressor.n)` $(results[k][1].regressor.n == __mb_n ? "✅" : "❌" )"""
			else
				a *= "- ✅ `$(k)`"
			end
		else
			a *= "- ❌ `$(k)`"
		end
		a *= "\n\n"
	end
	Markdown.parse(a)
			end

# ╔═╡ 3a0a1a93-0ea1-4570-9be6-aa892b515c7b
md"""
The results dictionary is the resulting data from the experiment we run using `regressor_init` as the intializors. You will see the same keys used as in the `regressor_init` dictionary. For each run the experiment returns the final regressor, the training error vector, and the final test error. You can get one of these components for a particular methd using `getindex` and broadcasting:
```julia
getindex.(results["Mean"], :test_error)
```

"""

# ╔═╡ ac703837-ed65-49c6-97ad-5b54e30a680e
let
	# Play with data here! You can explore how to get different values.
	mean(getindex.(results["Mean"], :test_error))
	mean(getindex.(results["Random"], :test_error))
	mean(getindex.(results["Range"], :test_error))
	mean(getindex.(results["Stochastic"], :test_error))
	mean(getindex.(results["Batch"], :test_error))
	mean(getindex.(results["Minibatch"], :test_error))
	mean(getindex.(results["StochasticHeuristic"], :test_error))
	mean(getindex.(results["BatchHeuristic"], :test_error))
	mean(getindex.(results["MinibatchHeuristic"], :test_error))
	std(getindex.(results["Mean"], :test_error))
	std(getindex.(results["Random"], :test_error))
	std(getindex.(results["Range"], :test_error))
	std(getindex.(results["Stochastic"], :test_error))
	std(getindex.(results["Batch"], :test_error))
	std(getindex.(results["Minibatch"], :test_error))
	std(getindex.(results["StochasticHeuristic"], :test_error))
	std(getindex.(results["BatchHeuristic"], :test_error))
	std(getindex.(results["MinibatchHeuristic"], :test_error))

end

# ╔═╡ 063dc493-af4d-4014-bc42-28705beaf216


# ╔═╡ 6f26e6f8-088c-4578-bf8b-07d38ff53d00
plt_lc = let
	plt = plot()
	# plt=plot(lw=2)
	for method ∈ keys(results)
		μ = mean(getindex.(results[method], :train_error))
		σ = sqrt.(var(getindex.(results[method], 2)) / length(results[method]))
		plot!(plt, μ, ribbon=σ, lw=2, label=method)
	end
	plt
end

# ╔═╡ e3224f78-4cfb-45e7-a1f7-6640051afcd2
md"""
## Learning Curves $(plt_lc isa Plots.Plot ? "✅" : "❌")

Plot the average learning curve with the standard error calculated as

```math
\sigma_{err}(\mathbf{x}) = \sqrt{\frac{\text{Var}(x)}{|x|}}
```

Note that $$\mathbf{x}$$ is a vector over runs, not over epochs.

_Note:_ if you notice one method is dominating the plot, change the axis limits to make sure the methods we are most concerned with (i.e. Stochastic, Batch, and Minibatch) are visible.
"""

# ╔═╡ 7bc3d05f-9af6-4163-8a31-4143c9606b5b
plt_fe = let
	plt = plot(legend=nothing)
	for method ∈ keys(results)
		if method ∈ ["Range", "Random"]
			continue
		end
		data = getindex.(results[method], :test_error)
		boxplot!([method], data)
	end
	plt
end

# ╔═╡ 4f22d201-69bb-4ee1-9393-5058eaffd3d1
md"""
## Final Errors $(plt_fe isa Plots.Plot ? "✅" : "❌")

Finally, we want to compare the final test errors of the different methods. One way to do this is through box plots. See [this great resource](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51) to learn how to compare data using a box and whisker plot. In this plot you can ignore the `Range` and `Random` baselines.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.8.5"
DataFrames = "~1.2.2"
Distributions = "~0.25.11"
PlutoUI = "~0.7.9"
StatsPlots = "~0.14.26"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "abe4ad222b26af3337262b8afb28fab8d215e9f8"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.3"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3889f646423ce91dd1055a76317e9a1d3a23fff1"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.11"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f985af3b9f4e278b1d24434cbb546d6092fca661"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.3"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3676abafff7e4ff07bbd2c42b3d8201f31653dcc"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.9+8"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8c8eac2af06ce35973c3eadb4ab3243076a408e7"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "182da592436e287758ded5be6e32c406de3a2e47"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d59e8320c2747553788e4fc42231489cc602fa50"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.58.1+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "44e3b40da000eab4ccb1aecdc4801c040026aeb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.13"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0f4a4836e5f3e0763243b8324200af6d0e0f90c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.5"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "8365fa7758e2e8e4443ce866d6106d8ecbb4474e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.20.1"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "7dff99fbc740e2f8228c6878e2aad6d7c2678098"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.1"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "a3a337914a035b2d59c9cbe7f1a38aaba1265b02"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.6"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[StatsFuns]]
deps = ["IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "20d1bb720b9b27636280f751746ba4abb465f19d"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.9"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e7d1e79232310bd654c7cef46465c537562af4fe"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.26"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "000e168f5cc9aded17b6999a560b7c11dda69095"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.0"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "a7cf690d0ac3f5b53dd09b5d613540b230233647"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.0.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "eae2fbbc34a79ffd57fb4c972b08ce50b8f6a00d"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.3"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─6fef79fe-aa3c-497b-92d3-6ddd87b6c26d
# ╠═a7b272bc-e8f5-11eb-38c6-8f61fea9941c
# ╠═131466aa-d30d-4f75-a99f-13f47e1c7956
# ╠═7f8a82ea-1690-490c-a8bc-3c1f9556af2e
# ╠═f3393367-4ded-4bf0-b030-13828cbaad82
# ╟─5924a9a5-88c7-4751-af77-4a217dfdc15f
# ╟─72d6b53f-f20d-4c05-abd4-d6fb4c997795
# ╠═9d38b22c-5893-4587-9c33-82d846fd0728
# ╟─c8ceaf7a-d740-4123-8b26-e652dd17a86e
# ╠═b3edc600-6670-4ae6-af7a-337e0600390e
# ╟─ac180a6e-009f-4db2-bdfc-a9011dc5eb7b
# ╠═999ce1f2-fd11-4584-9c2e-bb79585d11f7
# ╠═80f4168d-c5a4-41ef-a542-4e6d7772aa80
# ╠═65505c23-5096-4621-b3d4-d8d5ed339ad5
# ╠═67a8db8a-365f-4912-a058-d61648ac096e
# ╟─7e083787-7a16-4064-97d9-8e2fca6ed686
# ╟─5d5fe2d8-98ea-4ee1-b2e2-354eefaf8f77
# ╠═eb152f9f-908f-4e5b-9642-f7314eb66e09
# ╟─a7b7bdaf-f4fd-40fb-a704-a584a93920f2
# ╠═0a90ceec-fc50-41fb-bf8b-e96ed677a5c3
# ╟─7771ac7c-e265-4e45-bfbe-7508124f266b
# ╠═2d5aa32d-6d57-4a8c-b8ae-c49d4d8c2e2d
# ╟─e8f3981d-545c-4e35-9066-69fa4f78dbce
# ╠═6ef4d625-fdf6-4f11-81f1-b242b2195e8b
# ╠═579da3a3-0e4e-4ba2-8f44-2657229994e3
# ╠═67274c5b-2ee3-4865-a5e1-799db543d0c7
# ╠═57c4dbf9-b7ba-44e9-8120-0e1a85896976
# ╟─add25a22-7423-4077-b0b0-f71eed2d2e20
# ╠═b0160dc0-35af-4750-9ac6-d9018ce89ea9
# ╠═adedf31d-945f-491a-912a-bef9b03f6665
# ╟─9b69b182-ee8d-485e-ad6b-1f868874652c
# ╠═4d422513-34e3-4d04-8ff1-c5165d953342
# ╠═e5d78429-55f4-4a1a-831f-dbcb53c6a0f6
# ╠═2aa8bd30-4b65-4e89-9dd5-5333efbbda3f
# ╟─7b1d7d43-8613-46dc-9e92-00b91bfd5943
# ╠═51bc41b4-b27f-4e60-8dba-70783c60c1c2
# ╠═00fcafa5-308c-4895-ad52-961772348125
# ╠═375f1a57-50f4-49ba-b089-a38f5315bd81
# ╠═7d6027af-6bfe-4cca-8b81-3e8fb12a6f79
# ╠═297b3bd1-29cc-4a96-bf2c-6305f8d375e4
# ╟─5aba1286-42de-4241-ba74-565385043f0b
# ╠═fdca1cde-48ba-4542-b762-b470ee80b1d3
# ╟─210f224f-a6aa-4ba7-b560-84ddc18823bf
# ╟─ce1bc6dc-70db-4ad7-b70c-f5c7f6c1129e
# ╟─6f4005da-57ff-4161-b4a5-437a1c072dd9
# ╠═79dc3541-a75a-4f50-ab0c-2f6522a32eba
# ╟─4c764fe7-e407-4ed9-9f5a-e2740b9863f6
# ╟─790b9f50-7a3b-480c-92f4-33edd967b73d
# ╠═97dc155f-ef8a-4e55-960c-07fd0f5a6dde
# ╟─168ef51d-dbcd-44c9-8d2e-33e98471962d
# ╟─ae23765a-33fd-4a6b-a18c-3081c4382b14
# ╟─30177ca7-f9a5-4b97-8967-ffc154a509b0
# ╟─532ed065-1f61-4eb0-9b4f-a6c236ab334d
# ╟─fbb14b46-51a2-4041-860b-8259f85ef2b7
# ╟─74b7458e-c654-4b85-9a31-ea575e0aa548
# ╟─071b7886-ac26-4999-b20d-d63848331ebe
# ╟─6f2a0d1e-e76b-4326-a350-af49a8fd30e6
# ╟─76d0599d-b893-4cb2-b94d-0e787fd39a74
# ╟─42ab7e2a-d390-4528-80cd-0e30a1eb2133
# ╠═468b1077-6cf4-45d4-a4a6-41b134a6d3d7
# ╠═ed20ad2c-87ab-4833-ae78-0c1906aa90a6
# ╟─3a0a1a93-0ea1-4570-9be6-aa892b515c7b
# ╠═ac703837-ed65-49c6-97ad-5b54e30a680e
# ╠═063dc493-af4d-4014-bc42-28705beaf216
# ╟─e3224f78-4cfb-45e7-a1f7-6640051afcd2
# ╠═6f26e6f8-088c-4578-bf8b-07d38ff53d00
# ╟─4f22d201-69bb-4ee1-9393-5058eaffd3d1
# ╠═7bc3d05f-9af6-4163-8a31-4143c9606b5b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
