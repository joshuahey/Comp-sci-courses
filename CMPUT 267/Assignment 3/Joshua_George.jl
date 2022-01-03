### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ dc004086-db6e-4813-841e-d427520402f7
begin
	using CSV, DataFrames, StatsPlots, PlutoUI, Random, Statistics
	using LinearAlgebra: dot, norm, norm1, norm2, I
	using Distributions: Distributions, Uniform
	using MultivariateStats: MultivariateStats, PCA
	using StatsBase: StatsBase
	
end

# ╔═╡ 75441ce6-2137-4fcf-bba2-6ed67b9acb59
begin
	_check_complete(complete) = complete ? "✅" : "❌"
	
	md"""
	# Setup

	this section loads and installs all the packages. You should be setup already from assignment 1, but if not please read and follow the `instructions.md` for further details.
	"""
end

# ╔═╡ c9797979-7d16-4ba4-8f82-c7ec33f6e517
plotly() # In this notebook we use the plotly backend for Plots.

# ╔═╡ 693a3933-c1c2-4249-8c03-f5151267222f
md"""
### !!!IMPORTANT!!!

Insert your details below. You should see a green checkmark.
"""

# ╔═╡ def97306-1703-42bc-bc09-da623c545e87
student = (name="Joshua George", email="jjgeorge@ualberta.ca", ccid="jjgeorge", idnumber=1665548)

# ╔═╡ bc0c5abe-1c2a-4e5c-bb0c-8d76b5e81133
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

# ╔═╡ 14c30b50-157f-40bf-b0a6-0232aa2705c6
md"""
Important Note: You should only write code in the cells that has: """


# ╔═╡ 4a196e82-fe94-49fe-ab87-d57e2d7fbd34
#### BEGIN SOLUTION


#### END SOLUTION

# ╔═╡ dba201d8-a0a5-4444-bcf8-913696655a49
md"""

This is a followup question on Question 3 from Assignment 2. As before, you will implement an algorithm to estimate $p(y | x)$, for a data batch of pairs of $(x,y)$: $\mathcal{D} = \{ (x_i,y_i)\}_{i=1}^n$.

DistributionRegressor is a multivariate regressor with two variables $\mu$ and $\sigma^2$ for a gaussian distribution. Assume that $\mu$ is a linear function of $b$ with $\mu = xb$, and $\sigma^2 = exp(xa)$ depends on prameter $a$.  
Then for a randomly sampled $x_i, y_i$, you first need to derive the updates for both parameters $a$ and $b$ by calculating the following partial derivatives of the normal distribution 
```math
\begin{align*}
\frac{\partial c(w_t)}{\partial b} \qquad \& \qquad \frac{\partial c(w_t)}{\partial a}
\end{align*}
``` in Q1(a), where $c(w)$ is the loss function, proportional to the negative log likelihood for this problem. 


Q1(b): Implement the updates you have derived in Q1(a) by iterating over the entire dataset in a random order in each epoch:

```math
\begin{align*}
b_{t+1} &= b_{t} - \eta_{t}\frac{\partial c(w_t)}{\partial b}
\\
a_{t+1} &= a_{t} - \eta_{t} \frac{\partial c(w_t)}{\partial a}
\end{align*}
```
The heuristic for implementing an adaptive stepsize would be:
```math
\begin{align*}
\eta_{t} &=\left(1+\sqrt{\left(\frac{\partial c(w_t)}{\partial b}\right)^{2} + \left(\frac{\partial c(w_t)}{\partial a}\right)^{2}}\right)^{-1} \
\end{align*}
```
"""

# ╔═╡ 43870efd-4050-4b75-9a17-504ff66ddf70
begin
	"""
	
	"""
	mutable struct DistributionRegressor
		b::Float64
		a::Float64
	end
	DistributionRegressor() = DistributionRegressor(0.0, 0.0)
	predict(reg::DistributionRegressor, x::Float64) = reg.b * x, exp(reg.a * x)
end

# ╔═╡ 983693ea-4401-4115-bd40-47aa28caee2d
function epoch!(reg::DistributionRegressor, x::Float64, y::Float64)
	# Based on the above
	#### BEGIN SOLUTION
	p=x*(1-(y-x*reg.b)^(2)*exp(-x*reg.a))*(1/2)
	q=(1/exp(reg.a*x))*(x*reg.b-y)*x
	n=(1/(1+sqrt(p^2+q^2)))
    reg.b=reg.b-n*q
    reg.a=reg.a-n*p


	#### END SOLUTION
end

# ╔═╡ 01682bea-c735-427f-be82-9b6d8b06c028
md"""
## Q1(c) Testing the Distribution Regressor
"""

# ╔═╡ 279a3b98-0124-4d73-8a54-93947deaaf16
begin
	md"""
	In the cell below you will compare your implementation of the distribution regressor against the following:
	- Mean Regressor
	- Range Regressor
	"""
end

# ╔═╡ be72d249-9c8b-4f87-9404-e53c27ec83fb
begin
	"""
		RangeRegressor

	Predicts a value randomly from the range defined by `[minimum(Y), maximum(Y)]` as set in `epoch!`. Defaults to a unit normal distribution.
	"""
	mutable struct RangeRegressor
		min_value::Float64
		max_value::Float64
	end
	RangeRegressor() = RangeRegressor(0.0, 1.0)

	predict(reg::RangeRegressor, x::Number) = 
		rand(Uniform(reg.min_value, reg.max_value))
	predict(reg::RangeRegressor, x::AbstractVector) = 
		rand(Uniform(reg.min_value, reg.max_value), length(x))
	
	function train!(reg::RangeRegressor, X::AbstractVector, Y::AbstractVector, args...) 
		reg.min_value = minimum(Y)
		reg.max_value = maximum(Y)
	end
	
	"""
		MeanRegressor()
		
	Predicts the mean value of the regression targets passed in through `epoch!`.
	"""
	mutable struct MeanRegressor
		μ::Float64
	end
	MeanRegressor() = MeanRegressor(0.0)
	predict(reg::MeanRegressor, x::Number) = reg.μ
	function train!(reg::MeanRegressor, X::AbstractVector, Y::AbstractVector, args...) 
		reg.μ = mean(Y)
	end
	
	RangeRegressor, MeanRegressor
end

# ╔═╡ d011cc2f-c39d-475c-be03-fa2907126c91
md"""
We test the distribution regression against the mean and range baselines implemented above over 20 runs with standard error bars. We use the height and weight dataset from Assignment 2.
"""

# ╔═╡ e102cde6-3461-4dea-b8f2-dec803edd6fa
# loading dataset
df_hw_norm = let
	df_height_weight = DataFrame(CSV.File(joinpath(@__DIR__, "data/height_weight.csv"),  header=["sex", "height", "weight"]));
	df = copy(df_height_weight)
	σ_height = sqrt(var(df[!, :height]))
	μ_height = mean(df[!, :height])

	
	df[:, :height] .= (df[!, :height] .- μ_height) ./ σ_height
	
	σ_weight = sqrt(var(df[!, :weight]))
	μ_weight = mean(df[!, :weight])
	df[:, :weight] .= (df[!, :weight] .- μ_weight) ./ σ_weight
	
	df
end;

# ╔═╡ 51c4bc05-cecf-4695-94f4-5e39da34a8c5
begin
	md"""
	Compare the performance of DistributionRegressor against the baseline algorithms by reporting the average error and standard error after 1 epoch and after 20 epochs over 20 runs. You should only change num_epochs variable and report the resultant output. **You can get the average error and standard error to report from the plot or from the terminal where you run this notebook**.
	"""
end

# ╔═╡ a7aecd21-13f2-4cf2-8239-3a3d708602c9
md"""
# Q2: Multi-variate Regression

So far, we have only considered our algorithms when the features are drawn from a single dimension. But this is a considerable limitation. In the following section we will explore implementations of algorithms for multi-variate regression.

Unlike before, instead of having a struct be all the properties of an ML systems we will break our systems into smaller pieces. This will allow us to more easily take advantage of code we've already written, and will be more useful as we expand the number of algorithms we consider. We make several assumptions to simplify the code, but the general type hierarchy can be used much more broadly.

We split each system into:
- Model
- Gradient descent procedure
- Loss Function
- Optimization Strategy
"""

# ╔═╡ e3c3e903-e2c2-4de0-a3b2-56a27709e8c3
md"""
## Baselines
"""

# ╔═╡ 3237c4fc-56d1-4135-a8d2-cc6e88f2f5c0
md"""
### Mean Model
"""

# ╔═╡ a35944ae-7bbb-4f3c-bbe0-d2bbb57cd70b
md"""
### RandomModel
"""

# ╔═╡ 4855856c-9eed-4ab5-9170-7cc9ed4115fc
md"""
### RangeModel
"""

# ╔═╡ 7b513344-1cad-4eef-9faf-e77ba176323e
md"""
# Models

"""

# ╔═╡ 4f4029a2-c590-4bd3-a0db-d2380d4b4620
md"""
## The model interface

- `AbstractModel`: This is an abstract type which is used to derive all the model types in this assignment
- `predict`: This takes a matrix of samples and returns the prediction doing the proper data transforms.
- `get_features`: This transforms the features according to the non-linear transform of the model (which is the identity for linear).
- `get_linear_model`: All models are based on a linear model with transformed features, and thus have a linear model.
- `copy`: This returns a new copy of the model.
"""

# ╔═╡ dcfecc35-f25c-4856-8429-5c31d94d0a42
"""
	AbstractModel

Used as the root for all models in this notebook. We provide a helper `predict` function for `AbstractVectors` which transposes the features to a row vector. We also provide a default `update_transform!` which does nothing.
"""
abstract type AbstractModel end

# ╔═╡ d45dedf8-e46a-4ffb-ab45-f9d211d3a8ca
predict(alm::AbstractModel, x::AbstractVector) = predict(alm, x')[1]

# ╔═╡ 7cd46d84-a74c-44dc-8339-68010924bc39
update_transform!(AbstractModel, args...) = nothing

# ╔═╡ 8745fec1-47c8-428b-9ea4-1e6828618830
md"
#### Linear Model

As before, we define a linear model as a linear map
```math
f(x) = \hat{y} = \mathbf{w}^\top x
```

or with a data matrix $$X$$ of size `(samples, features)`

```math
f(X) = \hat{Y} = X \mathbf{w}
```

To make the predict function simpler we provide a convenience predict function for all abstract models which transforms a `Vector` (which in julia is always a column vector), to a row vector (or a 1xn matrix). So you can call `predict(model, rand(10))` without worrying about whether `x` is a column or row vector. You will still need to pay attention to this when implementing future code.

"

# ╔═╡ 2d43a3ba-2a2c-4114-882f-5834d42e302a
begin
	struct LinearModel <: AbstractModel
		W::Matrix{Float64} # Aliased to Array{Float64, 2}
	end
	
	LinearModel(in, out=1) = 
		LinearModel(zeros(in, out)) # feture size × output size
	
	Base.copy(lm::LinearModel) = LinearModel(copy(lm.W))
	predict(lm::LinearModel, X::AbstractMatrix) = X * lm.W
	get_features(m::LinearModel, x) = x

end

# ╔═╡ ded749bf-b9fa-4e2b-b15f-0693d820a9c3
md"""
Now, we will implement Polynomial Model which basically uses the linear model with non-linear features. To transform features, we apply polynomial transformation to our data. 

To acheive polynomial fit of degree $p$, we will have a non-linear map of features

```math
f(x)=\sum_{j=0}^p w_j x^j
```
which we can write as a basis function:

```math
f(x)=\sum_{j=0}^p w_j \phi_j (x) = \mathbf{w}^\top \Phi
```
where $\phi_j (x) = x^j$
so we simply apply this transformation to every data point $x_i$ to get the new dataset $\{(\phi(x_i), y_i)\}$.

Implement polynomial features transformation by constructing $\Phi$ with $p = 2$ degrees in the function ```get_features```.

"""

# ╔═╡ 2e69a549-aab4-4c88-aad8-dffb822d644f
begin
	struct Polynomial2Model <: AbstractModel 
		model::LinearModel
		ignore_first::Bool
	end
	Polynomial2Model(in, out=1; ignore_first=false) = if ignore_first
		in = in - 1
		Polynomial2Model(LinearModel(1 + in + Int(in*(in+1)/2), out), ignore_first)
	else
		Polynomial2Model(LinearModel(1 + in + Int(in*(in+1)/2), out), ignore_first)
	end
	Base.copy(lm::Polynomial2Model) = Polynomial2Model(copy(lm.model), lm.ignore_first)
	get_linear_model(lm::Polynomial2Model) = lm.model
	
end

# ╔═╡ 0ba5f9c8-5677-40e9-811b-25546e0df207
function get_features(pm::Polynomial2Model, _X::AbstractMatrix)
	
	# If _X already has a bias remove it.
	X = if pm.ignore_first
		_X[:, 2:end]
	else
		_X
	end
	
	m = size(X, 2)
	N = size(X, 1)
	num_features = 1 + # Bias bit
				   m + # p = 1
				   Int(m*(m+1)/2) # combinations (i.e. x_i*x_j)
	
	Φ = zeros(N, num_features)
	
	# Construct Φ
	#### BEGIN SOLUTION
	for i in 1:N
    	for j in 1:1
       		Φ[i,j] = 1
    	end
		for j in 1:m
			Φ[i,j+1] = X[i,j]
		end
		permutations = []
		for k in 1:m+1
			append!(permutations,0)
		end
		for a in 1:m
			for b in a:m
				append!(permutations,X[i,a]*X[i,b])
			end
		end

		for j in m+2:num_features
			Φ[i,j] = permutations[j]
		end
	end

	#### END SOLUTION
	
	Φ
end

# ╔═╡ c59cf592-a893-4ffa-b247-51d94c7cdb1a
begin
		
	_check_Poly2 = let
		pm = Polynomial2Model(2, 1)
		rng = Random.MersenneTwister(1)
		X = rand(rng, 3, 2)
		Φ = get_features(pm, X)
		Φ_true = [
			1.0 0.23603334566204692 0.00790928339056074 0.05571174026441932 0.0018668546204633095 6.25567637522e-5; 
			1.0 0.34651701419196046 0.4886128300795012 0.12007404112451132 0.16931265897503248 0.2387424977182995; 
			1.0 0.3127069683360675 0.21096820215853596 0.09778564804593431 0.06597122691230639 0.04450758232200489]
		check_1 = all(Φ .≈ Φ_true)
		pm = Polynomial2Model(2, 1; ignore_first=true)
		X_bias = ones(size(X, 1), size(X, 2) + 1)
		X_bias[:, 2:end] .= X
		Φ = get_features(pm, X_bias)
		check_2 = all(Φ .≈ Φ_true)
		check_1 && check_2
	end
	
	HTML("<h4 id=poly> (a) Polynomial Features $(_check_complete(_check_Poly2))</h4>")
end

# ╔═╡ 0608c93d-2f82-470c-8d9f-ca79af6b2612
predict(lm::Polynomial2Model, X) = predict(lm.model, get_features(lm, X))

# ╔═╡ d9935cc8-ec24-47e9-b39a-92c21377a161
struct MiniBatchGD
	n::Int
end

# ╔═╡ 5080cc19-d13f-4236-b49e-e338108dda80
begin
	"""
		MeanModel()
		
	Predicts the mean value of the regression targets passed in through `epoch!`.
	"""
	mutable struct MeanModel <: AbstractModel
		μ::Float64
	end
	MeanModel() = MeanModel(0.0)
	predict(reg::MeanModel, X::AbstractVector) = reg.μ
	predict(reg::MeanModel, X::AbstractMatrix) = fill(reg.μ, size(X,1))
	Base.copy(reg::MeanModel) = MeanModel(reg.μ)
	function train!(::MiniBatchGD, model::MeanModel, lossfunc, opt, X, Y, num_epochs)
		model.μ = mean(Y)
	end
end

# ╔═╡ e7712bd3-ea7e-4f4a-9efc-041b4b2be987
begin
	"""
		RandomModel
	
	Predicts `b*x` where `b` is sambled from a normal distribution.
	"""
	struct RandomModel <: AbstractModel # random weights
		W::Matrix{Float64}
	end
	RandomModel(in, out) = RandomModel(randn(in, out))
	predict(reg::RandomModel, X::AbstractMatrix) = X*reg.W
	Base.copy(reg::RandomModel) = RandomModel(randn(size(reg.W)...))
	train!(::MiniBatchGD, model::RandomModel, lossfunc, opt, X, Y, num_epochs) = 
		nothing
end

# ╔═╡ 725cadb3-3c4f-4344-9078-7ea7424bde1d
begin
	"""
		RangeModel
	
	Predicts a value randomly from the range defined by `[minimum(Y), maximum(Y)]` as set in `epoch!`. Defaults to a unit normal distribution.
	"""
	mutable struct RangeModel <: AbstractModel
		min_value::Float64
		max_value::Float64
	end
	RangeModel() = RangeModel(0.0, 1.0)
	
	predict(reg::RangeModel, x::AbstractMatrix) = 
		rand(Uniform(reg.min_value, reg.max_value), size(x, 1))
	Base.copy(reg::RangeModel) = RangeModel(reg.min_value, reg.max_value)
	function train!(::MiniBatchGD, model::RangeModel, lossfunc, opt, X, Y, num_epochs)
		model.min_value = minimum(Y)
		model.max_value = maximum(Y)
	end
end

# ╔═╡ 5714c84f-1653-4c4a-a2e4-003d8560484a
md"""

Gradient descent is another strategy for learning weights of a model. Instead of creating a closed form solution (like OLS) we learn iteratively following the gradient of the loss/cost function. When our data needs to be represented in more complex forms, we often will use some variant of gradient descent to learn complex parameterizations. Gradient Descent also doesn't require the `XᵀX` to be invertable to find a solution. 

In this notebook we will be focusing on minibatch gradient descent, and using 3 learning rate adaptation rules `ConstantLR`, `HeuristicLR`, and `AdaGrad`. All of these have their use in various parts of the literature and in various settings. 

Below you need to implement the function `epoch!` which goes through the data set in minibatches of size `mbgd.n`. Remember to randomize how you go through the data **and** that you are using the correct targets for the data passed to the learning update. In this implementation you will use 

```julia
update!(model, lossfunc, opt, X_batch, Y_batch)
```

to update your model. So you will basically randomize and divide the dataset into batches and call the update function for each batch. These functions are defined in the section on [optimizers](#opt).

"""

# ╔═╡ 9d96ede3-533e-42f7-ada1-6e71980bc6c2
function epoch!(mbgd::MiniBatchGD, model::AbstractModel, lossfunc, opt, X, Y)
	epoch!(mbgd, get_linear_model(model), lossfunc, opt, get_features(lp.model, X), Y)
end

# ╔═╡ 6ff92fca-6d66-4f27-8e09-11a3887e66ba
function train!(mbgd::MiniBatchGD, model::AbstractModel, lossfunc, opt, X, Y, num_epochs)
	train!(mbgd, get_linear_model(model), lossfunc, opt, get_features(model, X), Y, num_epochs)
end

# ╔═╡ 7e777dba-b389-4549-a93a-9b0394646c57
abstract type LossFunction end

# ╔═╡ f380a361-2960-471b-b29a-3bd1fe06252b
md"""
#### (c) Mean Squared Error
"""

# ╔═╡ 6d2d24da-9f3f-43df-9243-fc17f85e0b01
md"""
We will be implementing 1/2 MSE in the loss function.

```math
c(w) = \frac{1}{2n} \sum_i^n (f(x_i) - y_i)^2
```

where $f(x)$ is the prediction from the passed model.
"""

# ╔═╡ 4f43373d-42ee-4269-9862-f53695351ea3
struct MSE <: LossFunction end

# ╔═╡ ada800ba-25e2-4544-a297-c42d8b36a9ff
function loss(lm::AbstractModel, mse::MSE, X, Y)
	0.0
	#### BEGIN SOLUTION
	s=0
	for i in 1:size(X,1)
		s+=(predict(lm,X[i,:])-Y[i])^2	
	end
	mse=s/(2*size(X,1))

	#### END SOLUTION
end

# ╔═╡ 7bea0c90-077f-4eca-b880-02d1289244f3
md"""
#### (d) Gradient of Mean Squared Error
"""

# ╔═╡ 4ea14063-99ca-4caf-a862-fbf9590c68a2
md"""
You will implement the gradient of the MSE loss function `c(w)` in the `gradient` function with respect to `w`, returning a matrix of the same size of `lm.W`.
"""

# ╔═╡ 299116ea-66f3-4e52-ab0f-594249b9dd23
function gradient(lm::AbstractModel, mse::MSE, X::Matrix, Y::Vector)
	∇W = zero(lm.W) # gradients should be the size of the weights
	
	#### BEGIN SOLUTION
	for i in 1:length(∇W)
        for j in 1:length(predict(lm,X))
            ∇W[i] +=(predict(lm,X[j,:])- Y[j])*X[j,i] 
        end
    end
	∇W=∇W/size(X,1)	
	
	#### END SOLUTION
	
	@assert size(∇W) == size(lm.W)
	∇W
end

# ╔═╡ 36c1f5c8-ac43-41ea-9100-8f85c1ee3708
abstract type Optimizer end

# ╔═╡ 159cecd9-de77-4586-9479-383661bf3397
begin
	struct _LR <: Optimizer end
	struct _LF <: LossFunction end
	function gradient(lm::LinearModel, lf::_LF, X::Matrix, Y::Vector)
		sum(X, dims=1)
	end
	function update!(lm::LinearModel, 
		 			 lf::_LF, 
		 			 opt::_LR, 
		 			 x::Matrix,
		 			 y::Vector)
		
		ϕ = get_features(lm, x)
		
		ΔW = gradient(lm, lf, ϕ, y)[1, :]
		lm.W .-= ΔW
	end
end;

# ╔═╡ a3387a7e-436c-4724-aa29-92e78ea3a89f
begin
	# __check_mseGrad 
	__check_mseloss = loss(LinearModel(3, 1), MSE(), ones(4, 3), [1,2,3,4]) == 3.75
	__check_msegrad = all(gradient(LinearModel(3, 1), MSE(), ones(4, 3), [1,2,3,4]) .== -2.5)
	
	__check_MSE = __check_mseloss && __check_msegrad
	
md"""
For this notebook we will only be using MSE, but we still introduce the abstract type LossFunction for the future. Below you will need to implement the `loss` $(_check_complete(__check_mseloss)) function and the `gradient` $(_check_complete(__check_msegrad)) function for MSE.
"""
end

# ╔═╡ a17e5acd-d78d-4fab-9ab2-f01bd888339d
HTML("<h3 id=lossfunc> Loss Functions $(_check_complete(__check_MSE)) </h3>")

# ╔═╡ 0f6929b6-d869-4c3c-90f6-c923b265b164
struct ConstantLR <: Optimizer
	η::Float64
end

# ╔═╡ 8b8fd9b8-a41b-4fef-96b7-a146986c6f82
Base.copy(clr::ConstantLR) = ConstantLR(clr.η)

# ╔═╡ 344092df-c60b-4f8d-8992-cae088664632
function update!(lm::LinearModel, 
				 lf::LossFunction, 
				 opt::ConstantLR, 
				 x::Matrix,
				 y::Vector)

	g = gradient(lm, lf, x, y)
	
	#### BEGIN SOLUTION
	for i in 1:size(lm.W,1)
        lm.W[i]-=opt.η * g[i]
    end

	#### END SOLUTION
end

# ╔═╡ 695d7dea-c398-43dd-a342-c204c050477e
begin
	mutable struct HeuristicLR <: Optimizer
		g_bar::Float64
	end
	HeuristicLR() = HeuristicLR(1.0)
end

# ╔═╡ 7a4f745a-cb65-49d0-80fa-0e67a75df2c1
Base.copy(hlr::HeuristicLR) = HeuristicLR(hlr.g_bar)

# ╔═╡ fae6cbd5-b7fe-4043-a4b6-a4bc07caf7d9
function update!(lm::LinearModel, 
				 lf::LossFunction, 
				 opt::HeuristicLR, 
				 x::Matrix,
				 y::Vector)

	g = gradient(lm, lf, x, y)
	
	#### BEGIN SOLUTION

	s=0
	for i=1:length(g)
		s+=abs(g[i])
	end
	opt.g_bar+=s/(length(g))
	for i in 1:size(lm.W,1)
        lm.W[i]-=g[i]/(1+opt.g_bar)
    end 
	#### END SOLUTION
end

# ╔═╡ 77cda37c-c560-42d8-85b1-7f026d09edfe
md"""
AdaGrad is another technique for adapting the learning rate where we use a different learning rate for every parameter $W_i$

To implement AdaGrad optimizer, we use the following equations:

```math
\begin{align}
G_i &= G_i + g_i^2 \\
W_i &= W_i - \frac{\eta}{\sqrt{G_i + \epsilon}} * g_i
\end{align}
```
where $g$ is the gradient, and $W$ are the weights.

Implement ```AdaGrad```.
"""

# ╔═╡ 1fe7084a-9cf9-48a4-9e60-b712365eded9
begin
	mutable struct AdaGrad <: Optimizer
		η::Float64 # step size
		G::Matrix{Float64} # exponential decaying average
		ϵ::Float64 #
	end
	
	AdaGrad(η) = AdaGrad(η, zeros(1, 1), 1e-5)
	AdaGrad(η, lm::LinearModel) = AdaGrad(η, zero(lm.W), 1e-5)
	AdaGrad(η, lm::AbstractModel) = AdaGrad(η, get_linear_model(model))
	Base.copy(adagrad::AdaGrad) = AdaGrad(adagrad.η, zero(adagrad.G), adagrad.ϵ)
end

# ╔═╡ c2710a60-ebe1-4d01-b6d1-0d6fe45723f9
function update!(lm::LinearModel, 
				 lf::LossFunction,
				 opt::AdaGrad,
				 x::Matrix,
				 y::Vector)

	g = gradient(lm, lf, x, y)
	if size(g) !== size(opt.G) # need to make sure this is of the right shape.
		opt.G = zero(g)
	end
	
	# update opt.v and lm.W
	η, G, ϵ = opt.η, opt.G, opt.ϵ
	
	#### BEGIN SOLUTION
	opt.G+=g.^2
	for i in 1:size(lm.W,1)
        lm.W[i]-=g[i]*opt.η/sqrt(opt.G[i]+opt.ϵ)
    end 

	#### END SOLUTION
	
end

# ╔═╡ 69cf84e2-0aba-4595-8cb0-c082dbccdbe2
function epoch!(mbgd::MiniBatchGD, model::LinearModel, lossfunc, opt, X, Y)
	
	#### BEGIN SOLUTION
# 	A=[X Y]
# 	A = A[shuffle(1:end), :]
	
# 	for i in 1:mbgd.n:size(A, 1)
# 		first_row = i
# 		last_row = i + mbgd.n - 1
# 		if last_row>size(A,1)
# 			batch_i_matrix = A[first_row:last_row-1,:]
# 		else
# 			batch_i_matrix = A[first_row:last_row,:]
# 		end
# 		columns(batch_i_matrix) = [ batch_i_matrix[:,i] for i in 	1:size(batch_i_matrix, 2)]
# 		X_batch = columns(batch_i_matrix)[1]
# 		if length(columns(batch_i_matrix))>1
# 			for i=2:length(columns(batch_i_matrix))-1
# 				X_batch=[X_batch columns(batch_i_matrix)[i]]
# 			end
# 		end
# 		Y_batch=columns(batch_i_matrix)[size(batch_i_matrix,2)]
# 		update!(model, lossfunc, opt, X_batch, Y_batch)

	match=randperm(length(Y))
	b=1
	minibatch=length(Y)/mbgd.n
	for i in 1:minibatch
		X_batch=zeros(mbgd.n,size(X,2))
		Y_batch=zeros(mbgd.n)
		k=1
		for j in match[b:b+mbgd.n-1]
			X_batch[k, :]=X[j, :]
			Y_batch[k]=Y[j]
			k+=1
		end
		update!(model, lossfunc, opt, X_batch, Y_batch)
		b+=mbgd.n
	
	end
	#### END SOLUTION
end

# ╔═╡ 3edea00e-b84b-456f-9ac7-518f856f2b48
begin
	__check_dist_reg = let
		m = DistributionRegressor()
		epoch!(m, 1.0, 1.0)
		epoch!(m, 1.0, 0.1)
		m.b == 0.24683544303797472 && m.a == -0.26582278481012656
	end
	HTML("<h1 id=dist> Q1: Distribution Regression $(_check_complete(__check_dist_reg))")
end

# ╔═╡ 15101fb4-442a-4f2b-a344-76f2cc3da877
# Stochastic Gradent Descent for DistributionModel
function train!(reg::DistributionRegressor, X, Y, num_epochs)
	for i in 1:num_epochs
		for j in randperm(length(X))
			epoch!(reg, X[j], Y[j])
		end
	end
end

# ╔═╡ acf1b36c-0412-452c-ab4d-a388f84fd1fb
begin
	__check_MBGD = let

		lm = LinearModel(3, 1)
		opt = _LR()
		lf = _LF()
		X = ones(10, 3)
		Y = collect(0.0:0.1:0.9)
		mbgd = MiniBatchGD(5)
		epoch!(mbgd, lm, lf, opt, X, Y)
		all(lm.W .== -10.0)
	end
	str = "<h2 id=graddescent> (b) Mini-batch Gradient Descent $(_check_complete(__check_MBGD)) </h2>"
	HTML(str)
end

# ╔═╡ 2782903e-1d2e-47de-9109-acff4595de42
function train!(mbgd::MiniBatchGD, model::LinearModel, lossfunc, opt, X, Y, num_epochs)
	ℒ = zeros(num_epochs + 1)
	ℒ[1] = loss(model, lossfunc, X, Y)
	for i in 1:num_epochs
		epoch!(mbgd, model, lossfunc, opt, X, Y)
		ℒ[i+1] = loss(model, lossfunc, X, Y)
	end
	ℒ
end

# ╔═╡ eb5d3e74-f156-43a1-9966-4880f80a3d60
begin
	_check_ConstantLR = let
		lm = LinearModel(3, 1)
		opt = ConstantLR(0.1)
		lf = MSE()
		X = ones(4, 3)
		Y = [0.1, 0.2, 0.3, 0.4]
		update!(lm, lf, opt, X, Y)
		all(lm.W .== 0.025)
	end
	md"""
	#### (e) Constant Learning Rate $(_check_complete(_check_ConstantLR))

	To update the weights for mini-batch gradient descent, we can use `ConstantLR` optimizer which updates the weights using a constant learning rate `η`
	
	```math
	W = W - η*g
	```
	
	where `g` is the gradient defined by the loss function.
	
	Implement the `ConstantLR` optimizer.
	"""
end

# ╔═╡ 9100674f-c3f9-4b4f-bca5-dd8bef5bc6e9
begin
	_check_HeuristicLR = let
		lm = LinearModel(3, 1)
		opt = HeuristicLR()
		lf = MSE()
		X = ones(4, 3)
		Y = [0.1, 0.2, 0.3, 0.4]
		update!(lm, lf, opt, X, Y)
		println(lm.W)
		all(lm.W .≈ 0.11111111111111)
	end
	md"""
	#### (f) Heuristic Learning Rate $(_check_complete(_check_HeuristicLR))

	To update the weights for mini-batch gradient descent, we can use `HeuristicLR` optimizer which updates the weights using a learning rate `η` that is a function of the gradient. We define the learning rate at time $t$ as:
	
	```math
	\eta_t = (1 + \bar{g}_{t})^{-1}
	```
	where $\bar{g}_{t}$ is an accumulating gradient over time that uses the gradient ```g``` defined by the loss function. We use the following to compute $\bar{g}_{t}$

	```math
	\bar{g}_{t} = \bar{g}_{t-1} + \frac{1}{d+1} \sum_{j=0}^d |g_{t, j}|
	```
	
	Then, we use the update
	
	```math
	W_t = W_t - \eta_t g_t
	```
	Implement the `HeuristicLR` by implementing the adaptive learning rate and update rule.
	"""

end

# ╔═╡ 8dfd4734-5648-42f2-b93f-be304b4b1f27
begin
	 __check_AdaGrad_v, __check_AdaGrad_W = let
		lm = LinearModel(2, 1)
		opt = AdaGrad(0.1, lm)
		X = [0.1 0.5; 
			 0.5 0.0; 
			 1.0 0.2]
		Y = [1, 2, 3]
		update!(lm, MSE(), opt, X, Y)
		true_G = [1.8677777777777768, 0.13444444444444445]
		true_W = [0.09999973230327601, 0.099996281199188]
		all(opt.G .≈ true_G), all(lm.W .≈ true_W)
	end
	
	__check_AdaGrad = __check_AdaGrad_v && __check_AdaGrad_W
	
md"""
#### (g) AdaGrad $(_check_complete(__check_AdaGrad))

	
"""
end

# ╔═╡ a9d3c6c3-1cb7-4417-ba6a-54998841c87d
let
	q1_check= _check_complete(__check_dist_reg)
	q2_a_check = _check_complete(_check_Poly2)
	q2_b_check = _check_complete(__check_MBGD)
	q2_c_check = _check_complete(__check_mseloss)
	q2_d_check = _check_complete(__check_msegrad)
	q2_e_check = _check_complete(_check_ConstantLR)
	q2_f_check = _check_complete(_check_HeuristicLR)
	q2_g_check = _check_complete(__check_AdaGrad)

md"""
# Preamble 

In this assignment, we will implement:
- Q1 [Distributional Regression](#dist) $(q1_check)
- Q2(a) [Polynomial Features](#graddescent) $(q2_a_check)
- Q2(b) [Mini-batch Gradient Descent](#lossfunc) $(q2_b_check)
- Q2(c,d) [Loss functions](#lossfunc) MSE $(q2_c_check), and gradeint of MSE $(q2_d_check)
- Q2(e-g) [Optimizers](#opt): Constant LR $(q2_e_check), Heuristic Stepsize $(q2_f_check), and AdaGrad $(q2_g_check)
"""
end

# ╔═╡ af8acfdf-32bd-43c1-82d0-99008ee4cb3e
HTML("<h3 id=opt> Optimizers $(_check_complete(_check_ConstantLR && __check_AdaGrad)) </h3>")

# ╔═╡ 3738f45d-38e5-415f-a4e6-f8922df84d09
md"""
Below you will need to implement three optimizers

- Constant learning rate $(_check_complete(_check_ConstantLR))
- Heuristic learning rate $(_check_complete(_check_HeuristicLR))
- AdaGrad $(_check_complete(__check_AdaGrad))
"""

# ╔═╡ fa610de0-f8c7-4c48-88d8-f5398ea75ae2
md"""
# Evaluating models

In the following section, we provide a few helper functions and structs to make evaluating methods straightforward. The abstract type `LearningProblem` with children `GDLearningProblem` and `OLSLearningProblem` are used to construct a learning problem. You will notice these structs contain all the information needed to `train!` a model for both gradient descent and for OLS. We also provide the `run` and `run!` functions. These will update the transform according to the provided data and train the model. `run` does this with a copy of the learning problem, while `run!` does this inplace. 

"""

# ╔═╡ d695b118-6d0d-401d-990f-85ba467cc53e
abstract type LearningProblem end

# ╔═╡ 6edc243e-59ac-4c6f-b507-80d3ec13bc21
"""
	GDLearningProblem

This is a struct for keeping a the necessary gradient descent learning setting components together.
"""
struct GDLearningProblem{M<:AbstractModel, O<:Optimizer, LF<:LossFunction} <: LearningProblem
	gd::MiniBatchGD
	model::M
	opt::O
	loss::LF
end

# ╔═╡ 3bdde6cf-3b68-46d3-bf76-d68c20b661e9
Base.copy(lp::GDLearningProblem) = 
	GDLearningProblem(lp.gd, copy(lp.model), copy(lp.opt), lp.loss)

# ╔═╡ 7905f581-1593-4e06-8aaf-faec05c3b306
function run!(lp::GDLearningProblem, X, Y, num_epochs)
	update_transform!(lp.model, X, Y)
	train!(lp.gd, lp.model, lp.loss, lp.opt, X, Y, num_epochs)
end

# ╔═╡ 69b96fc3-dc9c-44de-bc7f-12bb8aba85d1
function run(lp::LearningProblem, args...)
	cp_lp = copy(lp)
	ℒ = run!(cp_lp, args...)
	return cp_lp, ℒ
end

# ╔═╡ eef918a9-b8af-4d41-85b1-bebf1c7889cc
HTML("<h4 id=cv> Run Experiment </h2>")

# ╔═╡ fd75ff49-b5de-48dc-ae89-06bf855d81b2
md"""

Below are the helper functions for running an experiment.

"""

# ╔═╡ d339a276-296a-4378-82ae-fe498e9b5181
"""
	run_experiment(lp, X, Y, num_epochs, runs; train_size)

Using `train!` do `runs` experiments with the same train and test split (which is made by `random_dataset_split`). This will create a copy of the learning problem and use this new copy to train. It will return the estimate of the error.
"""
function run_experiment(lp::LearningProblem, 
						train_data, 
						test_data,	 
						num_epochs,
						runs)

	err = zeros(runs)
	
	for i in 1:runs		
		# train
		cp_lp, train_loss = run(lp, train_data[1], train_data[2], num_epochs)
		
		# test
		Ŷ = predict(cp_lp.model, test_data[1])
		err[i] = sqrt(mean(abs2, test_data[2] - Ŷ))
	end

	err
end

# ╔═╡ 58e626f1-32fb-465a-839e-1f413411c6f3
md"
# Experiments

In this section, we will run three experiments on the different algorithms we implemented above. We provide the data in the `Data` section, and then follow with the three experiments and their descriptions. You will need to analyze and understand the three experiments for the written portion of this assignment.
"

# ╔═╡ 5ec88a5a-71e2-40c1-9913-98ced174341a
md"""
## Data

This section creates the datasets we will use in our comparisons. Feel free to play with them in `let` blocks.
"""

# ╔═╡ 12d0a092-e5d5-4aa1-ac82-5d262f54aa6d
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

# ╔═╡ cd368ac0-1729-4d84-91b3-ccd42ee56583
let
	trainset, testset = splitdataframe(df_hw_norm, 0.1; shuffle=true) do df
		(X=df[!, :height], Y=df[!, :weight]) # create named tuple from DF
	end
	num_runs = 20
	num_epochs = 1
	
	err_d = zeros(num_runs)
	for r in 1:num_runs
		m = DistributionRegressor()
		train!(m, trainset.X, trainset.Y, num_epochs)
		err_d[r] = mean(abs2, getindex.(predict.((m,), testset.X), 1) - testset.Y)
	end
	mean_error_d = mean(err_d)
	std_error_d = sqrt(var(err_d)/num_runs)
	
	println("Test error finished after $num_epochs epochs.")

	println("For DistributionRegressor: The mean is $mean_error_d and the standard error is $std_error_d")

	err_m = zeros(num_runs)
	for r in 1:num_runs
		m = MeanRegressor()
		train!(m, trainset.X, trainset.Y, num_epochs)
		err_m[r] = mean(abs2, getindex.(predict.((m,), testset.X), 1) - testset.Y)
	end
	mean_error_m = mean(err_m)
	std_error_m = sqrt(var(err_m)/num_runs)
	
	println("For MeanRegressor: the mean is $mean_error_m and the standard error is $std_error_m")
	
	err_r = zeros(num_runs)
	for r in 1:num_runs
		m = RangeRegressor()
		train!(m, trainset.X, trainset.Y, num_epochs)
		err_r[r] = mean(abs2, getindex.(predict.((m,), testset.X), 1) - testset.Y)
	end
	mean_error_r = mean(err_r)
	std_error_r = sqrt(var(err_r)/num_runs)
	
	println("For RangeRegressor: the mean is $mean_error_r and the standard error is $std_error_r")
	
	plt = bar(["DistReg" "Mean" "Range"], [mean_error_d mean_error_m mean_error_r], yerr=[std_error_d std_error_m std_error_r], legend=nothing)
	
	plt2 = boxplot(["DistReg" "Mean" "Range"], [err_d err_m err_r], legend=nothing)
	
	plot(plt, plt2)
end

# ╔═╡ d2c516c0-f5e5-4476-b7d6-89862f6f2472
function unit_normalize_columns!(df::DataFrame)
	for name in names(df)
		mn, mx = minimum(df[!, name]), maximum(df[!, name])
		df[!, name] .= (df[!, name] .- mn) ./ (mx - mn)
	end
	df
end

# ╔═╡ 72641129-5274-47b6-9967-fa37c8036552
md"""
### **Admissions Dataset**
"""

# ╔═╡ 90f34d85-3fdc-4e2a-ada4-085154103c6b
admissions_data = let
	data = CSV.read("data/admission.csv", DataFrame, delim=',', ignorerepeated=true)[:, 2:end]
	data[!, 1:end-1] = unit_normalize_columns!(data[:, 1:end-1])
	data
end;

# ╔═╡ 14b329fb-8053-4148-8d24-4458e592e7e3
md"""
## Plotting our data

The `plot_data` function produces two plots that can be displayed horizontally or vertically. The left or top plot is a box plot over the cv errors, the right or bottom plot is a bar graph displaying average cv errors with standard error bars. This function will be used for all the experiments, and you should use this to finish your written experiments.

"""


# ╔═╡ eebf5285-2336-4c07-a4fd-b1fd841dee52
function plot_data(algs, errs; vert=false)
	stderr(x) = sqrt(var(x)/length(x))
	
	plt1 = boxplot(reshape(algs, 1, :),
				   errs,
				   legend=false, ylabel="MSE",
				   pallette=:seaborn_colorblind)
	
	plt2 = bar(reshape(algs, 1, :),
			   reshape(mean.(errs), 1, :),
			   yerr=reshape(stderr.(errs), 1, :),
			   legend=false,
			   pallette=:seaborn_colorblind,
			   ylabel=vert ? "MSE" : "")
	
	if vert
		plot(plt1, plt2, layout=(2, 1), size=(600, 600))
	else
		plot(plt1, plt2)
	end
end

# ╔═╡ b689d666-37da-40f7-adb8-44aa2b9f5139
md"""
## (h) Non-linear feature transforms

We will compare the linear to non-linear models using the a simulated data set and the admissions dataset.

To run these experiments use $(@bind __run_nonlinear PlutoUI.CheckBox())
"""

# ╔═╡ 55ce32ff-dec3-4bd4-b6a2-95483e7637e9
md"""
This first expereiment uses a simulated training set which aims to predict this function

```julia
f(x) = sin(π*x[1]*x[2]^2) + cos(π*x[3]^3) + x[5]*sin(π*x[4]^4) + 0.001*randn()
```

from inputs $\mathbf{x} \in [0.0, 1.0]^5$. We compare a linear representation and a Polynomial (p=2) representation with two baselines.
"""

# ╔═╡ d381d944-5069-4f16-8194-bd49eb2fe1cd
let
	if __run_nonlinear
		algs = ["Random", "Mean", "Linear", "Poly"]
		non_linear_problems_sin = [
			GDLearningProblem(
				MiniBatchGD(30),
				RangeModel(),
				ConstantLR(0.0),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				MeanModel(),
				ConstantLR(0.0),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(5, 1),
				ConstantLR(1.0),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				Polynomial2Model(5, 1),
				ConstantLR(0.5),
				MSE())
			];
		nonlinear_errs_sin = let
			Random.seed!(2)
			X = rand(500, 5)
			f(x) = sin(π*x[1]*x[2]^2) + cos(π*x[3]^3) + x[5]*sin(π*x[4]^4) + 0.001*randn()
			Y = [f(x) for x in eachrow(X)]
			Y .= (Y.-minimum(Y))/(maximum(Y) - minimum(Y))
			plot(Y)
			errs = Vector{Float64}[]
			
			train_size=400
			
			rp = randperm(length(Y))
			train_idx = rp[1:train_size]
			test_idx = rp[train_size+1:end]
			train_data = (X[train_idx, :], Y[train_idx]) 
			test_data = (X[test_idx, :], Y[test_idx])
			
			for (idx, prblms) in enumerate(non_linear_problems_sin)
				cv_err = run_experiment(prblms, train_data, test_data, 10, 50)
				push!(errs, cv_err)
			end
			errs
		end

		plot_data(algs, nonlinear_errs_sin, vert=true)
		
	end
end

# ╔═╡ 80406819-83d2-4625-8ed3-959c127e3e2c
md"""
The following experiment uses the addmistions dataset, which you should report. **You can get the average error and standard error to report from the plot or from the terminal where your ran this notebook**.
"""

# ╔═╡ 5a4ac4ed-6ed5-4d28-981e-8bf7b6b8889d
let
	if __run_nonlinear
		algs = ["Range", "Mean", "Linear", "Poly"]
		non_linear_problems = [
			GDLearningProblem(
				MiniBatchGD(30),
				RangeModel(),
				ConstantLR(0.0),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				MeanModel(),
				ConstantLR(0.0),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(7, 1),
				ConstantLR(0.5),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				Polynomial2Model(7, 1),
				ConstantLR(0.2),
				MSE()),
		]
		nonlinear_errs = let

			Random.seed!(2)
			data = (X=Matrix(admissions_data[:, 1:end-1]), Y=admissions_data[:, end])
			errs = Vector{Float64}[]
			
			X, Y = data.X, data.Y
			train_size=350
			
			rp = randperm(length(Y))
			train_idx = rp[1:train_size]
			test_idx = rp[train_size+1:end]
			train_data = (X[train_idx, :], Y[train_idx]) 
			test_data = (X[test_idx, :], Y[test_idx])
			
			for (idx, prblms) in enumerate(non_linear_problems)
				err = run_experiment(
					prblms, train_data, test_data, 10, 50)
				push!(errs, err)
			end
			errs
		end
		
		num_runs = size(nonlinear_errs[4])
		stderr(x) = sqrt(var(x)/length(x))
		mean_error_linear = mean(nonlinear_errs[3])
		mean_error_poly = mean(nonlinear_errs[4])
		
		std_error_linear = stderr(nonlinear_errs[3])
		std_error_poly = stderr(nonlinear_errs[4])
		
		println("Average error on test set for Linear model is $mean_error_linear with standard error $std_error_linear")
		
		println("Average error on test set for Polynomial model is $mean_error_poly with standard error $std_error_poly")
		
		plot_data(algs, nonlinear_errs; vert=true)
		
	end
end

# ╔═╡ 0903dd95-5525-44e5-891d-acbe2fb2190f
md"""
## (i) Learning Rate adapatation

We will compare the different learning rate algorithms on a subset of the [Admissions dataset](). From this dataset we will be predicting the likelihood of admission.

To run this experiment click $(@bind __run_lra PlutoUI.CheckBox())

**You can get the average error and standard error to report from the plot or from the terminal where your ran this notebook**.
"""

# ╔═╡ c01ff616-e570-4013-a0b2-d97fcda6f279
let
	if __run_lra
		algs_lr = ["Mean", "Constant", "Heuristic", "AdaGrad"]
		lr_adapt_problems = [
			GDLearningProblem(
				MiniBatchGD(30),
				MeanModel(),
				ConstantLR(0.0),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(7, 1),
				ConstantLR(0.05),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(7, 1),
				HeuristicLR(),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(7, 1),
				AdaGrad(0.1),
				MSE()),
		];
		lr_errs = let
			
			Random.seed!(2)
			test_idx = 1
			data = (X=Matrix(admissions_data[:, 1:end-1]), Y=admissions_data[:, end])
			@show size(data.X)
			errs = Vector{Float64}[]
			
			X, Y = data.X, data.Y
			train_size=350
			
			rp = randperm(length(Y))
			train_idx = rp[1:train_size]
			test_idx = rp[train_size+1:end]
			train_data = (X[train_idx, :], Y[train_idx]) 
			test_data = (X[test_idx, :], Y[test_idx])
			
			for (idx, prblms) in enumerate(lr_adapt_problems)

				err = run_experiment(prblms, train_data, test_data, 5, 50)
				push!(errs, err)
			end
			errs
		end
		num_runs = size(lr_errs[4])
		stderr(x) = sqrt(var(x)/length(x))
		
		mean_error_constantLR = mean(lr_errs[2])
		mean_error_HeuristicLR = mean(lr_errs[3])
		mean_error_AdaGrad = mean(lr_errs[4])
		
		std_error_constantLR = stderr(lr_errs[2])
		std_error_HeuristicLR = stderr(lr_errs[3])
		std_error_AdaGrad = stderr(lr_errs[4])
		
		println("Average error on test set for Linear model with ConstantLR is $mean_error_constantLR with standard error $std_error_constantLR")
		
		println("Average error on test set for Linear model with HeuristicLR is $mean_error_HeuristicLR with standard error $std_error_HeuristicLR")
		
		println("Average error on test set for Linear model with AdaGrad is $mean_error_AdaGrad with standard error $std_error_AdaGrad")
		
		plot_data(algs_lr, lr_errs)
	end

end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.9.10"
DataFrames = "~1.2.2"
Distributions = "~0.25.23"
MultivariateStats = "~0.8.0"
PlutoUI = "~0.7.16"
StatsBase = "~0.33.12"
StatsPlots = "~0.14.28"
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
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "74147e877531d7c172f70b492995bc2b5ca3a843"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.10"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "3533f5a691e60601fe60c90d8bc47a27aa2907ec"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.0"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

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
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

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
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

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
git-tree-sha1 = "09d9eaef9ef719d2cd5d928a191dc95be2ec8059"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.5"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "d249ebaa67716b39f91cf6052daf073634013c0f"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.23"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FilePathsBase]]
deps = ["Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "d962b5a47b6d191dbcd8ae0db841bc70a05a3f5b"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.13"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

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
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "d189c6d2004f63fd3c91748c458b09f26de0efaa"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.61.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cafe0823979a5c9bff86224b3b8de29ea5a44b2e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.61.0+0"

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

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "5efcf53d798efede8fee5b2c8b09284be359bf24"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.2"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "19cb49649f8c41de7fea32d089d37de917b553da"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.0.1"

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

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "f0c6489b12d28fb4c2103073ec7452f3423bd308"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.1"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

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
deps = ["ChainRulesCore", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "6193c3815f13ba1b78a51ce391db8be016ae9214"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.4"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

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
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

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
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

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
git-tree-sha1 = "d911b6a12ba974dabe2291c6d450094a7226b372"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.1"

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
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "ca7d534a27b1c279f05cd094196cb70c35e3d892"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.2"

[[PlutoUI]]
deps = ["Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "4c8a7d080daca18545c56f1cac28710c362478f3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.16"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

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
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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
git-tree-sha1 = "f45b34656397a1f6e729901dc9ef679610bd12b5"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.8"

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
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "eb007bb78d8a46ab98cd14188e3cec139a4476cf"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.28"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

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

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

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
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─75441ce6-2137-4fcf-bba2-6ed67b9acb59
# ╠═dc004086-db6e-4813-841e-d427520402f7
# ╠═c9797979-7d16-4ba4-8f82-c7ec33f6e517
# ╟─693a3933-c1c2-4249-8c03-f5151267222f
# ╟─bc0c5abe-1c2a-4e5c-bb0c-8d76b5e81133
# ╠═def97306-1703-42bc-bc09-da623c545e87
# ╠═14c30b50-157f-40bf-b0a6-0232aa2705c6
# ╠═4a196e82-fe94-49fe-ab87-d57e2d7fbd34
# ╠═a9d3c6c3-1cb7-4417-ba6a-54998841c87d
# ╠═3edea00e-b84b-456f-9ac7-518f856f2b48
# ╠═dba201d8-a0a5-4444-bcf8-913696655a49
# ╠═43870efd-4050-4b75-9a17-504ff66ddf70
# ╠═983693ea-4401-4115-bd40-47aa28caee2d
# ╠═15101fb4-442a-4f2b-a344-76f2cc3da877
# ╟─01682bea-c735-427f-be82-9b6d8b06c028
# ╠═279a3b98-0124-4d73-8a54-93947deaaf16
# ╠═be72d249-9c8b-4f87-9404-e53c27ec83fb
# ╠═d011cc2f-c39d-475c-be03-fa2907126c91
# ╟─e102cde6-3461-4dea-b8f2-dec803edd6fa
# ╠═51c4bc05-cecf-4695-94f4-5e39da34a8c5
# ╠═cd368ac0-1729-4d84-91b3-ccd42ee56583
# ╠═a7aecd21-13f2-4cf2-8239-3a3d708602c9
# ╠═e3c3e903-e2c2-4de0-a3b2-56a27709e8c3
# ╟─3237c4fc-56d1-4135-a8d2-cc6e88f2f5c0
# ╠═5080cc19-d13f-4236-b49e-e338108dda80
# ╠═a35944ae-7bbb-4f3c-bbe0-d2bbb57cd70b
# ╠═e7712bd3-ea7e-4f4a-9efc-041b4b2be987
# ╠═4855856c-9eed-4ab5-9170-7cc9ed4115fc
# ╠═725cadb3-3c4f-4344-9078-7ea7424bde1d
# ╠═7b513344-1cad-4eef-9faf-e77ba176323e
# ╠═4f4029a2-c590-4bd3-a0db-d2380d4b4620
# ╠═dcfecc35-f25c-4856-8429-5c31d94d0a42
# ╠═d45dedf8-e46a-4ffb-ab45-f9d211d3a8ca
# ╠═7cd46d84-a74c-44dc-8339-68010924bc39
# ╠═8745fec1-47c8-428b-9ea4-1e6828618830
# ╠═2d43a3ba-2a2c-4114-882f-5834d42e302a
# ╠═c59cf592-a893-4ffa-b247-51d94c7cdb1a
# ╟─ded749bf-b9fa-4e2b-b15f-0693d820a9c3
# ╠═2e69a549-aab4-4c88-aad8-dffb822d644f
# ╠═0608c93d-2f82-470c-8d9f-ca79af6b2612
# ╠═0ba5f9c8-5677-40e9-811b-25546e0df207
# ╠═acf1b36c-0412-452c-ab4d-a388f84fd1fb
# ╠═159cecd9-de77-4586-9479-383661bf3397
# ╠═d9935cc8-ec24-47e9-b39a-92c21377a161
# ╠═5714c84f-1653-4c4a-a2e4-003d8560484a
# ╠═69cf84e2-0aba-4595-8cb0-c082dbccdbe2
# ╠═9d96ede3-533e-42f7-ada1-6e71980bc6c2
# ╠═6ff92fca-6d66-4f27-8e09-11a3887e66ba
# ╠═2782903e-1d2e-47de-9109-acff4595de42
# ╠═a17e5acd-d78d-4fab-9ab2-f01bd888339d
# ╠═a3387a7e-436c-4724-aa29-92e78ea3a89f
# ╠═7e777dba-b389-4549-a93a-9b0394646c57
# ╠═f380a361-2960-471b-b29a-3bd1fe06252b
# ╠═6d2d24da-9f3f-43df-9243-fc17f85e0b01
# ╠═4f43373d-42ee-4269-9862-f53695351ea3
# ╠═ada800ba-25e2-4544-a297-c42d8b36a9ff
# ╟─7bea0c90-077f-4eca-b880-02d1289244f3
# ╠═4ea14063-99ca-4caf-a862-fbf9590c68a2
# ╠═299116ea-66f3-4e52-ab0f-594249b9dd23
# ╠═af8acfdf-32bd-43c1-82d0-99008ee4cb3e
# ╠═3738f45d-38e5-415f-a4e6-f8922df84d09
# ╠═36c1f5c8-ac43-41ea-9100-8f85c1ee3708
# ╠═eb5d3e74-f156-43a1-9966-4880f80a3d60
# ╠═0f6929b6-d869-4c3c-90f6-c923b265b164
# ╠═8b8fd9b8-a41b-4fef-96b7-a146986c6f82
# ╠═344092df-c60b-4f8d-8992-cae088664632
# ╠═9100674f-c3f9-4b4f-bca5-dd8bef5bc6e9
# ╠═695d7dea-c398-43dd-a342-c204c050477e
# ╠═7a4f745a-cb65-49d0-80fa-0e67a75df2c1
# ╠═fae6cbd5-b7fe-4043-a4b6-a4bc07caf7d9
# ╠═8dfd4734-5648-42f2-b93f-be304b4b1f27
# ╠═77cda37c-c560-42d8-85b1-7f026d09edfe
# ╠═1fe7084a-9cf9-48a4-9e60-b712365eded9
# ╠═c2710a60-ebe1-4d01-b6d1-0d6fe45723f9
# ╠═fa610de0-f8c7-4c48-88d8-f5398ea75ae2
# ╠═d695b118-6d0d-401d-990f-85ba467cc53e
# ╠═6edc243e-59ac-4c6f-b507-80d3ec13bc21
# ╠═3bdde6cf-3b68-46d3-bf76-d68c20b661e9
# ╠═7905f581-1593-4e06-8aaf-faec05c3b306
# ╠═69b96fc3-dc9c-44de-bc7f-12bb8aba85d1
# ╠═eef918a9-b8af-4d41-85b1-bebf1c7889cc
# ╟─fd75ff49-b5de-48dc-ae89-06bf855d81b2
# ╠═d339a276-296a-4378-82ae-fe498e9b5181
# ╠═58e626f1-32fb-465a-839e-1f413411c6f3
# ╠═5ec88a5a-71e2-40c1-9913-98ced174341a
# ╠═12d0a092-e5d5-4aa1-ac82-5d262f54aa6d
# ╠═d2c516c0-f5e5-4476-b7d6-89862f6f2472
# ╠═72641129-5274-47b6-9967-fa37c8036552
# ╠═90f34d85-3fdc-4e2a-ada4-085154103c6b
# ╠═14b329fb-8053-4148-8d24-4458e592e7e3
# ╠═eebf5285-2336-4c07-a4fd-b1fd841dee52
# ╠═b689d666-37da-40f7-adb8-44aa2b9f5139
# ╠═55ce32ff-dec3-4bd4-b6a2-95483e7637e9
# ╠═d381d944-5069-4f16-8194-bd49eb2fe1cd
# ╠═80406819-83d2-4625-8ed3-959c127e3e2c
# ╟─5a4ac4ed-6ed5-4d28-981e-8bf7b6b8889d
# ╟─0903dd95-5525-44e5-891d-acbe2fb2190f
# ╠═c01ff616-e570-4013-a0b2-d97fcda6f279
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
