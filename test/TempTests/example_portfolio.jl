using LinearAlgebra, SparseArrays, Random, JuMP, Test
using Mosek, MosekTools
using Clarabel

# include("../../src\\Clarabel.jl")
using StatProfilerHTML

#############################################################
# The example for parallel computation on GPU
#############################################################

## generate the data
rng = Random.MersenneTwister(1)
k = 5; # number of factors
n = k * 10; # number of assets
D = spdiagm(0 => rand(rng, n) .* sqrt(k))
F = sprandn(rng, n, k, 0.5); # factor loading matrix
μ = (3 .+ 9. * rand(rng, n)) / 100. # expected returns between 3% - 12%
γ = 1.0; # risk aversion parameter
d = 1 # we are starting from all cash
x0 = zeros(n);

# ## Transaction costs
# In the model above we assume that trading the assets is free and does not impact the market. However, this is clearly not the case in reality. To make the example more realistic consider the following cost $c_j$ associated with the trade $δ_j = x_j - x_j^0$:
# $$
# c_j(\delta_j) = a_j |\delta_j|
# $$
# where the first term models the bid-ask spread and broker fees for asset $j$. 

#-
a = 1e-3
b = 1e-1
γ = 1.0;
# model = JuMP.Model(Mosek.Optimizer)

model = JuMP.Model(Clarabel.Optimizer)

# set_optimizer_attribute(model, "direct_solve_method", :qdldl)
set_optimizer_attribute(model, "direct_kkt_solver", false) 
set_optimizer_attribute(model, "max_iter", 20) 
set_optimizer_attribute(model, "static_regularization_enable", false) 
set_optimizer_attribute(model, "tol_gap_abs", 1e-5)
set_optimizer_attribute(model, "tol_gap_rel", 1e-5)
set_optimizer_attribute(model, "tol_feas", 1e-5)
set_optimizer_attribute(model, "tol_ktratio", 1e-4)
# set_optimizer_attribute(model, "min_primaldual_step_length", 1e-1)

@variable(model, x[1:n])
@variable(model, y[1:k])   #this is never used in the model?
@variable(model, s[1:n])
@variable(model, t[1:n])
@objective(model, Min, x' * D * x + y' * y - 1/γ * μ' * x);
@constraint(model, y .== F' * x);
@constraint(model, x .>= 0);

# transaction costs

@constraint(model, sum(x) + a * sum(s) == d + sum(x0) );
# SOC reformulation
# @constraint(model, [i = 1:n], x[i] - x0[i] <= s[i]); # model the absolute value with slack variable s
# @constraint(model, [i = 1:n], x0[i] - x[i] <= s[i]);
@constraint(model, [i = 1:n], x0[i] - x[i] == t[i]) 
@constraint(model, [i = 1:n], [s[i], t[i]] in MOI.SecondOrderCone(2));

# @constraint(model, sum(x) + a * sum(s) + b * sum(t) == d + sum(x0) );
# @constraint(model, [i = 1:n], x[i] - x0[i] <= s[i]); # model the absolute value with slack variable s
# @constraint(model, [i = 1:n], x0[i] - x[i] <= s[i]);
# @constraint(model, [i = 1:n], [t[i], 1, x[i] - x0[i]] in MOI.PowerCone(2/3));
JuMP.optimize!(model)
# Let's look at the expected return and the total transaction cost:

x_opt = JuMP.value.(x);
y_opt = JuMP.value.(y);
s_opt = JuMP.value.(s);
expected_return = dot(μ, x_opt)
#-
expected_risk = dot(y_opt, y_opt)
#-
transaction_cost = a * sum(s_opt)
