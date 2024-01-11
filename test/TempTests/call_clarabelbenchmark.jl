using Revise
using Clarabel, JuMP
using ClarabelBenchmarks


model = Model(Clarabel.Optimizer)
set_optimizer_attribute(model, "direct_kkt_solver", false) 
set_optimizer_attribute(model, "device", :gpu)
set_optimizer_attribute(model, "preconditioner", 3)
# set_optimizer_attribute(model, "iterative_refinement_abstol", 1e-12)
# set_optimizer_attribute(model, "iterative_refinement_reltol", 1e-12)
# set_optimizer_attribute(model, "static_regularization_constant", 1e-8)

# Conic Problems you will be interested in
# ClarabelBenchmarks.opf_socp_case30_ieee(model)              #SOCP in optimal power flow
# ClarabelBenchmarks.cblib_socp_chainsing_1000_1(model)     #SOCP in cblib
# ClarabelBenchmarks.cblib_exp_LogExpCR_n20_m400(model)     #Exponential cone
ClarabelBenchmarks.cblib_pow_HMCR_n20_m1600(model)       #Power cone 