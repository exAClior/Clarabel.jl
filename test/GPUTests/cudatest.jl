using Revise
using JuMP, Clarabel, ClarabelBenchmarks

model = Model(Clarabel.Optimizer)
set_optimizer_attribute(model,"max_iter", 500)
set_optimizer_attribute(model,"direct_solve_method", :cudss)
ClarabelBenchmarks.opf_socp_case9591_goc(model)
# ClarabelBenchmarks.cblib_pow_HMCR_n20_m1600(model)
# ClarabelBenchmarks.cblib_exp_LogExpCR_n20_m1600(model)
solver = model.moi_backend.optimizer.model.optimizer.solver
# KKTgpu = solver.kktsystem.kktsolver.KKTgpu
# KKTcpu = solver.kktsystem.kktsolver.KKTcpu


