
function solution_finalize!(
	solution::DefaultSolution{T},
	data::DefaultProblemData{T},
	variables::DefaultVariables{T},
	info::DefaultInfo{T},
	settings::Settings{T},
	use_gpu::Bool
) where {T}

	solution.status  = info.status
	solution.obj_val = info.cost_primal
	solution.obj_val_dual = info.cost_dual

    #if we have an infeasible problem, normalize
    #using κ to get an infeasibility certificate.
    #Otherwise use τ to get a solution.
    if status_is_infeasible(info.status)
        scaleinv = one(T) / variables.κ
		solution.obj_val = NaN
		solution.obj_val_dual = NaN
    else
        scaleinv = one(T) / variables.τ
    end

	#also undo the equilibration
	if use_gpu
		d = data.equilibration.d_gpu
		e = data.equilibration.e_gpu 
		einv = data.equilibration.einv_gpu
		map = data.presolver.reduce_map_gpu
	else
		d = data.equilibration.d
		e = data.equilibration.e 
		einv = data.equilibration.einv
		map = data.presolver.reduce_map
	end
	cscale = data.equilibration.c[]

	@. solution.x = variables.x * d * scaleinv

	if !isnothing(map) 
		@. solution.z[map.keep_index] = variables.z * e * (scaleinv / cscale)
		@. solution.s[map.keep_index] = variables.s * einv * scaleinv

		#eliminated constraints get huge slacks 
		#and are assumed to be nonbinding 
		@. solution.s[!map.keep_logical] = T(data.presolver.infbound)
		@. solution.z[!map.keep_logical] = zero(T)

	else
		@. solution.z = variables.z * e * (scaleinv / cscale)
		@. solution.s = variables.s * einv * scaleinv
	end
 

	solution.iterations  = info.iterations
	solution.solve_time  = info.solve_time
	solution.r_prim 	   = info.res_primal
	solution.r_dual 	   = info.res_dual

	return nothing

end



function Base.show(io::IO, solution::DefaultSolution)
	print(io,">>> Clarabel - Results\nStatus: ")
	if solution.status == SOLVED
		status_color = :green
	else
		status_color = :red
	end
	printstyled(io,"$(string(solution.status))\n", color = status_color)
	println(io,"Iterations: $(solution.iterations)")
    println(io,"Objective: $(@sprintf("%#.4g", solution.obj_val))")
    println(io,"Solve time: ",TimerOutputs.prettytime(solution.solve_time*1e9))

end
