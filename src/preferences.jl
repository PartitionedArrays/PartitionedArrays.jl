
"""
    set_default_find_rcv_ids(algorithm::String)

Sets the default algorithm to discover communication neighbors. The available algorithms are:

- `gather_scatter`: Gathers neighbors in a single processor, builds the communications graph 
  and then scatters the information back to all processors. See [`find_rcv_ids_gather_scatter`](@ref).

- `ibarrier`: Implements Alg. 2 in https://dl.acm.org/doi/10.1145/1837853.1693476. See [`find_rcv_ids_ibarrier`](@ref).

Feature only available in Julia 1.6 and later due to restrictions from `Preferences.jl`.
"""
function set_default_find_rcv_ids(algorithm::String)
    if !(algorithm in ("gather_scatter", "ibarrier"))
        throw(ArgumentError("Invalid algorihtm: \"$(algorithm)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("default_find_rcv_ids" => algorithm)
    @info("New deafult algorithm set; restart your Julia session for this change to take effect!")
end

@static if VERSION >= v"1.6"
    const default_find_rcv_ids_algorithm = @load_preference("default_find_rcv_ids", "gather_scatter")
else
    const default_find_rcv_ids_algorithm = "gather_scatter"
end
