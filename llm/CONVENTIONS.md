Follow these principles:
- use math notation when possible instead of the function names. i.e. x ⋅ y instead of the function dot(x, y)
- Don't write 1:length(num_steps), use eachindex(A...), eachindex(::IndexStyle, A::AbstractArray...)
- Use @fastmath to use faster versions of common mathematical functions and use @cuda fastmath=true for even faster square roots for CUDA operations.
- Helper functions should not start with an underscore _. They should look like any other function, described as an action, ideally a single word with it's types giving context.
- Make the names generic. Don't have the type in the variable names. 
- Do Verb-oriented naming: map, filter, reduce, predict, train!, update!
- Do Predicate Prefixes for functions that return a BOOL: isnothing, ismissing, isapprox, haskey
- Do not do Narrative or Sentence-Style Naming. Multiple actions should be composed.
- Do not use the Dataframes package unless it's already being used or was asked for.
- Use logging @info to output information instead of printing to terminal

### Use Unicode when variables represent math symbols
```julia
t₀ = rand(1:length(T) - window - 1)
xₜ = T[t₀ : t₀ + window - 1]
```

### Don't use types unless dispatching
GOOD: No types needed  
```julia
double(x) = 2 * x
square(x) = x * x
```

GOOD: Use types only for dispatching  
```julia
format(x::String) = lowercase(x)
format(x::Int) = string(x)
```

BAD: Unnecessary type annotations  
```julia
double(x::Int) = 2 * x
square(x::Float64) = x * x
```

### Use multiple dispatch to simplify function names
GOOD: One function name, use dispatch  
```julia
format(x::String) = lowercase(x)
format(x::Int) = string(x)
```

BAD: Different function names for each type  
```julia
format_string(x::String) = lowercase(x)
format_int(x::Int) = string(x)
```

### Break functions into multiple definitions

Writing a function as many small definitions allows the compiler to directly call the most applicable code, or even inline it.

Here is an example of a "compound function" that should really be written as multiple definitions:
```julia
using LinearAlgebra

function mynorm(A)
    if isa(A, Vector)
        return sqrt(real(dot(A,A)))
    elseif isa(A, Matrix)
        return maximum(svdvals(A))
    else
        error("mynorm: invalid argument")
    end
end
```

This can be written more concisely and efficiently as:
```julia
mynorm(x::Vector) = sqrt(real(dot(x, x)))
mynorm(A::Matrix) = maximum(svdvals(A))
```

### Separate kernel functions (aka, function barriers)

Many functions follow a pattern of performing some set-up work, and then running many iterations to perform a core computation. Where possible, it is a good idea to put these core computations in separate functions. For example, the following contrived function returns an array of a randomly-chosen type:
```julia 
julia> function strange_twos(n)
           a = Vector{rand(Bool) ? Int64 : Float64}(undef, n)
           for i = 1:n
               a[i] = 2
           end
           return a
       end;

julia> strange_twos(3)
3-element Vector{Int64}:
 2
 2
 2
```

This should be written as:
```julia
julia> function fill_twos!(a)
           for i = eachindex(a)
               a[i] = 2
           end
       end;

julia> function strange_twos(n)
           a = Vector{rand(Bool) ? Int64 : Float64}(undef, n)
           fill_twos!(a)
           return a
       end;

julia> strange_twos(3)
3-element Vector{Int64}:
 2
 2
 2
 ```

Julia's compiler specializes code for argument types at function boundaries, so in the original implementation it does not know the type of a during the loop (since it is chosen randomly). Therefore the second version is generally faster since the inner loop can be recompiled as part of fill_twos! for different types of a.

The second form is also often better style and can lead to more code reuse.

This pattern is used in several places in Julia Base. For example, see vcat and hcat in abstractarray.jl, or the fill! function, which we could have used instead of writing our own fill_twos!.

Functions like strange_twos occur when dealing with data of uncertain type, for example data loaded from an input file that might contain either integers, floats, strings, or something else.

### Avoid nested control flow. Use helper functions.
GOOD: Flat structure using helpers  
```julia
is_valid(item) = item isa String && length(item) > 5
to_output(item) = println(uppercase(item))
process(data) = [to_output(item) for item in data if is_valid(item)]
```

BAD: Deep nesting  
```julia
function process(data)
    for item in data
        if item isa String
            if length(item) > 5
                println(uppercase(item))
            end
        end
    end
end
```
### Do not have single-use variables with the same name as the function  
Use the function directly in the expression if the operation is simple.

GOOD: Direct calculation  
```julia
sharpe(returns) = mean(returns) / (std(returns) + eps())
```

BAD: Redundant single-use variables  
```julia
function sharpe_ratio(returns::Vector{Float64})::Float64
    mean_return = mean(returns)
    std_dev = std(returns)
    return mean_return / (std_dev + eps())
end
```

### Use logging instead of print statements
GOOD:  
```julia
@info "Computation started"
result = compute()
@info "Result: $result"
```

BAD:  
```julia
println("Computation started")
result = compute()
println("Result: ", result)
```

### Don't use module prefixes
GOOD:  
```julia
using Statistics

mean(data)
```

BAD:  
```julia
Statistics.mean(data)
```

### Use `eps()` instead of conditional logic for division by zero
GOOD:  
```julia
mean_return / (std_dev + eps())
```

BAD:  
```julia
std_dev == 0.0 ? 0.0 : mean_return / std_dev
```
### Other Rules
- If a function modifys their arguments, just append with !.
- Avoid variable names that merely repeat the type when a type annotation is already specified.
- Don't Handle potential empty vector cases unless specified.
- Write functions, not just scripts in global scope. Don't wrap many functions in one larger main(), just keep those in global scope.
- Don't refactor existing functions unless specified or necessary.
- I'm running in REPL in Julia you don't need to save or display plots usually. plot() is enough.
- Use native eps() function in Julia, than defining epsilon. Don't do eps(Float64) since that's default.
- Keep to minimal code edits, don't refactor existing code unless specified.
- Don't write comments or docstrings unless asked
- For non trivial requests ask questions before commiting any code and provide examples of potential solutions.
- Don't use pythonic shorthands like df, use the full name like dataframe.
- In Julia, use Ternary operators instead of the ifelse function. Multiline if elsif is fine though.
- Keep your edits minimal to the directions given. Don't rewrite other functions unless necessary.
- Write in the same style of the provided code.
- prefer dot notation over for loops
- Code should be easy to read and understand.
- Keep the code as simple as possible. Avoid unnecessary complexity.
- Use meaningful names for variables, functions, etc. Names should reveal
  intent.
- Functions should be small and do one thing well. They should not exceed a few
  lines.
- Function names should describe the action being performed.
- Prefer fewer arguments in functions. Ideally, aim for no more than two or
  three.
- Do not write comments or docstrings. The code should self document.
- Properly handle errors and exceptions to ensure the software's robustness.
- Use exceptions rather than error codes for handling errors.
- Instead of configuration use default values in the function or struct
- Don't use CONSTS if they're only being used once 
- When importing new packages, make sure to install them. 
- do not write new validations, error handling, or checks unless explicitly mentioned.
- Don't use array[1] do first(array), and other functions like only, firstindex, last.
- Do import entire packages in a single line like `using DataFrames, RollingFunctions, Statistics`

Dataframes.jl
- Use dataframe.col instead of dataframe[!,:col] if only referencing 1 column. If doing multiple columns then dataframe[!, [:datetime, :close, :rolling_skewness, :rolling_kurtosis]] is appropriate.