include("..\\MyML\\MyML.jl")
using PyPlot, LinearAlgebra, .MyML
# -------------------------------------------------------------
# this function allows us to generate random data where we want
# -------------------------------------------------------------
function generate_rnd(d,a,b)
    p = rand(d)
    r = zeros(d)
    for i=1:d
        r[i] = p[i]*(b-a) + a
    end
    return r
end
# ---------------------------
# Decide what function to use
# ---------------------------
function f(x)
    return sum([v^2 for v in x])
end
# ------------------
# Auxiliar functions
# ------------------
function distance(v1,v2)
    d = 0
    for i=1:length(v1)
        d += (v1[i]-v2[i])^2
    end
    return d^0.5
end

function sub_lists_m(l,m)
    sublists = []
    a,b = 1,m
    while b <= length(l)
        push!(sublists, l[a:b])
        a = b+1
        b = b+m
    end
    if a < length(l)
        push!(sublists, l[a:length(l)])
    end
    return sublists
end
# ---------------
# kernel and alpha function
# ---------------
function φ(z)
    return MathConstants.e^(-(1/2)*(z^2))
end

function α(x,xn,r)
    return φ(distance(xn,x)/r)
end

# Solves Linear Regression to get best w
function find_w(S,C,r)
    neighbors = [k for k in keys(S)]
    y = [S[p] for p in neighbors]
    N = length(neighbors)
    k = length(C)
    Z = zeros(N,k+1)
    Z[:,1] = ones(N)
    for i=1:N
        for j=2:k+1
            Z[i,j] = α(neighbors[i],C[j-1],r)
        end
    end
    if issuccess(lu(Z'Z,check=false))
        w = inv(Z'Z)Z'y
    else
        println("WARNING: Found Z'Z not inversible")
    end
    return w
end

function h(x,w,r,C)
    return w[1] + sum([w[n]*α(x,C[n-1],r) for n=2:length(w)])
end

function m_cross_validate_r(S,r_interval,m)
    r_interval = [r for r in r_interval if r != 0]
    real_r = 0
    minor_error = 10^10
    neighbors = [k for k in keys(S)]
    # Divide my data in parts of m
    sub_datasets = sub_lists_m(neighbors,m)
    println("Starting cross validation")
    println("Progress: 0%")
    prctg = 0
    total_length = length(r_interval)
    im_in = 0
    for q in r_interval
        im_in += 1
        q_error = 0
        for sd in sub_datasets
            U = Dict()
            for k in keys(S)
                if !(k in sd)
                    U[k] = S[k]
                end
            end
            new_neighbors = [k for k in keys(U)]
            w = find_w(U,new_neighbors,q)
            this_error = 0
            for p in sd
                this_error += (h(p,w,q,new_neighbors) - S[p])^2
            end
            q_error += this_error
        end
        if q_error < minor_error
            real_r = q
            minor_error = q_error
        end
        this_prctg = floor(Int,round(im_in*100/total_length, digits=0))
        if this_prctg != prctg
            prctg = this_prctg
            println("Progress: $prctg%")
        end
    end
    println("Best r found: $real_r")
    return real_r
end
# -------------------
# Defining parameters
# -------------------
N = 100
m = 20
d = 1
k = 50
a,b = -1, 1
r_interval = [i for i=0:0.0005:1]
MYPATH = "D:\\Personal\\EMAp\\Machine Learning\\zGitFolder\\Giovanni Amorim\\k-RBF-Network"
# --------------------
# Generating some data
# --------------------
data = MyML.myrand_float(N,d,a,b)
x = MyML.myrand_float(1,d,a,b)[1]
# --------------------------
# Saving values for our data
# --------------------------
S = Dict()
for n=1:N
    S[data[n]] = f(data[n])
end
neighbors = [k for k in keys(S)]
r = m_cross_validate_r(S,r_interval,m)
C = MyML.k_cluster(k,neighbors)
w = find_w(S,C,r)
# ------------------------------------
# Print data when d = 1 and regression
# ------------------------------------
if d == 1
    PyPlot.clf()
    title("Parametric RBF Regression")
    PyPlot.scatter([p for p in keys(S)], [S[p] for p in keys(S)], color="blue")
    PyPlot.scatter([y[1] for y in x], [h(x,w,r,C) for y in x], color="red")
    # PyPlot.scatter([y[1] for y in keys(S)], [h(y,w,r,C) for y in keys(S)], color="purple")
    PyPlot.xlim((a,b))
    if f([a]) == f([b])
        PyPlot.ylim((0,f([a])))
    else
        PyPlot.ylim((f([a]),f([b])))
    end
    PyPlot.savefig("$MYPATH\\k-RBF-network.pdf")
end
