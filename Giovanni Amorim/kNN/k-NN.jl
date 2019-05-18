include("..\\MyML\\MyML.jl")
using PyPlot, .MyML
# ---------------------------
# Decide what function to use
# ---------------------------
function f(x)
    return sum([v^2 for v in x])
end
# ------------------------
# Functions that will help
# ------------------------
function relative_diference(a,b)
    return abs((a-b)/b - 1)
end

function k_closest_neighbors(neighbors,point,k)
    ns = [[MyML.distance(neighbors[j],point), neighbors[j]] for j=1:length(neighbors) if neighbors[j] != point]
    if length(ns) == 0
        return "Empty neighborhood!"
    end
    sort!(ns)
    js = [ns[i][2] for i=1:k]
    return js
end

function closest_neighbor_correted(p,S,U,k,threshold)
    neighbors = [k for k in keys(S)]
    for n in k_closest_neighbors(neighbors,p,length(neighbors)-1)
        println(n, "   ", relative_diference(g(n,U,k),g(n,S,k)))
        if relative_diference(g(n,U,k),g(n,S,k)) <= threshold
            return n
        end
    end
    return Nothing
end

function g(x,S,k)
    neighbors = [p for p in keys(S)]
    gsum = sum([S[i] for i in k_closest_neighbors(neighbors,x,k)])
    return gsum/k
end
# ---------------
# Condense S to U
# ---------------
function condensate(S,k,threshold)
    U = Dict()
    neighbors = [k for k in keys(S)]
    for i=1:k
        p = neighbors[i]
        U[p] = S[p]
    end
    t = true
    while t
        nt = 0
        for p in [k for k in keys(S) if get(U,k,false) == false]
            # Teste se o ponto está mal "classificado"
            if relative_diference(g(p,U,k), g(p,S,k)) > threshold
                # Buscar o vizinho bem "classificado" mais próximo
                q = closest_neighbor_correted(p,S,U,k,threshold)
                if q != Nothing
                    U[q] = S[q]
                    nt += 1
                    #break
                end
            end
        end
        println(nt,t)
        if nt == 0
            t = false
        end
    end
    return U
end
# --------------------------------------------
# Defining parameters
# --------------------------------------------
N = 200
d = 1
a,b = -5,5
prev = 5
k = 4
want_to_condensate = false
MYPATH = "D:\\Personal\\EMAp\\Machine Learning\\zGitFolder\\Giovanni Amorim\\kNN"
# To classification functions only
L = 10
# --------------------------------
# Generating some data
# --------------------------------
x = MyML.myrand_float(prev,d,a,b)
data = MyML.myrand_float(N,d,a,b)
# --------------------------
# Saving values for our data
# --------------------------
S = Dict()
for n=1:N
    S[data[n]] = f(data[n])
end
if want_to_condensate == true
    U = condensate(S,k)
else
    U = S
end
# ------------------------------------
# Print data when d = 1 and regression
# ------------------------------------
if d == 1
    PyPlot.clf()
    title("1d-k-NN-Regression")
    PyPlot.scatter([y[1] for y in x], [g(y,U,k) for y in x], color="red")
    PyPlot.scatter([p for p in keys(U)], [U[p] for p in keys(U)], color="blue")
    l1 = -1:0.1:1
    l2 = f.(l1)
    PyPlot.plot(l1,l2)
    PyPlot.xlim((a,b))
    if f([a]) == f([b])
        PyPlot.ylim((-1,f([a])))
    else
        PyPlot.ylim((f([a]),f([b])))
    end
    PyPlot.savefig("$MYPATH\\1d-k-NN-Regression.pdf")
end
# ------------------------------------------------------------------------------------
# Print data divided (only makes sense with d = 2 and f with breakpoint = L)
# ------------------------------------------------------------------------------------
if d == 2
    title("2d-k-NN-Classification-by-Regression")
    PyPlot.clf()
    for y in x
        if g(y,U,k) >= L
            color = "purple"
        else
            color = "pink"
        end
        PyPlot.scatter([y[1]],[y[2]],color=color,s = 50)
    end
    neighbors = [p for p in keys(U)]
    PyPlot.scatter([x[1] for x in neighbors if U[x] >= L],[x[2] for x in neighbors if U[x] >= L],color="blue", label = "Positivo", s = 40)
    PyPlot.scatter([x[1] for x in neighbors if U[x] < L],[x[2] for x in neighbors if U[x] < L],color="red", label = "Negativo", s = 40)
    PyPlot.xlim((a,b))
    PyPlot.ylim((a,b))
    PyPlot.savefig("$MYPATH\\2d-k-NN-Regression.pdf")
end
