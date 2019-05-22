include("..\\MyML\\MyML.jl")
using .MyML
# my start data is a point in [-2,2]x[-2,2] with a bias factor
N = 10
data = MyML.myrand_float(N,2,-1.414,1.414)
Ndata = [vcat([1],i) for i in data]
# function that validates array values binarly
function checks_1d(arr,l,tp=1)
    N = length(arr)
    results = ones(N)*-1
    for i=1:N
        if tp == 1
            if arr[i] > l
                results[i] = 1
            end
        else
            if arr[i] < l
                results[i] = 1
            end
        end
    end
    return results
end
function three_layer_four_perceptron(X)
    # All weights vector and all layers results
    W, H = [], []
    # each of my first layer neuron 'is' one line of my square (perceptron)
    w1 = [1,1,0]  # x > -1
    w2 = [1,0,1]  # y > -1
    w3 = [1,-1,0] # x < 1
    w4 = [1,0,-1] # y < 1
    # W is my first hidden layer
    push!(W,[w1'; w2'; w3'; w4'])
    push!(H,vcat([1],checks_1d(W[length(W)]*X,0)))
    # The second hidden layer has 4 neurons
    # each one represents a statement 'point is out positioned'
    w1 = [-1.5,-1,0,1,0] # !h1 and h3
    w2 = [-1.5,0,-1,0,1] # !h2 and h4
    w3 = [-1.5,1,0,-1,0] # h1 and !h3
    w4 = [-1.5,0,1,0,-1] # h2 and !h4
    push!(W,[w1';w2';w3';w4'])
    push!(H,vcat([1],checks_1d(W[length(W)]*H[length(H)],0)))
    # Finally, the last layer will output whether or not
    # our point unsatisfies all last neurons, so it's inside
    # the square
    w1 = [-3.5,-1,-1,-1,-1]
    push!(W,w1')
    return checks_1d(W[length(W)]*H[length(H)],0)[1]
end

for X in Ndata
    x,y = round.(X[2:3],digits=2)
    z = three_layer_four_perceptron(X)
    if z > 0
        println("The point ($x,$y) is inside the square :)")
    else
        println("The point ($x,$y) is outside the square :(")
    end
end
