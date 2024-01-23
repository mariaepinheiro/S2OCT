
using Gurobi, JuMP,LinearAlgebra
function OCTH(X,pos,D,α,Nmin, maxtime) ###x∈[0,1]
    n,p = size(X)
    TB = 2^D-1
    TL = 2^D
    μ = 0.005
    Y = ones(n,2)
    Y[1:pos,2] = -ones(pos)
    Y[pos+1:n,1] = -ones(n-(pos))
    Y = Y.+1
    hat_L = max(pos, n-pos)
    model = Model(optimizer_with_attributes(Gurobi.Optimizer))
    set_time_limit_sec(model,maxtime)
    @variable(model, a[1:TB,1:p])
    @variable(model, hat_a[1:TB,1:p])
    @variable(model, d[1:TB],Bin)
    @variable(model, b[1:TB])
    @variable(model, z[1:TL,1:n],Bin)
    @variable(model, ℓ[1:TL],Bin)
    @variable(model, s[1:TB,1:p],Bin)
    @variable(model, N_k[1:2,1:TL])
    @variable(model, N[1:TL])
    @variable(model, c[1:2,1:TL],Bin)
    @variable(model, L[1:TL]≥0)
    if D == 2
        AL = [1,2],[1],[3],zeros(0)
        AR = zeros(0), [2],[1],[1,3]
    elseif D==3
        AL = [1,2,4], [1,2],[1,5],[1],[3,6],[3],[7],zeros(0)
        AR = zeros(0),[4],  [2], [2,5],[1],[1,6],[1,3],[1,3,7]
    elseif D==4
        AL = [1,2,4,8], [1,2,4],[1,2,9],[1,2],[1,5,10],[1,5],[1,11], [1], [3,6,12], [3,6],[3,13],[3], [7,14], [7],[15], zeros(0)
        AR = zeros(0),  [8],  [4], [4,9],[2],[2,10], [2,5], [2,5,11], [1], [1,12], [1,6], [1,6,13],[1,3],[1,3,14],[1,3,7], [1,3,7,15]
    end
    @constraint(model, [t = 1 : TL, k =1:2], L[t] ≥ N[t]-N_k[k,t] - n*(1-c[k,t]))
    @constraint(model, [t = 1 : TL, k =1:2], L[t] ≤  N[t]-N_k[k,t] + n*c[k,t])
    @constraint(model, [t = 1 : TL, k =1:2], N_k[k,t] == 0.5*(sum(Y[:,k].*z[t,:])))
    @constraint(model, [t = 1 : TL], N[t] == sum(z[t,:]))
    @constraint(model, [t = 1 : TL], ℓ[t] == sum(c[:,t]))
    @constraint(model, [t = 1: TL, m ∈ AL[t], i = 1 : n], dot(a[m,:],X[i,:]) + μ ≤ b[m] + (2+μ)*(1-z[t,i]))
    @constraint(model, [t = 1: TL, m ∈ AR[t], i = 1 : n], dot(a[m,:],X[i,:]) ≥ b[m] -2*(1-z[t,i]))
    @constraint(model, [i=1:n], sum(z[:,i]) == 1)
    @constraint(model, [i=1:n, t = 1:TL], z[t,i]≤ ℓ[t])
    @constraint(model, [t = 1 : TL], sum(z[t,:]) ≥ Nmin*ℓ[t])
    @constraint(model, [t = 1 : TB], sum(hat_a[t,:]) ≤ d[t])
    @constraint(model, [t = 1 : TB, j = 1 : p], hat_a[t,j] ≥ a[t,j])
    @constraint(model, [t = 1 : TB, j = 1 : p], hat_a[t,j] ≥ -a[t,j])
    @constraint(model, [t = 1 : TB, j = 1 : p], a[t,j] ≥ -s[t,j])
    @constraint(model, [t = 1 : TB, j = 1 : p], a[t,j] ≤ s[t,j])
    @constraint(model, [t = 1 : TB, j = 1 : p], s[t,j] ≤ d[t])
    @constraint(model, [t = 1 : TB], b[t] ≤ d[t])
    @constraint(model, [t = 1 : TB], b[t] ≥ -d[t])
    @constraint(model, [t = 2 : TB], d[t] ≤ d[div(t,2)])
    @objective(model, Min, (1/hat_L)*sum(L) + α*sum(s))
    print(model)
    optimize!(model)
    a,b,c,z= value.(a), value.(b), value.(c),value.(z)
    return a,b,c,z
end

function predictioOCTH(x,a,b,c,D)
    m,p = size(a)
    branch= zeros(m)
    for i = 1 : m 
        if dot(a[i,:],x) + b[i] ≥ 0 
            branch[i] = 1
        end
    end
    if D == 2
        if branch[1] ==0
            if branch[2] == 0
                class = argmax(c[:,1])
            else
                class = argmax(c[:,2])
            end 
        else
            if branch[3] == 0
                class = argmax(c[:,3])
            else
                class = argmax(c[:,4])
            end 
        end
    elseif D == 3
        if branch[1] ==0
            if branch[2] == 0
                if branch[4] == 0
                    class = argmax(c[:,1])
                else
                    class = argmax(c[:,2])
                end 
            else 
                if branch[5] == 0
                    class = argmax(c[:,3])
                else
                    class = argmax(c[:,4])
                end 
            end 
        else
            if branch[3] == 0
                if branch[6] == 0
                    class = argmax(c[:,5])
                else
                    class = argmax(c[:,6])
                end 
            else 
                if branch[7] == 0
                    class = argmax(c[:,7])
                else
                    class = argmax(c[:,8])
                end 
            end 
        end

    elseif D == 4
        if branch[1] ==0
            if branch[2] == 0
                if branch[4] == 0
                    if branch[8] == 0
                        class = argmax(c[:,1])
                    else 
                        class = argmax(c[:,2])
                    end
                else
                    if branch[9] == 0
                        class = argmax(c[:,3])
                    else 
                        class = argmax(c[:,4])
                    end
                end 
            else 
                if branch[5] == 0
                    if branch[10] == 0
                        class = argmax(c[:,5])
                    else
                        class = argmax(c[:,6])
                    end 
                else 
                    if branch[11] == 0
                        class = argmax(c[:,7])
                    else
                        class = argmax(c[:,8])
                    end 
                end 
            end 
        else
            if branch[3] == 0
                if branch[6] == 0
                    if branch[12] == 0
                        class = argmax(c[:,9])
                    else
                        class = argmax(c[:,10])
                    end 
                else
                    if branch[13] == 0
                        class = argmax(c[:,11])
                    else
                        class = argmax(c[:,12])
                    end 
                end 
            else 
                if branch[7] == 0
                    if branch[14] == 0
                        class = argmax(c[:,13])
                    else
                        class = argmax(c[:,14])
                    end 
                else
                    if branch[15] == 0
                        class = argmax(c[:,15])
                    else
                        class = argmax(c[:,16])
                    end 
                end 
            end 
        end
    end
    class = -2*class + 3
    return class

end

function rescalingOCH(X) #function for rescaling and get the elements of the matrix between -0 and 1
    m,n = size(X)
    for i = 1 :n
        b = maximum(X[:,i])
        l = minimum(X[:,i])
        u = (l+b)/2
        X[:,i] =  X[:,i] - u*ones(m)
        b1 = b - u
        l1 = l - u
        w= b1-l1
        zb = 1
        zl = 0
        if b1 > zb || l1 < zl
            for j = 1 : m
                X[j,i] = (zb-zl)*(X[j,i] - l1)/(b1-l1) + zl
            end
        end
    end
    return X
end

