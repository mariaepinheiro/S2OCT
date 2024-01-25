

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

