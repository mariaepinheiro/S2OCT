using DelimitedFiles,StatsBase, CSV, Random, DataFrames,LinearAlgebra,TickTock,Distances
include("S2OCT.jl")
include("BertsimasOptimalDT.jl")

function check(x,w,γ,n=2)
    ac = 0
    if n == 2
        if dot(w[:,1],x)  - γ[1]<0
            if dot(w[:,2],x) - γ[2]< 0
                ac = 1
            else
                ac = -1
            end
        else
            if dot(w[:,3],x) - γ[3]< 0
                ac = 1
            else
                ac = -1
            end
        end
    end 
    if n == 3
        if dot(w[:,1],x)  - γ[1]<0
            if dot(w[:,2],x) - γ[2]< 0
                if dot(w[:,4],x) -γ[4]< 0
                    ac = 1
                else
                    ac = -1
                end
            else 
                if dot(w[:,5],x) -γ[5]< 0
                    ac = 1
                else
                    ac = -1
                end
            end
        else
            if dot(w[:,3],x) - γ[3]< 0
                if dot(w[:,6],x) -γ[6]< 0
                    ac = 1
                else
                    ac = -1
                end
            else
                if dot(w[:,7],x) -γ[7]< 0
                    ac = 1
                else
                    ac = -1
                end   
            end
        end

    end 
    return ac
end

function comparing(P1,Y1,s,g,MCnum,M1,D,R,bd,p0,TY==0)
    rng = MersenneTwister(70)
    S = Vector{Vector{Int32}}(undef,MCnum) 
    if TY==1 # sample under simple sample
        for i = 1:MCnum
           S[i] = sample(rng,1:s , g; replace=false) 
        end
    else # sample under Biased sample
        SamplingProb = fill(0.15, s)
        SamplingProb[Y1 .== 1] .= 0.85 # sample under Biased sample
        for i = 1:MCnum
            S[i] = StatsBase.wsample(rng, 1:s, SamplingProb, g; replace=false)
        end
    end
    TP1 = zeros(MCnum)
    TP2 = zeros(MCnum)
    FP1 = zeros(MCnum)
    FP2 = zeros(MCnum)
    TN1 = zeros(MCnum)
    TN2 = zeros(MCnum)
    FN1 = zeros(MCnum)
    FN2 = zeros(MCnum)
    TP1u = zeros(MCnum)
    TP2u = zeros(MCnum)
    FP1u = zeros(MCnum)
    FP2u = zeros(MCnum)
    TN1u = zeros(MCnum)
    TN2u = zeros(MCnum)
    FN1u = zeros(MCnum)
    FN2u = zeros(MCnum)
    ti1 = zeros(MCnum)
    ti2 = zeros(MCnum)
    for i = 1 : MCnum
        p=p0
        X = P1[S[i],:]
        y = Y1[S[i]]
        k = sort(S[i])
        H= deleteat!(collect(1:s),k)
        yA = findall(==(1),y)
        yB = findall(==(-1),y)
        AX = X[yA,:]
        BX = X[yB,:]
        y = [y[yA]; y[yB]]
        X1 = P1[H,:]
        y1 = Y1[H]
        τA = 0
        for L = 1 : s-g
            if y1[L] == 1
                τA+=1
            end
        end
        ma = size(AX,1)
        XAB = [AX; BX]
        XAB0 = rescalingOCH(XAB)
        X10 = rescalingOCH(X1)
        t2 = @elapsed  H2 =  OCTH(XAB0,ma,p0,0,0.05*size(XAB0,1),7200)
        ti1[i] = t2
        for j= 1:g
            ac = argmax(H2[4][:,j])
            ac = argmax(H2[3][:,ac])
            if ac == 1
                if y[j] == 1
                    TP1[i] +=1
                else
                    FP1[i] +=1
                end
            else
                if y[j] == -1
                   TN1[i] +=1
                else
                    FN1[i] +=1
                end
            end                  
        end
        for j= 1:s-g
            ac = predictioOCTH(X10[j,:],H2[1],H2[2],H2[3],p0)
            if ac >= 0
                if y1[j] == 1
                    TP1u[i] +=1
                else
                    FP1u[i] +=1
                end
            else
                if y1[j] == -1
                    TN1u[i] +=1
                else
                    FN1u[i] +=1
                end
            end
        end
        ########S²OCT 
        t2 = @elapsed  H2 = S2OCT(XAB,X1,ma,τA,p0,1,bd*sqrt(M1)*R+1,7200,bd)
        ti2[i] = t2
        for j = 1:ma
           ac = check(XAB[j,:],H2[1],H2[2],p0)
           if ac >= 0
                TP2[i] +=1
            else
                FN2[i] +=1
            end
        end 
        for j = ma+1:g
            ac = check(XAB[j,:],H2[1],H2[2],p0)
            if ac ≤ 0
                 TN2[i] +=1
             else
                 FP2[i] +=1
             end
         end 
        for j= 1:s-g
            ac =  H2[end][j]
             if ac >= 0
                 if y1[j] == 1
                     TP2u[i] +=1
                 else
                    FP2u[i] +=1
               end
            else
                if y1[j] == -1
                    TN2u[i] +=1
                else
                    FN2u[i] +=1
                end
             end
         end


        println("comparing $i finished")
    end
    TP = [TP1 TP2 TP1u TP2u]
    CSV.write("results/$T/TruePositive$D.csv",DataFrame(TP,:auto))

    TN = [TN1 TN2 TN1u TN2u]
    CSV.write("results/$T/TrueNegative$D.csv",DataFrame(TN,:auto))

    FP = [FP1 FP2 FP1u FP2u]
    CSV.write("results/$T/FalsePositive$D.csv",DataFrame(FP,:auto))
    
    FN = [FN1 FN2 FN1u FN2u]
    CSV.write("results/$T/FalseNegative$D.csv",DataFrame(FN,:auto))
    t = [ti1 ti2]
    CSV.write("results/$T/TIMEs$D.csv",DataFrame(t,:auto))
    

end





function rescaling(X) #function for rescaling and get the elements of the matrix between -100 and 100
    m,n = size(X)
    for i = 1 :n
        b = maximum(X[:,i])
        l = minimum(X[:,i])
        u = (l+b)/2
        X[:,i] =  X[:,i] - u*ones(m)
        b1 = b - u
        l1 = l - u
        w= b1-l1
        zb = 1e2
        zl = -1e2
        if b1 > zb || l1 < zl
            for j = 1 : m
                X[j,i] = (zb-zl)*(X[j,i] - l1)/(b1-l1) + zl
            end
        end
    end
    return X
end

function preprocessingdata(A)
    D =  CSV.read("datasetsfolder/$A.tsv",DataFrame)
    D = unique(D)
    s, p = size(D)
    D = Matrix(D)
    D = convert(Matrix{Float64},D)
    if A == 28
        Y = D[:,end]
        D = D[:,1:end-1]
        s, p = size(D)
        for i = 1 : s
            if Y[i] ==0
                Y[i] = 1
            else 
                Y[i] = -1
            end
        end 
    elseif A== 54 || A == 80
        Y = D[:,1]
        D = D[:,2:end]
        s, p = size(D)
        for i = 1 : s
            if Y[i] !=1
                Y[i] = -1
            end
        end
    else 
        Y = D[:,end]
        D = D[:,1:end-1]
        s, p = size(D)
        for i = 1 : s
            if Y[i] !=1
                Y[i] = -1
            end
        end
    end
    D= rescaling(D)
    g = Int64(round(0.1*s))
    R = maximum(pairwise(euclidean, D, dims=1))
    return D,Y, p,s,g,R
end


function running(i,MCnum)
    P1,Y1,M1,s,g,R = preprocessingdata(i)
    bd = max(10,499/(sqrt(M1)*R))
    p0 = 2
    if s >= 1000
        p0 = 3
    end
    if s ≥ 650
        bd = max(20,499/(sqrt(M1)*R))
    end
    if s ≥ 1500
        bd = max(40,499/(sqrt(M1)*R))
    end
    D = i
    comparing(P1,Y1,s,g,MCnum,M1,D,R,bd,p0)
end
