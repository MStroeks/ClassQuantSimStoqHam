import numpy as np
from scipy.linalg import expm

'''
----------------------------------------------------------------------
'GtrotSim' takes as input the locality, system size, Trotter variable, 
interaction strength, magnetic field, and tau. It outputs an array co-
ntaining the Trotterized imaginary-time propagation operator.
----------------------------------------------------------------------
'''
def GtrotSim(k,N,M,Jx,Jy,Jz,g,tau):
    Paulis = np.zeros([2,2,4], dtype = np.complex128)
    Paulis[:,:,0] = np.array([[1,0],[0,1]])
    Paulis[:,:,1] = np.array([[0,1],[1,0]])
    Paulis[:,:,2] = np.array([[0,-1j],[1j,0]])
    Paulis[:,:,3] = np.array([[1,0],[0,-1]])
    seqint = np.zeros((k,3))
    Int = np.zeros([2**k,2**k,3])
    MagInt = np.zeros([2**k,2**k,3])
    seqmagx = np.zeros(1)
    Gsingle = np.zeros([2**k,2**k,1])
    Gtrot = np.zeros([2**k,2**k,M*N])

    for j in range(k):
        seqint[j,0] = 1
        seqint[j,1] = 3
        seqint[j,2] = 2
        if j == 0:
            seqmagx[j] = 1
    
    # Interactions with neighbouring sites
    for m in range(3):
        for i in range(k-1):
            if i == 0:
                Intper = np.kron(Paulis[:,:,int(seqint[i,m])],Paulis[:,:,int(seqint[i+1,m])])
            else:
                Intper = np.kron(Intper,Paulis[:,:,int(seqint[i+1,m])])
        Int[:,:,m] = Intper.real
    # Interactions with external (transverse) magnetic field
    for i in range(k-1):
        if i == 0:
            MagInt = np.kron(Paulis[:,:,int(seqmagx[i])],Paulis[:,:,0])
        else:
            MagInt = np.kron(MagInt,Paulis[:,:,0])
    
    subHi = Jx*Int[:,:,0] + Jz*Int[:,:,1] + Jy*Int[:,:,2] + g*MagInt
    # Rescaling subHi such that lambda_min = 0
    eigvaluesubHi = np.linalg.eig(subHi)[0]
    eigvectorsubHi = np.linalg.eig(subHi)[1]
    subHiRescaled = np.matmul(np.matmul(eigvectorsubHi,(np.diag(eigvaluesubHi) - np.amin(eigvaluesubHi)*np.eye(2**k))),np.linalg.inv(eigvectorsubHi))
    
    Gsingle = expm(-tau*subHiRescaled/M)
    Gtrot = np.repeat(Gsingle[:,:,np.newaxis], M*N, axis = 2)
    Gtrot[np.abs(Gtrot) < 10**(-10)] = 0
    
    return Gtrot

'''
----------------------------------------------------------------------
'DiagBlocksEig' takes as input the system size, Trotter variable, and 
Gtrot. It outputs the maximum eigenvalue and associated strictly posi-
tive eigenvector of all irreducible and element-wise non-negative blo-
cks along the diagonal of the local propagation operators.
----------------------------------------------------------------------
'''
def DiagBlocksEig(M,N,Gtrot):
    maxeigval = np.zeros((np.shape(Gtrot[:,:,0])[0],M*N))
    poseigvec = np.zeros((np.shape(Gtrot[:,:,0])[0],np.shape(Gtrot[:,:,0])[0],M*N))
    connectedstates = np.ones((np.size(Gtrot[:,:,0])+1,np.size(Gtrot[:,:,0])+1,M*N))*(-1)
    numsets = np.zeros(M*N)

    for t in range(M*N):
        connectedindices = np.c_[np.nonzero(Gtrot[:,:,t])[0],np.nonzero(Gtrot[:,:,t])[1]]
        numsetssing = 0
        csetcheck = np.array([[-1,-1],[-1,-1]])

        for m in range(np.shape(connectedindices)[0]):
            cset = np.reshape(connectedindices[m,:], (1,2))
            if ((np.reshape(cset.astype(int), (2,))).tolist() in csetcheck.tolist()) == False:
            
                n = 0
                while n < np.shape(cset)[0]:
                    startnode = cset[n,1]
                    n += 1
                    newnodes = np.where(connectedindices[:,0] == startnode)
                    for i in range(np.size(newnodes)):
                        cset = np.unique(np.vstack((cset, connectedindices[np.asarray(newnodes)[0,i],:])), axis = 0)
        
                csetcheck = np.concatenate((csetcheck,cset), axis = 0)
                connectedset = cset
            
                Gb = np.zeros((int((np.shape(connectedset)[0])**(1/2)),int((np.shape(connectedset)[0])**(1/2))))
                for i in range(int((np.shape(connectedset)[0])**(1/2))):
                    for j in range(int((np.shape(connectedset)[0])**(1/2))):
                        Gb[i,j] = ((Gtrot[:,:,t])[connectedset[:,0],connectedset[:,1]])[i*int((np.shape(connectedset)[0])**(1/2))+j].real
        
                maxeigval[m,t] = np.max(np.linalg.eig(Gb)[0])
                poseigvec[0:int((np.shape(connectedset)[0])**(1/2)),numsetssing,t] = abs((np.linalg.eig(Gb)[1])[:,np.argmax(np.linalg.eig(Gb)[0])])
            
                connectedstates[0:np.shape(connectedset)[0],(2*numsetssing):(2*numsetssing+2),t] = connectedset
                numsetssing = numsetssing + 1
        numsets[t] = numsetssing
    
    return maxeigval, poseigvec, connectedstates, numsets

'''
----------------------------------------------------------------------
'PiandFtau' takes as input the locality, system size, Trotter variable
, the set of connected states and the associated maximum eigenvalues 
and strictly positive eigenvectors of the blocks along the diagonal of
the local propagation operators, Gtrot, numsamples and the elements of
the state Phi. It implements an N-term first-order Trotterization sch-
eme and sets up the stochastic process to estimate F(tau). It outputs 
the estimate of F(tau). 
----------------------------------------------------------------------
'''
def PiandFTau(maxeigval,poseigvec,connectedstates,numsets,M,N,k,Gtrot,numsamples,phi_coefficients):
    ftau = 0
    for samples in range(numsamples):
        zeroth_state = (np.where(np.cumsum(phi_coefficients**2) > np.random.rand(1))[0])[0]
        x0 = np.reshape(np.flip(np.unpackbits(zeroth_state.astype(np.uint8),axis=0),axis=0)[0:N], (N,1))
        phi0 = phi_coefficients[zeroth_state]
        
        xMN = np.zeros((N,1))
        
        x = np.concatenate((np.concatenate((x0, np.zeros((N,M*N-1))), axis = 1), xMN), axis = 1)
        P = np.zeros(M*N)
        singleR = np.zeros(M*N)
        
        for t in range(M*N):
            if t == 0:
                xleft = np.roll(x[:,t], -t)[0:k]
            xleft_dec = int(np.array2string(xleft.astype(int)).replace('[','').replace(']','').replace(' ',''), 2)
                    
            for c in range(int(numsets[t])):
                if np.any(connectedstates[:,2*c,t] == xleft_dec) == True:
                    rightset = c
            lambdat = ((maxeigval[:,t])[maxeigval[:,t] != 0])[rightset]
            randnum = np.random.rand(1)
            
            if t < M*N:
                Pot_xright_dec = np.ones((2**k)**2)*(2**k + 1)
                Pot_xright_dec = connectedstates[np.argwhere(connectedstates[:,2*rightset,t] == xleft_dec),2*rightset+1,t]
                Pot_xright_dec = Pot_xright_dec[Pot_xright_dec != 2**k + 1]
                
                phix_t = np.zeros(np.size(Pot_xright_dec))
                phix_tmin = np.zeros(np.size(Pot_xright_dec))
                Prob = np.zeros(np.size(Pot_xright_dec))
                for i in range(np.size(Pot_xright_dec)):
                    eigindexright = np.argwhere(np.unique(connectedstates[:,2*rightset,t]) == Pot_xright_dec[i]) - 1
                    phix_t[i] = poseigvec[eigindexright,rightset,t]
                    eigindexleft = np.argwhere(np.unique(connectedstates[:,2*rightset,t]) == xleft_dec) - 1
                    phix_tmin[i] = poseigvec[eigindexleft,rightset,t]
                    Prob[i] = (Gtrot[int(xleft_dec),int(Pot_xright_dec[i]),t]/lambdat*phix_t[i]/phix_tmin[i]).real
                    if randnum < np.sum(Prob):
                        singleR[t] = lambdat*phix_tmin[i]/phix_t[i]
                        P[t] = Prob[i]
                        xright_dec = Pot_xright_dec[i]
                        xright = np.flip(np.flip(np.unpackbits(xright_dec.astype(np.uint8)),axis=0)[0:k],axis=0)
                        x[:,t + 1] = np.roll(np.concatenate((xright,np.roll(x[:,t], -(t))[k:N])), t)
                        xleft = (np.roll(x[:, t + 1], -(t + 1)))[0:k]
                        break
            
            if t == M*N-1:
                Lth_state_bin = x[:,t+1]
                Lth_state = int(np.array2string(Lth_state_bin.astype(int)).replace('[','').replace(']','').replace(' ',''), 2)
                phiL = phi_coefficients[Lth_state]
            
        Pi = np.prod(P[0:M*N])
        R = np.prod(singleR)
        estimatorforF = phiL/phi0*R
        ftau = ftau + estimatorforF/numsamples
        
    return ftau

'''
-----------------------------------------------------------------------
'PiandFtauCB' takes as input the locality, system size, Trotter varia-
ble, the set of connected states and the associated maximum eigenvalues 
and strictly positive eigenvectors of the blocks along the diagonal of
the local propagation operators, Gtrot, numsamples and the elements of
the state Phi. It implements a Gamma-term (checkerboard for k=2 and 
even system size) first-order Trotterization scheme and sets up the s-
tochastic process to estimate F(tau). It outputs the estimate of F(tau). 
-----------------------------------------------------------------------
'''
def PiandFTauCB(maxeigval,poseigvec,connectedstates,numsets,M,N,k,Gtrot,numsamples,phi_coefficients):
    ftau = 0
    for samples in range(numsamples):
        zeroth_state = (np.where(np.cumsum(phi_coefficients**2) > np.random.rand(1))[0])[0]
        x0 = np.reshape(np.flip(np.unpackbits(zeroth_state.astype(np.uint8),axis=0),axis=0)[0:N], (N,1))
        phi0 = phi_coefficients[zeroth_state]
        
        xM2 = np.zeros((N,1))
        
        x = np.concatenate((np.concatenate((x0, np.zeros((N,M*2-1))), axis = 1), xM2), axis = 1)
        P = np.zeros(M*N)
        singleR = np.zeros(M*N)
        
        for t in range(M*2):
            xleft_dec = np.zeros(np.int(N/2))
            rightset = np.zeros(np.int(N/2))
            lambdat = np.zeros(np.int(N/2))
            randnum = np.random.rand(np.int(N/2))
            xright_saved = np.zeros(N)
            
            if t == 0:
                xleft = x[:,0]
            for i in range(np.int(N/2)):
                if np.remainder(t,2) == 0:
                    xleft_dec[i] = int(np.array2string(xleft[2*i:2*i+2].astype(int)).replace('[','').replace(']','').replace(' ',''), 2)
                if np.remainder(t,2) == 1:
                    xleft_dec[i] = int(np.array2string((np.roll(xleft,-1)[2*i:2*i+2]).astype(int)).replace('[','').replace(']','').replace(' ',''), 2)
                    
                for c in range(int(numsets[np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)])):
                    if np.any(connectedstates[:,2*c,np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)] == xleft_dec[i]) == True:
                        rightset[i] = c
                lambdat[i] = ((maxeigval[:,np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)])[maxeigval[:,np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)] != 0])[np.int(rightset[i])]
            
            if t < M*2:
                for i in range(np.int(N/2)):
                    Pot_xright_dec = np.ones((2**2)**2)*(2**2 + 1)
                    Pot_xright_dec = connectedstates[np.argwhere(connectedstates[:,2*np.int(rightset[i]),np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)] == xleft_dec[i]),2*np.int(rightset[i])+1,np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)]
                    Pot_xright_dec = Pot_xright_dec[Pot_xright_dec != 2**2 + 1]
                    
                    phix_t = np.zeros(np.size(Pot_xright_dec))
                    phix_tmin = np.zeros(np.size(Pot_xright_dec))
                    Prob = np.zeros(np.size(Pot_xright_dec))
                    for p in range(np.size(Pot_xright_dec)):
                        eigindexright = np.argwhere(np.unique(connectedstates[:,2*np.int(rightset[i]),np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)]) == Pot_xright_dec[p]) - 1
                        phix_t[p] = poseigvec[eigindexright,np.int(rightset[i]),np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)]
                        eigindexleft = np.argwhere(np.unique(connectedstates[:,2*np.int(rightset[i]),np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)]) == xleft_dec[i]) - 1
                        phix_tmin[p] = poseigvec[eigindexleft,np.int(rightset[i]),np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)]
                        Prob[p] = (Gtrot[int(xleft_dec[i]),int(Pot_xright_dec[p]),np.int(np.round(t/2)*N/2 + np.remainder(t,2) + 2*i)]/lambdat[i]*phix_t[p]/phix_tmin[p]).real
                        if randnum[i] < np.sum(Prob):
                            singleR[np.int(t*N/2 + i)] = lambdat[i]*phix_tmin[p]/phix_t[p]
                            P[np.int(t*N/2 + i)] = Prob[p]
                            xright_dec = Pot_xright_dec[p]
                            xright = np.flip(np.flip(np.unpackbits(xright_dec.astype(np.uint8)),axis=0)[0:2],axis=0)
                            xright_saved[2*i:2*i+2] = xright
                            break
                    
                if np.remainder(t,2) == 1:
                    xright_saved = np.roll(xright_saved, 1)
                
                x[:,t + 1] = xright_saved
                xleft = x[:,t + 1]
                
            if t == M*2-1:
                Lth_state_bin = x[:,t + 1]
                Lth_state = int(np.array2string(Lth_state_bin.astype(int)).replace('[','').replace(']','').replace(' ',''), 2)
                phiL = phi_coefficients[Lth_state]
        
        Pi = np.prod(P[0:M*N])
        R = np.prod(singleR)
        estimatorforF = phiL/phi0*R
        ftau = ftau + estimatorforF/numsamples
        
    return ftau

'''
----------------------------------------------------------------------
'FullHandQPE' takes as input the locality, system size, Trotter varia-
ble, interaction strength, magnetic field, tau, elements of the state
Phi, numsamples and the choice of Trotterization schemes. It (!ineffi-
ciently!) implements the exact diagonalization of H, the exact calcul-
ation of the MC signal, and the QPE scheme.
----------------------------------------------------------------------
'''
def fullHandQPE(k,N,M,Jx,Jy,Jz,g,tau,phi_coefficients,numsamples,checkerboard):
    Paulis = np.zeros([2,2,4], dtype = np.complex128)
    Paulis[:,:,0] = np.array([[1,0],[0,1]])
    Paulis[:,:,1] = np.array([[0,1],[1,0]])
    Paulis[:,:,2] = np.array([[0,-1j],[1j,0]])
    Paulis[:,:,3] = np.array([[1,0],[0,-1]])
    Intcompl = np.zeros([2**N,2**N,3*N])
    MagIntcompl = np.zeros([2**N,2**N,N])
    seqmagx = np.zeros(N)
    seqint = np.zeros((N,3))
    H_i = np.zeros([2**N,2**N,N])
    HiRescaled = np.zeros([2**N,2**N,N])
    HRescaled = np.zeros([2**N,2**N])
    G = np.zeros([2**N,2**N])
    GtrotQPE = np.zeros((2**N,2**N,N),dtype=np.complex)
    GtrotQPEsingleoperator = np.zeros((2**N,2**N),dtype=np.complex)
    GnontrotQPE = np.zeros((2**N,2**N),dtype=np.complex)
    
    for j in range(k):
        seqint[j,0] = 1 
        seqint[j,1] = 3
        seqint[j,2] = 2
        if j == 0:
            seqmagx[j] = 1
    
    for i in range(N):
        for m in range(3):
            l = 2
            for p in range(N-1):
                if p == 0:
                    Intpercompl = np.kron(Paulis[:,:,int(seqint[np.remainder(i,N),m])],Paulis[:,:,int(seqint[np.remainder(i+1,N),m])])
                else:
                    Intpercompl = np.kron(Intpercompl,Paulis[:,:,int(seqint[np.remainder(i+l,N),m])])
                    l = l+1
            Intcompl[:,:,i + N*m] = Intpercompl.real 
            
        l = 2
        for p in range(N-1):
            if p == 0:
                Intmagpercompl = np.kron(Paulis[:,:,int(seqmagx[np.remainder(i,N)])],Paulis[:,:,int(seqmagx[np.remainder(i+1,N)])])
            else:
                Intmagpercompl = np.kron(Intmagpercompl,Paulis[:,:,int(seqmagx[np.remainder(i+l,N)])])
                l = l+1
        MagIntcompl[:,:,i] = Intmagpercompl.real
    
    for i in range(N):                                        
        H_i[:,:,i] = Jx*Intcompl[:,:,i] + Jz*Intcompl[:,:,N+i] + Jy*Intcompl[:,:,2*N+i] + g*MagIntcompl[:,:,i]
        # Rescaling each H_i such that lambda_min(H_i) = 0
        eigvalueHi = np.linalg.eig(H_i[:,:,i])[0]
        eigvectorHi = np.linalg.eig(H_i[:,:,i])[1]
        HiRescaled[:,:,i] = np.matmul(np.matmul(eigvectorHi,(np.diag(eigvalueHi) - np.amin(eigvalueHi)*np.eye(2**N))),np.linalg.inv(eigvectorHi))
        #HiRescaled[:,:,i] = H_i[:,:,i]
        GtrotQPE[:,:,i] = expm(-(1j)*tau*HiRescaled[:,:,i]/M)
        HRescaled = HRescaled + HiRescaled[:,:,i]
    FullHeigvals = np.linalg.eig(HRescaled)[0]
    FullHeigvecs = np.linalg.eig(HRescaled)[1]
    G = (expm(-tau*HRescaled)).real
    GnontrotQPE = expm(-(1j)*tau*HRescaled)
    realt_evolution_overlap_nontrot = np.matmul(np.matmul(np.transpose(phi_coefficients),GnontrotQPE),phi_coefficients)
    
    # Calculating Trotter Bounds
    # First-Order N-Term Decomposition
    commutatornorm = 0
    for i in range(N-1):
        for n in range(N-1-i):
            j = i + 1 + n
            commutatornorm = commutatornorm + np.linalg.norm(np.matmul(HiRescaled[:,:,i],HiRescaled[:,:,j]) - np.matmul(HiRescaled[:,:,j],HiRescaled[:,:,i]), ord = 2)
    trivialtrotterbound = commutatornorm*tau**2/(2*M)
    # First-Order Checkerboard Decomposition
    H_even = np.zeros((2**N,2**N))
    H_odd = np.zeros((2**N,2**N))
    for i in range(N):
        if np.remainder(i,2) == 0:
            H_even = H_even + HiRescaled[:,:,i]
        if np.remainder(i,2) == 1:
            H_odd = H_odd + HiRescaled[:,:,i]
    CBtrotterbound = np.linalg.norm(np.matmul(H_even,H_odd) - np.matmul(H_odd,H_even), ord = 2)*tau**2/(2*M)
    '''
    normcheck = 0
    for i in range(N):
        normcheck = normcheck + np.linalg.norm(HiRescaled[:,:,i], ord = 2)
    print(normcheck*tau/M)
    '''
    
    GtrotQPEsingleoperator = GtrotQPE[:,:,0]
    for i in range(N-1):
        if checkerboard == 0:
            GtrotQPEsingleoperator = np.matmul(GtrotQPEsingleoperator,GtrotQPE[:,:,i+1])
        if checkerboard == 1:
            if i < np.int(N/2)-1:
                GtrotQPEsingleoperator = np.matmul(GtrotQPEsingleoperator,GtrotQPE[:,:,2*(i+1)])
            if i >= np.int(N/2)-1:
                GtrotQPEsingleoperator = np.matmul(GtrotQPEsingleoperator,GtrotQPE[:,:,2*(i-np.int(N/2)+1)+1])
    GtrotQPEsingleoperator = np.linalg.matrix_power(GtrotQPEsingleoperator, M)
    realt_evolution_overlap = np.matmul(np.matmul(np.transpose(phi_coefficients),GtrotQPEsingleoperator),phi_coefficients)
    
    # sampling from real and imaginary parts of realt_evolution_overlap
    realpart_overlap = np.real(realt_evolution_overlap)
    imagpart_overlap = np.imag(realt_evolution_overlap)
    prob_ancilla_zero = 1/2 + 1/2*realpart_overlap
    prob_ancilla_pitwo = 1/2 - 1/2*imagpart_overlap
    
    rand_zero = np.random.rand(numsamples, 1)
    rand_pitwo = np.random.rand(numsamples, 1)
    
    prob_ancilla_zero_sampled = np.mean((rand_zero < prob_ancilla_zero).astype(int))
    prob_ancilla_pitwo_sampled = np.mean((rand_pitwo < prob_ancilla_pitwo).astype(int))
    
    sampled_realpart_overlap = 2*prob_ancilla_zero_sampled - 1
    sampled_imagpart_overlap = 1 - 2*prob_ancilla_pitwo_sampled
    
    return G, FullHeigvals, FullHeigvecs, sampled_realpart_overlap, sampled_imagpart_overlap, realt_evolution_overlap_nontrot, trivialtrotterbound, CBtrotterbound

'''
----------------------------------------------------------------------
'MatrixPencilMethod' takes as input a (noisy) signal, measpoints, the 
pencil parameter L and the truncation factor. It outputs the estimates
of the parameters z_{j}.
----------------------------------------------------------------------
'''
def MatrixPencilMethod(signal,measpoints,L,truncation_factor):
    
    Y = np.zeros((measpoints-L,L + 1), dtype = np.complex)
    for i in range(measpoints-L):
        Y[i,:] = signal[i:L + i + 1]
    
    # Without truncating the Y1 and Y2 SVD's (unstable when signal is noisy)
    '''
    Y1 = Y[:,0:L]
    Y2 = Y[:,1:L + 1]
    generalizedeigvals = np.linalg.eig(np.matmul(np.linalg.pinv(Y2),Y1))[0]
    generalizedeigvals = generalizedeigvals[generalizedeigvals > 10**(-10)]
    I = np.matmul(np.linalg.pinv(Y2),Y2)
    '''
    
    # Truncating the Y1 and Y2 SVD's (allows for noisy signal)
    Y1 = Y[:,0:L]
    Y2 = Y[:,1:L + 1]
    u1, sigma1, vh1 = np.linalg.svd(Y1) # SVD Y1
    u2, sigma2, vh2 = np.linalg.svd(Y2) # SVD Y2
    # Locating the set of largest singular values
    relevantargs1 = np.argwhere(sigma1/np.max(sigma1) > truncation_factor)
    relevantargs2 = np.argwhere(sigma2/np.max(sigma2) > truncation_factor)
    
    sigma1 = sigma1[sigma1/np.max(sigma1) > truncation_factor]
    sigma_matrix1 = np.diag(sigma1)
    sigma2 = sigma2[sigma2/np.max(sigma2) > truncation_factor]
    sigma_matrix2 = np.diag(sigma2)
    
    vh1prime = np.transpose(np.take(np.transpose(vh1),relevantargs1,axis = 1))
    vh2prime = np.transpose(np.take(np.transpose(vh2),relevantargs2,axis = 1))
    u1prime = np.take(u1,relevantargs1,axis = 1)
    u2prime = np.take(u2,relevantargs2,axis = 1)
    
    vh1prime = np.reshape(vh1prime, (np.shape(vh1prime)[1],np.shape(vh1prime)[2]))
    vh2prime = np.reshape(vh2prime, (np.shape(vh2prime)[1],np.shape(vh2prime)[2]))
    u1prime = np.reshape(u1prime, (np.shape(u1prime)[0],np.shape(u1prime)[1]))
    u2prime = np.reshape(u2prime, (np.shape(u2prime)[0],np.shape(u2prime)[1]))
    
    Y1_truncatedSVD = np.matmul(np.matmul(u1prime, sigma_matrix1),vh1prime)
    Y2_truncatedSVD = np.matmul(np.matmul(u2prime, sigma_matrix2),vh2prime)
    generalizedeigvals = np.linalg.eig(np.matmul(np.linalg.pinv(Y1_truncatedSVD),Y2_truncatedSVD))[0]
    I = np.matmul(np.linalg.pinv(Y1_truncatedSVD),Y1_truncatedSVD) # Check
    
    return generalizedeigvals, I
