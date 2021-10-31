'''
------------------------------------------------------------------------------------------------
This code tracks the (Trotterized) evolution of the state of a (1D) spin system of size n in im-
aginary (Euclidian) and real time. The real time evolution is inefficiently tracked in general. 
The imganary evolution is efficiently tracked given the Hamiltonian is piece-wise stoquastic wi-
th interactions of (general) locality k = log(poly(n)). The interactions need not be translatio-
nally invariant. Using the Matrix Pencil Method, the eigenvalues of the Hamiltonian are estimat-
ed from the state evolutions.

This main code file (StoqMC.py) calls functions from the separate code files (StoqMCfunctions.p-
y) and (StoqMCsimulation.py). The files StoqMCfunctions.py and StoqMCsimulation.py contain desc-
riptions of each function that is called in this main code file.

We note that the Hamiltonian of the spin system used here is of the form (where periodic bounda-
ry conditions are imposed):
    H = sum_i (J_x*X_i*...*X_i+k + J_y*Y_i*...*Y_i+k + J_z*Z_i*...*Z_i+k 
                                                           + g*X_i + g_y*Y_i + g_z*Z_i).
We note it is the users own responsibility to make sure that the Hamiltonian is expressed in a 
basis in which it is piece-wise stoquastic (if possible). 
-------------------------------------------------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
import StoqMCfunctions as f
import StoqMCsimulations as s

# Initialization
N = 7                           # Length of Spin Chain
M = 60                          # Number of Trotter Steps
Jx = 0                          # X Spin-Spin Interaction Strength
Jy = 0                          # Y Spin-Spin Interaction Strength
Jz = -1                         # Z Spin-Spin Interaction Strength
g = -4                          # (External) Transverse Magnetic Field (in x direction)
gy = 0                          # (External) Magnetic Field (in y direction)
gz = 0                          # (External) Magnetic Field (in z direction)
k = 2                           # Locality of the Spin-Spin Interactions
numsamples = 4000               # Sample Size at each instance in time (|Sigma|)
measpoints = 101                # Number of Measurement Points in time (K)
tau_increment = 0.025           # Time Increment (Measured Time Interval = K*Time Increment)
checkerboard = 0                # 0 (Trivial Trotter Decomposition), 1 (Checkerboard Decomposition: when in use, it is assumed that k = 2 and n is even)
L = np.int(measpoints/2)        # Pencil Parameter
truncation_factor = 2*10**(-2)  # Truncation Factor

gfullmeas = np.zeros(measpoints)                # Initialization of Noisless and Trotter-Errorless Decaying Signal
ftaumeas = np.zeros(measpoints)                 # Initialization of Noise- and Trotter-Error Infected Decaying Signal (MC Signal)
QPEnontrotsignalreal = np.zeros(measpoints)     # Initialization of Real Part of Noisless and Trotter-Errorless Oscillating Signal
QPEnontrotsignalimag = np.zeros(measpoints)     # Initialization of Imag Part of Noisless and Trotter-Errorless Oscillating Signal
QPEsampled_realsignal = np.zeros(measpoints)    # Initialization of Real Part of Noise- and Trotter-Error Infected Oscillating Signal (Real Part QPE Signal)
QPEsampled_imagsignal = np.zeros(measpoints)    # Initialization of Imag Part of Noise- and Trotter-Error Infected Oscillating Signal (Imag Part QPE Signal)

# Initialization of the state |Phi>
# |+>:
#phi_coefficients = np.ones(2**N)/(2**(N/2)) # |\Phi> = |+>
# |Phi_{opt,p=0,1}>:
alpha = 1/((N+1)**(1/2))
beta = (N/(N+1))**(1/2)
phi_coefficients = np.zeros(2**N)
phi_labels = np.arange(2**N)
phi_binary = np.flip(np.unpackbits(np.reshape(phi_labels, (1,2**N)).astype(np.uint8), axis = 0), axis = 0)[0:N,:]
for i in range(N):
    phi_coefficients = phi_coefficients + (phi_binary[i,:] == 0).astype(int)*(alpha + beta)/((2)**(1/2))/((2**(N-1))**(1/2))
    phi_coefficients = phi_coefficients + (phi_binary[i,:] == 1).astype(int)*(alpha - beta)/((2)**(1/2))/((2**(N-1))**(1/2))
phi_coefficients = phi_coefficients/((N*(alpha**2 + beta**2) + N*(N-1)*alpha**2)**(1/2))

eigvaldata = np.zeros((4,100,1), dtype = np.complex)            # Initialization of Storage Matrix for Eigenvalue Estimates
gfulldata = np.zeros((1,measpoints))                            # Initialization of Storage Matrix for Idealized Decaying Signal
ftaudata = np.zeros((1,measpoints))                             # Initialization of Storage Matrix for MC Signal
qpedata = np.zeros((1,measpoints), dtype = np.complex)          # Initialization of Storage Matrix for Idealized Oscillating Signal
qpenoisydata = np.zeros((1,measpoints), dtype = np.complex)     # Initialization of Storage Matrix for QPE Signal

for m in range(1):
    #numsamples = np.int(4546*1.25*1.25**x[m])
    
    for Q in range(measpoints):
        tau = tau_increment*(Q) # 'Time (Imaginary and/or Real)'
        
        # Calculate the Matrix Element Directly and Inefficiently Implement the QPE Procedure (!!!USE ONLY FOR SMALL N!!!)
        Gfull, FullHeigvals, FullHeigvecs, QPEsampled_real, QPEsampled_imag, realt_nontrot_evolution, trivialtrotterbound, CBtrotterbound = s.fullHandQPE(k,N,M,Jx,Jy,Jz,g,tau,phi_coefficients,numsamples,checkerboard)
        gfullmeas[Q] = np.matmul(np.matmul(np.transpose(phi_coefficients),Gfull),phi_coefficients)
        QPEsampled_realsignal[Q] = QPEsampled_real
        QPEsampled_imagsignal[Q] = QPEsampled_imag
        QPEnontrotsignalreal[Q] = np.real(realt_nontrot_evolution)
        QPEnontrotsignalimag[Q] = np.imag(realt_nontrot_evolution)
        #Ntermtrotterbound[Q] = trivialtrotterbound
        #Twotermtrotterbound[Q] = CBtrotterbound
        
        # Simulate String of Local Imaginary Time Propagation Operators
        Gtrot = s.GtrotSim(k,N,M,Jx,Jy,Jz,g,tau)
        
        # Simulate Connected Sets of States, and Eigenstates & Eigenvalues of the Block Diagonal Matrices
        maxeigval, poseigvec, connectedstates, numsets = s.DiagBlocksEig(M,N,Gtrot)
        
        # Obtain |Sigma| Samples of High-Dimensional Probability Distribution (Pi) and Estimate the MC Signal at tau (F(tau))
        if checkerboard == 0:
            ftau = s.PiandFTau(maxeigval,poseigvec,connectedstates,numsets,M,N,k,Gtrot,numsamples,phi_coefficients)
        if checkerboard == 1:
            ftau = s.PiandFTauCB(maxeigval,poseigvec,connectedstates,numsets,M,N,k,Gtrot,numsamples,phi_coefficients)
        ftaumeas[Q] = ftau
        
        print(m,'step ',Q) # To indicate the progress of the simulation
        
    # Construct Complete QPE Signals (with and without Noise and Trotter Error) using their Real and Imaginary Parts
    QPEsampled_signal = QPEsampled_realsignal + 1j*QPEsampled_imagsignal
    QPEnontrotsignal = QPEnontrotsignalreal + 1j*QPEnontrotsignalimag
    
    # Store State Evolution Data in Storage Matrices
    ftaudata[m,:] = ftaumeas
    gfulldata[m,:] = gfullmeas
    qpenoisydata[m,:] = QPEsampled_signal
    qpedata[m,:] = QPEnontrotsignal
    
    # Implement Matrix Pencil Method to obtain Eigenvalue Estimates from State Evolution Data
    generalizedeigvals, I = s.MatrixPencilMethod(ftaudata[m,:],measpoints,L,truncation_factor)
    generalizedeigvals = generalizedeigvals[generalizedeigvals >= 10**(-10)]
    eigval_estimates = np.sort(-np.log(generalizedeigvals.real)/tau_increment)
    eigvaldata[0,0:np.size(eigval_estimates),m] = eigval_estimates
   
    generalizedeigvals, I = s.MatrixPencilMethod(gfulldata[m,:],measpoints,L,truncation_factor)
    generalizedeigvals = generalizedeigvals[generalizedeigvals >= 10**(-10)]
    eigval_estimates = np.sort(-np.log(generalizedeigvals.real)/tau_increment)
    eigvaldata[1,0:np.size(eigval_estimates),m] = eigval_estimates
    
    generalizedeigvals, I = s.MatrixPencilMethod(qpenoisydata[m,:],measpoints,L,truncation_factor)
    generalizedeigvals = generalizedeigvals[np.abs(generalizedeigvals) >= 10**(-10)]
    eigval_estimates = np.sort(1j*np.log(generalizedeigvals)/tau_increment)
    eigvaldata[2,0:np.size(eigval_estimates),m] = eigval_estimates
    
    generalizedeigvals, I = s.MatrixPencilMethod(qpedata[m,:],measpoints,L,truncation_factor)
    generalizedeigvals = generalizedeigvals[np.abs(generalizedeigvals) >= 10**(-10)]
    eigval_estimates = np.sort(1j*np.log(generalizedeigvals)/tau_increment)
    eigvaldata[3,0:np.size(eigval_estimates),m] = eigval_estimates
    
# Plot the Imaginary and Real Time State Evolution 
#f.propagation_plot(measpoints, tau_increment, numsamples, gfulldata, ftaudata, qpedata, qpenoisydata, M)

# Save State Evolution Data and Eigenvalue Estimate Data
'''
np.save('eigvaldata_g',eigvaldata)
np.save('ftaudata',ftaudata)
np.save('gfulldata',gfulldata)
np.save('qpenoisydata',qpenoisydata)
np.save('qpedata',qpedata)
'''
plt.plot(np.arange(100)/20-5,1/np.tanh(np.arange(100)/20-5),'k')
plt.plot(5-np.arange(100)/20,1/np.tanh(5-np.arange(100)/20),'k')
plt.ylabel('coth($x$)', fontsize = 14)
plt.xlabel('$x$', fontsize = 14)

