'''
eigvals = np.array([0.41, 0.79, 1.34, 3.69, 4.07, 4.89])
eigvaldata = np.zeros((2,100,30))
time_interval = 3
truncation_factor = 0

for j in range(30):
    K = 2 + j
    L = np.int(K/2)
    time = np.arange(K)/(K-1)*time_interval
    time_increment = time[1] - time[0]
    print(eigvals*time_increment)
    decaysignal = np.zeros(K)
    oscillsignal = np.zeros(K, dtype = np.complex)
    for i in range(np.size(eigvals)):
        decaysignal = decaysignal + 1/np.size(eigvals)*np.exp(-time*eigvals[i])
        oscillsignal = oscillsignal + 1/np.size(eigvals)*np.exp(-1j*time*eigvals[i])
        
    generalizedeigvals, I = s.MatrixPencilMethod(decaysignal,K,L,truncation_factor)
    generalizedeigvals = generalizedeigvals[generalizedeigvals >= 10**(-8)]
    eigval_estimates = np.sort(-np.log(generalizedeigvals.real)/time_increment)
    eigvaldata[0,0:np.size(eigval_estimates),j] = eigval_estimates
    
    generalizedeigvals, I = s.MatrixPencilMethod(oscillsignal,K,L,truncation_factor)
    generalizedeigvals = generalizedeigvals[np.abs(generalizedeigvals) >= 10**(-8)]
    eigval_estimates = np.sort(1j*np.log(generalizedeigvals)/time_increment)
    eigvaldata[1,0:np.size(eigval_estimates),j] = eigval_estimates

eigvaldata[eigvaldata == 0] = -10

fig, axs = plt.subplots(2,1)
plt.subplots_adjust(hspace = 0.02)
fig.set_size_inches(14.5, 10.5)
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs[0].axvline(12, linestyle = '-', color = 'k', linewidth = 2, alpha = 0.7)
axs[0].text(12.15,6.4,'K=2R', fontsize = 18)
axs[0].plot(-1+np.arange(35), np.repeat(np.reshape(eigvals, (1,6)), 35, axis = 0), 'k--', linewidth = 2)
axs[0].plot(2+np.arange(30), np.transpose(eigvaldata[0,:,:]), 'o', color='firebrick', markersize = 8)
axs[0].plot(2+np.arange(30), np.ones(30)*(-10), 'k--', label = 'Exact Eigenvalues', linewidth = 2)
axs[0].plot(2+np.arange(30), np.ones(30)*(-10), 'o', color='firebrick', label = 'Estimates from Decaying Signal', markersize = 7)
#axs[0].set_xlabel('$K$', fontsize = 21)
axs[0].xaxis.set_tick_params(labelsize=18)
axs[0].yaxis.set_tick_params(labelsize=18)
axs[0].set_ylabel('Eigenvalues $[a.u.]$', fontsize = 20)
axs[0].set_ylim(-0.5,7)
axs[0].set_xlim(1.8,31.2)
axs[0].grid(alpha = 0.5)
axs[0].legend(loc = 'upper right',prop={'size': 15})

axs[1].axvline(12, linestyle = '-', color = 'k', linewidth = 2, alpha = 0.7)
axs[1].text(12.15,6.4,'K=2R', fontsize = 18)
axs[1].plot(-1+np.arange(35), np.repeat(np.reshape(eigvals, (1,6)), 35, axis = 0), 'k--', linewidth = 2)
axs[1].plot(2+np.arange(30), np.transpose(eigvaldata[1,:,:]), 'o', color='royalblue', markersize = 8)
axs[1].plot(2+np.arange(30), np.ones(30)*(-10), 'k--', label = 'Exact Eigenvalues', linewidth = 2)
axs[1].plot(2+np.arange(30), np.ones(30)*(-10), 'o', color='royalblue', label = 'Estimates from Oscillating Signal', markersize = 7)
axs[1].set_xlabel('$K$', fontsize = 21)
axs[1].set_ylabel('Eigenvalues $[a.u.]$', fontsize = 20)
axs[1].xaxis.set_tick_params(labelsize=18)
axs[1].yaxis.set_tick_params(labelsize=18)
axs[1].set_ylim(-0.5,7)
axs[1].set_xlim(1.8,31.2)
axs[1].grid(alpha = 0.5)
axs[1].legend(loc = 'upper right',prop={'size': 15})
plt.savefig('MPMtest.png', bbox_inches='tight')
plt.savefig('MPMtest.eps', format='eps', bbox_inches='tight')
'''



'''
# Plots
epsilon = 1/20
fig, axs = plt.subplots(2,3)
plt.subplots_adjust(wspace = 0.35, hspace = 0.25)
fig.set_size_inches(20.5, 16.0)
#fig.suptitle('Ferromagnetic Ising Chain in a Transverse Field ($g=4*g_{c}$, $n = 7$).', fontsize = 18, y = 1.05)
axs[0,0].plot(np.arange(measpoints)*tau_increment, gfulldataGS[0,:], 'k', label='Exact $\\langle \\Phi| e^{-\\tau H} |\\Phi \\rangle$')
axs[0,0].set_xlabel('$\\tau$', fontsize = 23)
axs[0,0].set_ylabel('$\\langle \\Phi| e^{-\\tau H} |\\Phi \\rangle$', fontsize = 20, labelpad = 7)
axs[0,0].set_title('$\mathbf{(a)}$ Monte Carlo Scheme, $|\\Phi\\rangle = |+\\rangle^{n}$.', fontsize = 18)
axs[0,0].fill_between(np.arange(measpoints)*tau_increment,gfulldataGS[0,:]-epsilon,gfulldataGS[0,:]+epsilon,color='lightgrey',label='90% confidence bounds') # Confidence bounds
axs[0,0].set_ylim(0,1.05)
axs[0,0].set_xlim(-0.01,5.01)
axs[0,0].plot(np.arange(measpoints)*tau_increment,ftaudataGS[0,:],'o',color='firebrick', markersize=5.5,label = 'Estimates for $M=60$')
axs[0,0].legend(fontsize=10,prop={'size': 17})
axs[0,0].grid(alpha = 0.4)
axs[0,0].xaxis.set_tick_params(labelsize=17)
axs[0,0].yaxis.set_tick_params(labelsize=17)
axs[0,1].plot(np.arange(measpoints)*tau_increment, np.real(qpedataGS[0,:]), 'k', label='Exact $Re(\\langle \\Phi| e^{-it H} |\\Phi \\rangle)$')
axs[0,1].set_xlabel('$t$', fontsize = 23)
axs[0,1].set_ylabel('$Re(\\langle \\Phi| e^{-it H} |\\Phi \\rangle)$', fontsize = 20, labelpad = 7)
axs[0,1].fill_between(np.arange(measpoints)*tau_increment,np.real(qpedataGS[0,:])-epsilon,np.real(qpedataGS[0,:])+epsilon,color='lightgrey',label='90% confidence bounds') # Confidence bounds
axs[0,1].set_title('$\mathbf{(b)}$ Real Part QPE Signal, $|\\Phi\\rangle = |+\\rangle^{n}$.', fontsize = 18)
axs[0,1].set_ylim(-1.05,1.05)
axs[0,1].set_xlim(-0.01,5.01)
axs[0,1].plot(np.arange(measpoints)*tau_increment,np.real(qpenoisydataGS[0,:]),'o',color='royalblue', markersize=5.5,label = 'Estimates for $M=60$')
axs[0,1].legend(fontsize=10,prop={'size': 17})
axs[0,1].grid(alpha = 0.4)
axs[0,1].xaxis.set_tick_params(labelsize=17)
axs[0,1].yaxis.set_tick_params(labelsize=17)
axs[0,2].plot(np.arange(measpoints)*tau_increment, np.imag(qpedataGS[0,:]), 'k', label='Exact $Im(\\langle \\Phi| e^{-it H} |\\Phi \\rangle)$')
axs[0,2].set_xlabel('$t$', fontsize = 23)
axs[0,2].set_ylabel('$Im(\\langle \\Phi| e^{-it H} |\\Phi \\rangle)$', fontsize = 20, labelpad = 7)
axs[0,2].fill_between(np.arange(measpoints)*tau_increment,np.imag(qpedataGS[0,:])-epsilon,np.imag(qpedataGS[0,:])+epsilon,color='lightgrey',label='90% confidence bounds') # Confidence bounds
axs[0,2].set_title('$\mathbf{(c)}$ Imaginary Part QPE Signal, $|\\Phi\\rangle = |+\\rangle^{n}$.', fontsize = 18)
axs[0,2].set_ylim(-1.05,1.05)
axs[0,2].set_xlim(-0.01,5.01)
axs[0,2].plot(np.arange(measpoints)*tau_increment,np.imag(qpenoisydataGS[0,:]),'o',color='seagreen', markersize=5.5,label = 'Estimates for $M=60$')
axs[0,2].legend(fontsize=10,prop={'size': 17})
axs[0,2].grid(alpha = 0.4)
axs[0,2].xaxis.set_tick_params(labelsize=17)
axs[0,2].yaxis.set_tick_params(labelsize=17)
axs[1,0].plot(np.arange(measpoints)*tau_increment, gfulldataES[0,:], 'k', label='Exact $\\langle \\Phi| e^{-\\tau H} |\\Phi \\rangle$')
axs[1,0].set_xlabel('$\\tau$', fontsize = 23)
axs[1,0].set_ylabel('$\\langle \\Phi| e^{-\\tau H} |\\Phi \\rangle$', fontsize = 20, labelpad = 7)
axs[1,0].set_title('$\mathbf{(d)}$ Monte Carlo Scheme, $|\\Phi\\rangle = |\\Phi_{opt,p=0,1}\\rangle$.', fontsize = 18)
axs[1,0].fill_between(np.arange(measpoints)*tau_increment,gfulldataES[0,:]-epsilon,gfulldataES[0,:]+epsilon,color='lightgrey',label='90% confidence bounds') # Confidence bounds
axs[1,0].set_ylim(0,1.05)
axs[1,0].set_xlim(-0.01,5.01)
axs[1,0].plot(np.arange(measpoints)*tau_increment,ftaudataES[0,:],'o',color='firebrick', markersize=5.5,label = 'Estimates for $M=60$')
axs[1,0].legend(fontsize=10,prop={'size': 17})
axs[1,0].grid(alpha = 0.4)
axs[1,0].xaxis.set_tick_params(labelsize=17)
axs[1,0].yaxis.set_tick_params(labelsize=17)
axs[1,1].plot(np.arange(measpoints)*tau_increment, np.real(qpedataES[0,:]), 'k', label='Exact $Re(\\langle \\Phi| e^{-it H} |\\Phi \\rangle)$')
axs[1,1].set_xlabel('$t$', fontsize = 23)
axs[1,1].set_ylabel('$Re(\\langle \\Phi| e^{-it H} |\\Phi \\rangle)$', fontsize = 20, labelpad = 7)
axs[1,1].fill_between(np.arange(measpoints)*tau_increment,np.real(qpedataES[0,:])-epsilon,np.real(qpedataES[0,:])+epsilon,color='lightgrey',label='90% confidence bounds') # Confidence bounds
axs[1,1].set_title('$\mathbf{(e)}$ Real Part QPE Signal, $|\\Phi\\rangle = |\\Phi_{opt,p=0,1}\\rangle$.', fontsize = 18)
axs[1,1].set_ylim(-1.05,1.05)
axs[1,1].set_xlim(-0.01,5.01)
axs[1,1].plot(np.arange(measpoints)*tau_increment,np.real(qpenoisydataES[0,:]),'o',color='royalblue', markersize=5.5,label = 'Estimates for $M=60$')
axs[1,1].legend(fontsize=10,prop={'size': 17})
axs[1,1].grid(alpha = 0.4)
axs[1,1].xaxis.set_tick_params(labelsize=17)
axs[1,1].yaxis.set_tick_params(labelsize=17)
axs[1,2].plot(np.arange(measpoints)*tau_increment, np.imag(qpedataES[0,:]), 'k', label='Exact $Im(\\langle \\Phi| e^{-it H} |\\Phi \\rangle)$')
axs[1,2].set_xlabel('$t$', fontsize = 23)
axs[1,2].set_ylabel('$Im(\\langle \\Phi| e^{-it H} |\\Phi \\rangle)$', fontsize = 20, labelpad = 7)
axs[1,2].fill_between(np.arange(measpoints)*tau_increment,np.imag(qpedataES[0,:])-epsilon,np.imag(qpedataES[0,:])+epsilon,color='lightgrey',label='90% confidence bounds') # Confidence bounds
axs[1,2].set_title('$\mathbf{(f)}$ Imaginary Part QPE Signal, $|\\Phi\\rangle = |\\Phi_{opt,p=0,1}\\rangle$.', fontsize = 18)
axs[1,2].set_ylim(-1.05,1.05)
axs[1,2].set_xlim(-0.01,5.01)
axs[1,2].plot(np.arange(measpoints)*tau_increment,np.imag(qpenoisydataES[0,:]),'o',color='seagreen', markersize=5.5,label = 'Estimates for $M=60$')
axs[1,2].legend(fontsize=10,prop={'size': 17})
axs[1,2].grid(alpha = 0.4)
axs[1,2].xaxis.set_tick_params(labelsize=17)
axs[1,2].yaxis.set_tick_params(labelsize=17)
plt.savefig('realimagevolution.png', bbox_inches='tight')
plt.savefig('realimagevolution.eps', format='eps', bbox_inches='tight')
'''
'''
plt.plot(gdata, eigspectrum, 'k', linewidth = '2')
plt.plot(np.arange(61), eigspectrumnn, 'b', linewidth = '2')
plt.ylim(-20,20)
plt.xlim(0,3.1)
'''
'''
fig, axs = plt.subplots(2, 2)
fig.set_size_inches(15.0, 10.0)
plt.subplots_adjust(hspace = 0.2, wspace = 0.1)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#fig = plt.figure(2)
#fig.set_size_inches(5.0, 6.0)
plt.xlabel('g', fontsize = 20)
plt.ylabel('Eigenvalues($H$) $[\:J\:]$', fontsize = 20)
axs[0,0].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[0,0].plot(gdataest, np.transpose(eigvaldata[1,:,:]), 'o', color = 'firebrick', markersize = 7.5)
axs[0,0].grid(alpha = 0.4)
axs[0,0].set_ylim(0,11)
axs[0,0].set_xlim(0,6.1)
axs[0,0].label_outer()
axs[0,0].set_title('$\mathbf{MC \:Scheme}$ (no Trotter Error, no Noise).', fontsize = 17)
axs[0,1].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[0,1].plot(gdataest, np.transpose(eigvaldata[0,:,:]), 'D', color = 'firebrick', markersize = 7.5)
axs[0,1].grid(alpha = 0.4)
axs[0,1].set_ylim(0,11)
axs[0,1].set_xlim(0,6.1)
axs[0,1].label_outer()
axs[0,1].set_title('$\mathbf{MC \:Scheme}$ ($M = 60$, $|\\Sigma| = 12000$).', fontsize = 17)
axs[1,0].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[1,0].plot(gdataest, np.transpose(eigvaldata[3,:,:]), 'o', color = 'royalblue', markersize = 7.5)
axs[1,0].grid(alpha = 0.4)
axs[1,0].set_ylim(0,11)
axs[1,0].set_xlim(0,6.1)
axs[1,0].label_outer()
axs[1,0].set_title('$\mathbf{QPE}$ (no Trotter Error, no Noise).', fontsize = 17)
axs[1,1].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[1,1].plot(gdataest, np.transpose(eigvaldata[2,:,:]), 'D', color = 'royalblue', markersize = 7.5)
axs[1,1].grid(alpha = 0.4)
axs[1,1].set_ylim(0,11)
axs[1,1].set_xlim(0,6.1)
axs[1,1].label_outer()
axs[1,1].set_title('$\mathbf{QPE}$ ($M = 60$, $|\\Sigma| = 12000$).', fontsize = 17)
axs[0,0].xaxis.set_tick_params(labelsize=15)
axs[0,0].yaxis.set_tick_params(labelsize=15)
axs[0,1].xaxis.set_tick_params(labelsize=15)
axs[0,1].yaxis.set_tick_params(labelsize=15)
axs[1,0].xaxis.set_tick_params(labelsize=15)
axs[1,0].yaxis.set_tick_params(labelsize=15)
axs[1,1].xaxis.set_tick_params(labelsize=15)
axs[1,1].yaxis.set_tick_params(labelsize=15)
plt.savefig('esspectrum25threesets.png', bbox_inches='tight')
plt.savefig('esspectrum25threesets.eps', format='eps', bbox_inches='tight')
'''
'''
fig, axs = plt.subplots(3, 2)
fig.set_size_inches(15.0, 15.0)
plt.subplots_adjust(hspace = 0.2, wspace = 0.1)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#fig = plt.figure(2)
#fig.set_size_inches(5.0, 6.0)
plt.xlabel('g', fontsize = 22)
#plt.ylabel('Eigenvalues($H$) $[\:J\:]$', fontsize = 20)
axs[0,0].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[0,0].plot(gdataest, np.transpose(eigvaldata1[1,:,:]), 'o', color = 'orangered', markersize = 7.5)
axs[0,0].grid(alpha = 0.4)
axs[0,0].set_ylim(0,11)
axs[0,0].set_xlim(0,6.1)
axs[0,0].label_outer()
axs[0,0].set_title('$\mathbf{MC \:Scheme}$ (no Trotter Error, no Noise):$\:\:TF \:=\: 0.02}$.', fontsize = 16)
axs[0,0].set_ylabel('Eigenvalues($H$) $[\:J\:]$', fontsize = 20)
axs[0,1].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[0,1].plot(gdataest, np.transpose(eigvaldata1[0,:,:]), 'D', color = 'orangered', markersize = 7.5)
axs[0,1].grid(alpha = 0.4)
axs[0,1].set_ylim(0,11)
axs[0,1].set_xlim(0,6.1)
axs[0,1].label_outer()
axs[0,1].set_title('$\mathbf{MC \:Scheme}$ ($M = 60$, $|\\Sigma| = 12000$):$\:\:TF \:=\: 0.02}$.', fontsize = 16)
axs[1,0].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[1,0].plot(gdataest, np.transpose(eigvaldata2[1,:,:]), 'o', color = 'orangered', markersize = 7.5)
axs[1,0].grid(alpha = 0.4)
axs[1,0].set_ylim(0,11)
axs[1,0].set_xlim(0,6.1)
axs[1,0].label_outer()
axs[1,0].set_title('$\mathbf{MC \:Scheme}$ (no Trotter Error, no Noise):$\:\:TF \:=\: 0.002}$.', fontsize = 16)
axs[1,0].set_ylabel('Eigenvalues($H$) $[\:J\:]$', fontsize = 20)
axs[1,1].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[1,1].plot(gdataest, np.transpose(eigvaldata2[0,:,:]), 'D', color = 'orangered', markersize = 7.5)
axs[1,1].grid(alpha = 0.4)
axs[1,1].set_ylim(0,11)
axs[1,1].set_xlim(0,6.1)
axs[1,1].label_outer()
axs[1,1].set_title('$\mathbf{MC \:Scheme}$ ($M = 60$, $|\\Sigma| = 12000$):$\:\:TF \:=\: 0.002}$.', fontsize = 16)
axs[2,0].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[2,0].plot(gdataest, np.transpose(eigvaldata3[1,:,:]), 'o', color = 'orangered', markersize = 7.5)
axs[2,0].grid(alpha = 0.4)
axs[2,0].set_ylim(0,11)
axs[2,0].set_xlim(0,6.1)
axs[2,0].label_outer()
axs[2,0].set_title('$\mathbf{MC \:Scheme}$ (no Trotter Error, no Noise):$\:\:TF \:=\: 0.0002}$.', fontsize = 16)
axs[2,0].set_ylabel('Eigenvalues($H$) $[\:J\:]$', fontsize = 20)
axs[2,1].plot(gdata, eigspectrum, 'k', linewidth = '2')
axs[2,1].plot(gdataest, np.transpose(eigvaldata3[0,:,:]), 'D', color = 'orangered', markersize = 7.5)
axs[2,1].grid(alpha = 0.4)
axs[2,1].set_ylim(0,11)
axs[2,1].set_xlim(0,6.1)
axs[2,1].label_outer()
axs[2,1].set_title('$\mathbf{MC \:Scheme}$ ($M = 60$, $|\\Sigma| = 12000$):$\:\:TF \:=\: 0.0002}$.', fontsize = 16)
axs[0,0].xaxis.set_tick_params(labelsize=15)
axs[0,0].yaxis.set_tick_params(labelsize=15)
axs[0,1].xaxis.set_tick_params(labelsize=15)
axs[0,1].yaxis.set_tick_params(labelsize=15)
axs[1,0].xaxis.set_tick_params(labelsize=15)
axs[1,0].yaxis.set_tick_params(labelsize=15)
axs[1,1].xaxis.set_tick_params(labelsize=15)
axs[1,1].yaxis.set_tick_params(labelsize=15)
axs[2,0].xaxis.set_tick_params(labelsize=15)
axs[2,0].yaxis.set_tick_params(labelsize=15)
axs[2,1].xaxis.set_tick_params(labelsize=15)
axs[2,1].yaxis.set_tick_params(labelsize=15)
plt.savefig('esspectrum25TFnew.png', bbox_inches='tight')
plt.savefig('esspectrum25TFnew.eps', format='eps', bbox_inches='tight')
'''
'''
fig = plt.figure(4)
plt.xlabel('g', fontsize = 14)
plt.ylabel('Eigenvalues($H$) $[\:J\:]$', fontsize = 14)
plt.plot(gdata, lowset, 'k')
plt.plot(gdata, expzero, 'k--', label = '$\\langle 0|^{\otimes n} H |0 \\rangle^{\otimes n}$', linewidth = 2)
plt.plot(gdata, expplus, 'k-.', label = '$\\langle +|^{\otimes n} H |+ \\rangle^{\otimes n}$', linewidth = 2)
plt.ylim(0,5.5)
plt.xlim(0,5)
plt.grid(alpha = 0.4)
fig.set_size_inches(7.0, 4)
plt.legend(loc = 'upper right',prop={'size': 14})
plt.savefig('expvalues.png', bbox_inches='tight')
plt.savefig('expvalues.eps', format='eps', bbox_inches='tight')
'''
'''
Kdata = np.arange(20)*2 + 4
fig = plt.figure(2)
fig.set_size_inches(9.0, 5.5)
plt.xlabel('$\\tau_{interval}/\\Delta \\tau$', fontsize = 19, size = 15)
plt.ylabel('Eigenvalues($H$) $[\:J\:]$', fontsize = 18, size = 15)
plt.plot(Kdata, Kconvergence_eigvalconvergenceM_exact[:,0], '-o', markersize=8, color='k',label='Eigenvalue Estimates of Noiseless signal')
plt.title('Ground state energy estimates obtained from the $\\langle +|^{\otimes n} e^{-\\tau H} |+ \\rangle^{\otimes n}$ signal for an Ising Chain in \n a Transverse Field ($g=3*g_{c}$, $n = 7$) as a function of $\\tau_{interval}/\\Delta \\tau$ (for $|\\Sigma|=1500$).', fontsize = 17)
plt.ylim(0,0.85)
plt.xlim(3.5,42.5)
plt.grid(alpha = 0.4)
plt.plot(Kdata,Kconvergence_eigvalconvergenceM5[:,0], '-o', markersize=8, color='tomato',label='M=5 Estimates')
plt.plot(Kdata,Kconvergence_eigvalconvergenceM15[:,0], '-o', markersize=8, color='dodgerblue',label='M=15 Estimates')
plt.plot(Kdata,Kconvergence_eigvalconvergenceM25[:,0], '-o', markersize=8, color='seagreen',label='M=25 Estimates')
plt.legend(loc = 'upper right',prop={'size': 13})
plt.savefig('Kconvergence2.png', bbox_inches='tight')
plt.savefig('Kconvergence2.eps', format='eps', bbox_inches='tight')
'''
'''
Mdata = np.arange(25) + 1
fig = plt.figure(6)
fig.set_size_inches(9.5, 4.5)
plt.xlabel('M', fontsize = 14)
plt.ylabel('Relative Error $[\:\%\:]$', fontsize = 14)
plt.plot(Mdata, 100*M500convergencedata, '-o', markersize=6, color='k',label='$|\\Sigma| = 500$')
plt.plot(Mdata, 100*M2000convergencedata, '-o', markersize=6, color='firebrick',label='$|\\Sigma| = 2000$')
plt.plot(Mdata, 100*M4000convergencedata, '-o', markersize=6, color='forestgreen',label='$|\\Sigma| = 4000$')
#plt.title('Eigenvalue estimates obtained from $\\langle +|^{\otimes n} e^{-\\tau H} |+ \\rangle^{\otimes n}$ as a function of $\\tau$ \n for an Ising Chain with a Transverse Field ($g=3*g_{c}$, $N = 7$) as a function of $\\tau_{interval}/\\Delta \\tau$', fontsize = 14)
plt.ylim(0,110)
plt.xlim(0.5,25.5)
plt.grid(alpha = 0.4)
plt.legend(loc = 'upper right')
plt.savefig('Mconvergence.png', bbox_inches='tight')
plt.savefig('Mconvergence.eps', format='eps', bbox_inches='tight')
'''
'''
fig, axs = plt.subplots(2, 9)
fig.set_size_inches(12.7, 11.0)
plt.subplots_adjust(hspace = 0.2, wspace = 2.0)

a1 = plt.subplot2grid((2,9), (0,0), colspan = 4, rowspan = 1)
a2 = plt.subplot2grid((2,9), (0,4), colspan = 4, rowspan = 1)
a1.imshow(relerror200*100,cmap=plt.cm.gnuplot,extent=[0,5.1,25,1],aspect=0.20)
im = a2.imshow(relerror4000*100,cmap=plt.cm.gnuplot,extent=[0,5.1,25,1],aspect=0.20)
a1.set_xlabel('$\\tau$', fontsize = 19)
a1.set_ylabel('$M$', fontsize = 18, rotation = 0, labelpad = 15)
a2.set_xlabel('$\\tau$', fontsize = 19)
a2.set_ylabel('$M$', fontsize = 18, rotation = 0, labelpad = 15)
a1.set_title('$\mathbf{(a)}$     $|\\Sigma| = 200$', fontsize = 18)
a2.set_title('$\mathbf{(b)}$     $|\\Sigma| = 4000$', fontsize = 18)
a1.xaxis.set_tick_params(labelsize=14)
a1.yaxis.set_tick_params(labelsize=14)
a2.xaxis.set_tick_params(labelsize=14)
a2.yaxis.set_tick_params(labelsize=14)
#fig.suptitle('Relative error $[\\%]$ \n in $\\langle +|^{\otimes n} e^{-\\tau H} |+ \\rangle^{\otimes n}$ estimates.', fontsize = 18)
axc = plt.subplot2grid((2,9),(0,8), colspan = 1, rowspan = 1)
fig.colorbar(im, cax=axc, fraction=0.001, pad=0.06, orientation = 'vertical')
axc.set_ylabel('Relative error $[\\%]$ \n in $\\langle +|^{\otimes n} e^{-\\tau H} |+ \\rangle^{\otimes n}$ estimates.', fontsize = 16, rotation = 270, labelpad = 40)
a4 = plt.subplot2grid((2,9), (1,0), colspan = 9, rowspan = 1)
a4.plot(Mdata, 100*M200convergencedata, '-o', markersize=9, label='$|\\Sigma| = 200$')
#a4.plot(Mdata, 100*M500convergencedata, '-o', markersize=9, label='$|\\Sigma| = 500$')
#a4.plot(Mdata, 100*M2000convergencedata, '-o', markersize=9, label='$|\\Sigma| = 2000$')
a4.plot(Mdata, 100*M4000convergencedata, '-o', markersize=9, label='$|\\Sigma| = 4000$')
a4.set_xlabel('$M$', fontsize = 17)
a4.set_ylabel('Relative Error $[\:\%\:]$', fontsize = 17)
a4.set_title('$\mathbf{(c)}$                                                                                                                                 ', fontsize = 18)
a4.set_ylim(0,105)
a4.set_xlim(0.5,25.5)
a4.grid(alpha = 0.4)
a4.legend(loc = 'upper right',prop={'size': 15})
a4.xaxis.set_tick_params(labelsize=14)
a4.yaxis.set_tick_params(labelsize=14)
plt.savefig('noiseresilienceGS2.png', bbox_inches='tight')
plt.savefig('noiseresilienceGS2.eps', format='eps', bbox_inches='tight')
'''
'''
n001, bins, _ = plt.hist(ftauestimatesS200g4M35n7VAR[0,:]-gfullS200g4M35n7VAR[0,:], range = (-0.25,0.25), bins = 29)
n01, bins, _ = plt.hist(ftauestimatesS200g4M35n7VAR[1,:]-gfullS200g4M35n7VAR[1,:], range = (-0.25,0.25), bins = 29)
n1, bins, _ = plt.hist(ftauestimatesS200g4M35n7VAR[2,:]-gfullS200g4M35n7VAR[2,:], range = (-0.25,0.25), bins = 29)
bincenters = 0.5*(bins[1:] + bins[:-1])
fig = plt.figure(9)
fig.set_size_inches(5.5, 4.5)
plt.plot(np.linspace(min(bincenters),max(bincenters),1000),spline(bincenters,n001,np.linspace(min(bincenters),max(bincenters),1000)), label = '$\\tau = 0.01$', linewidth = 3)
plt.plot(np.linspace(min(bincenters),max(bincenters),1000),spline(bincenters,n01,np.linspace(min(bincenters),max(bincenters),1000)), label = '$\\tau = 0.1$', linewidth = 3)
plt.plot(np.linspace(min(bincenters),max(bincenters),1000),spline(bincenters,n1,np.linspace(min(bincenters),max(bincenters),1000)), label = '$\\tau = 1$', linewidth = 3)
plt.xlabel('Absolute Error', fontsize = 17)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.ylabel('Ticks', fontsize = 17)
plt.ylim(0,1100)
plt.xlim(-0.25,0.25)
#plt.grid(alpha = 0.4)
plt.legend(loc = 'upper right',prop={'size': 16})
plt.axvline(1/(20**(1/2)), color = 'k', linestyle = '--')
plt.text(-0.22,40,'$-\\epsilon$',fontsize=18)
plt.axvline(-1/(20**(1/2)), color = 'k', linestyle = '--')
plt.text(0.175,40,'$+\\epsilon$',fontsize=18)
plt.savefig('tauVAR2.png', bbox_inches='tight')
plt.savefig('tauVAR2.eps', format='eps', bbox_inches='tight')
'''
'''
fig, axs = plt.subplots(1,2)
plt.subplots_adjust(wspace = 0.05)
fig.set_size_inches(14.5, 5.5)
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs[0].plot(np.arange(35)*6/19, np.ones(35)*1/200, 'k--', label = '$\\tau$-independent variance bound', linewidth = 3)
axs[0].plot(np.arange(35)*6/19, (1-(VARANALYSISgfulldataM60g4plus[:,0])**2)/200, 'k', label = '$\\tau$-dependent variance bound', linewidth = 2)
axs[0].plot(np.arange(35)*6/19, np.var(VARANALYSISftaudataM60g4plus, axis = 1), 'o', color='firebrick', label = 'variance of MC samples', markersize = 7)
axs[0].set_xlabel('$\\tau$', fontsize = 19)
axs[0].xaxis.set_tick_params(labelsize=14)
axs[0].yaxis.set_tick_params(labelsize=14)
axs[0].set_ylabel('Variance', fontsize = 18)
axs[0].set_ylim(-0.0001,0.0071)
axs[0].set_xlim(-0.1,10.8)
#axs[0].grid(alpha = 0.4)
axs[0].legend(loc = 'upper right',prop={'size': 13}, frameon=False)
axs[0].set_title('$\mathbf{(a)}$ Monte Carlo Scheme.', fontsize = 17)

axs[1].plot(np.arange(35)*6/19, np.ones(35)*1/200, 'k--', label = '$t$-independent \n variance bound', linewidth = 3)
axs[1].plot(np.arange(241)*0.05, (1-(QPEnontrotsignalrealuse)**2)/200, 'royalblue', label = '$t$-dependent \n variance (Re)', linewidth = 2)
axs[1].plot(np.arange(35)*6/19, np.var(np.real(VARANALYSISqpenoisydataM60g4plus), axis = 1), 'o', color='royalblue', label = 'variance of QPE \n samples (Re)', markersize = 7)
axs[1].plot(np.arange(241)*0.05, (1-(QPEnontrotsignalimaguse)**2)/200, 'seagreen', label = '$t$-dependent \n variance (Im)', linewidth = 2)
axs[1].plot(np.arange(35)*6/19, np.var(np.imag(VARANALYSISqpenoisydataM60g4plus), axis = 1), 'o', color='seagreen', label = 'variance of QPE \n samples (Im)', markersize = 7)
axs[1].plot(np.arange(35)*6/19, np.var(VARANALYSISqpenoisydataM60g4plus, axis = 1), 'D', color='k', label = 'variance of QPE \n samples (complete \n signal)', markersize = 6.5)
axs[1].set_xlabel('$t$', fontsize = 19)
axs[1].xaxis.set_tick_params(labelsize=14)
axs[1].set_yticks([],[])
axs[1].set_ylim(-0.0001,0.0071)
axs[1].set_xlim(-0.1,10.8)
#axs[1].grid(alpha = 0.4)
axs[1].legend(loc = 'upper right',prop={'size': 13}, bbox_to_anchor=(1.48, 1.05), frameon=False)
axs[1].set_title('$\mathbf{(b)}$ Quantum Phase Estimation.', fontsize = 17)
plt.savefig('variancetdep.png', bbox_inches='tight')
plt.savefig('variancetdep.eps', format='eps', bbox_inches='tight')


fig, axs = plt.subplots(1,2)
plt.subplots_adjust(wspace = 0.05)
fig.set_size_inches(14.5, 5.5)
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs[0].plot(np.arange(24)*2/10-0.5, np.ones(24)*1/200, 'k--', label = '$\\tau$-independent variance bound', linewidth = 3)
axs[0].plot(np.arange(300)*0.01, (1-(gfulldata)**2)/200, 'k', label = '$\\tau$-dependent variance bound', linewidth = 2)
axs[0].plot(np.arange(24)*1/10, np.var(VARftaudataES, axis = 1), 'o', color='firebrick', label = 'variance of MC samples', markersize = 7)
axs[0].set_xlabel('$\\tau$', fontsize = 19)
axs[0].xaxis.set_tick_params(labelsize=14)
axs[0].yaxis.set_tick_params(labelsize=14)
axs[0].set_ylabel('Variance', fontsize = 18)
axs[0].set_ylim(-0.0001,0.0110)
axs[0].set_xlim(-0.03,2.31)
#axs[0].grid(alpha = 0.4)
axs[0].legend(loc = 'upper right',prop={'size': 13}, frameon=False)
axs[0].set_title('$\mathbf{(a)}$ Monte Carlo Scheme.', fontsize = 17)

axs[1].plot(np.arange(24)*2/10-0.5, np.ones(24)*1/200, 'k--', label = '$t$-independent \n variance bounds', linewidth = 3)
axs[1].plot(np.arange(24)*2/10-0.5, np.ones(24)*2/200, 'k--', linewidth = 3)
axs[1].plot(np.arange(300)*0.01, (1-(np.real(qpedata))**2)/200, 'royalblue', label = '$t$-dependent \n variance (Re)', linewidth = 2)
#axs[1].plot(np.arange(300)*0.01, (1-(1/2*np.cos(0.4224*np.arange(300)*0.01)+1/2*np.cos(6.4225*np.arange(300)*0.01))**2)/200, 'royalblue', label = '$t$-dependent \n variance (Re)', linewidth = 2)
axs[1].plot(np.arange(24)*1/10, np.var(np.real(VARqpenoisydataES), axis = 1), 'o', color='royalblue', label = 'variance of QPE \n samples (Re)', markersize = 7)
axs[1].plot(np.arange(300)*0.01, (1-(np.imag(qpedata))**2)/200, 'seagreen', label = '$t$-dependent \n variance (Im)', linewidth = 2)
#axs[1].plot(np.arange(300)*0.01, (1-(1/2*np.sin(0.4224*np.arange(300)*0.01)+1/2*np.sin(6.4225*np.arange(300)*0.01))**2)/200, 'seagreen', label = '$t$-dependent \n variance (Im)', linewidth = 2)
axs[1].plot(np.arange(24)*1/10, np.var(np.imag(VARqpenoisydataES), axis = 1), 'o', color='seagreen', label = 'variance of QPE \n samples (Im)', markersize = 7)
#axs[1].plot(np.arange(300)*0.01, (1-(np.imag(qpedata))**2)/200 + (1-(np.real(qpedata))**2)/200, 'k', label = '$t$-dependent \n variance (complete \n signal)', linewidth = 2)
axs[1].plot(np.arange(300)*0.01, 1/200*(3/2-1/2*np.cos((6.4225-0.4224)*np.arange(300)*0.01)), 'k', label = '$t$-dependent \n variance (complete \n signal)', linewidth = 2)
axs[1].plot(np.arange(24)*1/10, np.var(VARqpenoisydataES, axis = 1), 'D', color='k', label = 'variance of QPE \n samples (complete \n signal)', markersize = 6.5)
axs[1].set_xlabel('$t$', fontsize = 19)
axs[1].xaxis.set_tick_params(labelsize=14)
axs[1].set_yticks([],[])
axs[1].set_ylim(-0.0001,0.0110)
axs[1].set_xlim(-0.03,2.31)
#axs[1].grid(alpha = 0.4)
axs[1].legend(loc = 'upper right',prop={'size': 13}, bbox_to_anchor=(1.48, 1.05), frameon=False)
axs[1].set_title('$\mathbf{(b)}$ Quantum Phase Estimation.', fontsize = 17)
plt.savefig('variancetdepES.png', bbox_inches='tight')
plt.savefig('variancetdepES.eps', format='eps', bbox_inches='tight')


fig, axs = plt.subplots(1,2)
plt.subplots_adjust(wspace = 0.25)
fig.set_size_inches(14, 10)
axs[0].plot(1+np.arange(measpoints)*5, 3*np.exp(2)*Ntermtrotterbound, color = 'firebrick', linewidth = 2)
axs[0].plot(1+np.arange(measpoints)*5, Ntermimagtrotter, 'o', color = 'firebrick')
axs[0].plot(1+np.arange(measpoints)*5, 3*np.exp(2)*Twotermtrotterbound, color = 'orangered', linewidth = 2)
axs[0].plot(1+np.arange(measpoints)*5, Twotermimagtrotter, 'o', color = 'orangered')
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].text(2.5, 3800, r'$N$-Term Decomposition Bound (1st order)', fontsize=14, rotation = -29, color = 'firebrick')
axs[0].text(3.5, 900, r'$\Gamma$-Term Decomposition Bound (1st order)', fontsize=14, rotation = -29, color = 'orangered')
axs[0].grid(alpha = 0.4, which = 'both')
axs[0].set_ylim(0.001,5000)
axs[0].set_xlim(0.9,210)
axs[0].set_title('$\mathbf{(a)}$ Imaginary-Time Propagation.', fontsize = 17)
axs[0].xaxis.set_tick_params(labelsize=14)
axs[0].yaxis.set_tick_params(labelsize=14)
axs[0].set_xlabel('$M$', fontsize = 19)
axs[0].set_ylabel('Absolute Error', fontsize = 18)
axs[1].plot(1+np.arange(measpoints)*5, Ntermtrotterbound, color = 'royalblue', linewidth = 2)
axs[1].plot(1+np.arange(measpoints)*5, Ntermrealtrotter, 'o', color = 'royalblue')
axs[1].plot(1+np.arange(measpoints)*5, Twotermtrotterbound, color = 'seagreen', linewidth = 2)
axs[1].plot(1+np.arange(measpoints)*5, Twotermrealtrotter, 'o', color = 'seagreen')
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].text(1, 440, r'$N$-Term Decomposition Bound (1st order)', fontsize=14, rotation = -29, color = 'royalblue')
axs[1].text(3.5, 40, r'$\Gamma$-Term Decomposition Bound (1st order)', fontsize=14, rotation = -29, color = 'seagreen')
axs[1].grid(alpha = 0.4, which = 'both')
axs[1].set_ylim(0.001,5000)
axs[1].set_xlim(0.9,210)
axs[1].set_title('$\mathbf{(b)}$ Real-Time Propagation.', fontsize = 17)
axs[1].xaxis.set_tick_params(labelsize=14)
axs[1].yaxis.set_tick_params(labelsize=14)
axs[1].set_xlabel('$M$', fontsize = 19)
axs[1].set_ylabel('Absolute Error', fontsize = 18)
plt.savefig('trotteranalysis.png', bbox_inches='tight')
plt.savefig('trotteranalysis.eps', format='eps', bbox_inches='tight')

fig, axs = plt.subplots(1,2, sharey = True)
plt.subplots_adjust(wspace = 0.05)
fig.set_size_inches(15, 7)
axs[0].plot(xnumsamplesprime, np.sort(FullHeigvals)[0]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--', label = 'Exact Eigenvalues                         ')
axs[0].plot(xnumsamplesprime, np.sort(FullHeigvals)[1]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[0].plot(xnumsamples, np.transpose(eigvaldata[2,:,:]), 'X', color = 'royalblue', markersize = 10)
axs[0].plot(np.array([10**(-3)]), np.array([-100]), 'X', color = 'firebrick', markersize = 10, label = 'Estimates obtained from MC signal')
axs[0].plot(np.array([10**(-3)]), np.array([-100]), 'X', color = 'royalblue', markersize = 10, label = 'Estimates obtained from QPE signal')
axs[0].set_xscale('log')
axs[0].text(700, -0.8, r'Ground State', fontsize=15, color = 'k')
axs[0].text(400, 5, r'1st Excited State', fontsize=15, color = 'k')
axs[0].grid(alpha = 0.4, which = 'both')
axs[0].set_ylim(-1.5,17)
axs[0].set_xlim(2,6250)
#axs[0].set_title('$\mathbf{(a)}$ Imaginary-Time Propagation.', fontsize = 17)
axs[0].xaxis.set_tick_params(labelsize=18)
axs[0].yaxis.set_tick_params(labelsize=17)
axs[0].set_xlabel('$|\Sigma|$', fontsize = 25)
axs[0].set_ylabel('Eigenvalues $[\:J\:]$', fontsize = 21)
axs[0].legend(loc = 'lower right', prop={'size': 16}, bbox_to_anchor=(1.11, -0.75), frameon=False, mode = 'expand', ncol = 3, borderpad = 12)

axs[1].plot(xnumsamplesprime, np.sort(FullHeigvals)[0]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[1].plot(xnumsamplesprime, np.sort(FullHeigvals)[1]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[1].plot(xnumsamples, np.transpose(eigvaldata[0,:,:]), 'X', color = 'firebrick', markersize = 10)
axs[1].set_xscale('log')
axs[1].text(700, -0.8, r'Ground State', fontsize=15, color = 'k')
axs[1].text(400, 5, r'1st Excited State', fontsize=15, color = 'k')
#axs[1].text(1, 440, r'$N$-Term Decomposition Bound (1st order)', fontsize=14, rotation = -29, color = 'royalblue')
#axs[1].text(3.5, 40, r'$\Gamma$-Term Decomposition Bound (1st order)', fontsize=14, rotation = -29, color = 'seagreen')
axs[1].grid(alpha = 0.4, which = 'both')
axs[1].set_ylim(-1.5,17)
axs[1].set_xlim(2,6250)
#axs[1].set_title('$\mathbf{(b)}$ Real-Time Propagation.', fontsize = 17)
axs[1].xaxis.set_tick_params(labelsize=18)
axs[1].yaxis.set_tick_params(labelsize=17)
axs[1].set_xlabel('$|\Sigma|$', fontsize = 25)
#axs[1].set_ylabel('Eigenvalues $[\:J\:]$', fontsize = 19)
plt.savefig('qpevmcNumSamplesanalysis2.png', bbox_inches='tight')
plt.savefig('qpevmcNumSamplesanalysis2.eps', format='eps', bbox_inches='tight')

colors = plt.cm.winter(np.linspace(0,1,4))
fig, axs = plt.subplots(1,4, sharey = True)
plt.subplots_adjust(wspace = 0.10)
fig.set_size_inches(20, 7)
axs[0].plot(np.array([10**(-3)]), np.array([-100]), 'o', color = colors[3], markersize = 8, label = 'QPE Estimates $\mathbf{(TF = 0.97)}$')
axs[0].plot(np.array([10**(-3)]), np.array([-100]), 'o', color = colors[2], markersize = 8, label = 'QPE Estimates $\mathbf{(TF = 0.94)}$')
axs[0].plot(np.array([10**(-3)]), np.array([-100]), 'o', color = colors[1], markersize = 8, label = 'QPE Estimates $\mathbf{(TF = 0.91)}$')
axs[0].plot(np.array([10**(-3)]), np.array([-100]), 'o', color = colors[0], markersize = 8, label = 'QPE Estimates $\mathbf{(TF = 0.88)}$')
axs[0].plot(xnumsamplesprime, np.sort(FullHeigvals)[0]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[1].plot(xnumsamplesprime, np.sort(FullHeigvals)[0]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--', label = 'Exact Eigenvalues')
axs[2].plot(xnumsamplesprime, np.sort(FullHeigvals)[0]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[3].plot(xnumsamplesprime, np.sort(FullHeigvals)[0]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[0].plot(xnumsamplesprime, np.sort(FullHeigvals)[1]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[1].plot(xnumsamplesprime, np.sort(FullHeigvals)[1]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[2].plot(xnumsamplesprime, np.sort(FullHeigvals)[1]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[3].plot(xnumsamplesprime, np.sort(FullHeigvals)[1]*np.ones(17), color = 'k', linewidth = 2.5, linestyle = '--')
axs[0].plot(xnumsamples, np.transpose(qpeeigvaldata[0,:,:]), 'o', color = colors[0], markersize = 8)
axs[1].plot(xnumsamples, np.transpose(qpeeigvaldata[1,:,:]), 'o', color = colors[1], markersize = 8)
axs[2].plot(xnumsamples, np.transpose(qpeeigvaldata[2,:,:]), 'o', color = colors[2], markersize = 8)
axs[3].plot(xnumsamples, np.transpose(qpeeigvaldata[3,:,:]), 'o', color = colors[3], markersize = 8)
axs[0].set_xscale('log')
axs[1].set_xscale('log')
axs[2].set_xscale('log')
axs[3].set_xscale('log')
axs[0].text(150, -0.4, r'Ground State', fontsize=15, color = 'k')
axs[0].text(70, 6.9, r'1st Excited State', fontsize=15, color = 'k')
axs[1].text(150, -0.4, r'Ground State', fontsize=15, color = 'k')
axs[1].text(70, 6.9, r'1st Excited State', fontsize=15, color = 'k')
axs[2].text(150, -0.4, r'Ground State', fontsize=15, color = 'k')
axs[2].text(70, 6.9, r'1st Excited State', fontsize=15, color = 'k')
axs[3].text(150, -0.4, r'Ground State', fontsize=15, color = 'k')
axs[3].text(70, 6.9, r'1st Excited State', fontsize=15, color = 'k')
axs[0].grid(alpha = 0.4, which = 'both')
axs[1].grid(alpha = 0.4, which = 'both')
axs[2].grid(alpha = 0.4, which = 'both')
axs[3].grid(alpha = 0.4, which = 'both')
axs[0].set_ylim(-1.5,9)
axs[0].set_xlim(2,2700)
axs[1].set_ylim(-1.5,9)
axs[1].set_xlim(2,2700)
axs[2].set_ylim(-1.5,9)
axs[2].set_xlim(2,2700)
axs[3].set_ylim(-1.5,9)
axs[3].set_xlim(2,2700)
axs[0].xaxis.set_tick_params(labelsize=16)
axs[0].yaxis.set_tick_params(labelsize=16)
axs[1].xaxis.set_tick_params(labelsize=16)
axs[1].yaxis.set_tick_params(labelsize=16)
axs[2].xaxis.set_tick_params(labelsize=16)
axs[2].yaxis.set_tick_params(labelsize=16)
axs[3].xaxis.set_tick_params(labelsize=16)
axs[3].yaxis.set_tick_params(labelsize=16)
axs[0].set_ylabel('Eigenvalues $[\:J\:]$', fontsize = 19)
axs[0].set_xlabel('$|\Sigma|$', fontsize = 24)
axs[1].set_xlabel('$|\Sigma|$', fontsize = 24)
axs[2].set_xlabel('$|\Sigma|$', fontsize = 24)
axs[3].set_xlabel('$|\Sigma|$', fontsize = 24)
axs[0].legend(loc = 'lower right', prop={'size': 15}, frameon = False, bbox_to_anchor=(2.1, -1.05), mode = 'expand', ncol = 4, borderpad = 20)
axs[1].legend(loc = 'lower right', prop={'size': 15}, frameon = False, bbox_to_anchor=(0.7, -0.35), mode = 'expand', ncol = 1, borderpad = 0)
plt.savefig('qpetfanalysis.png', bbox_inches='tight')
plt.savefig('qpetfanalysis.eps', format='eps', bbox_inches='tight')

'''
'''
fig, axs = plt.subplots(1,2)
plt.subplots_adjust(wspace = 0.05)
fig.set_size_inches(14.5, 5.5)
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs[0].plot(p, numpropt1eps0001, 'o', color='seagreen',label = '$\\epsilon=0.001$', markersize = 8)
axs[0].plot(p, numpropt1eps001, 'o', color='royalblue',label = '$\\epsilon=0.01$', markersize = 8)
axs[0].plot(p, numpropt1eps01, 'o', color='orangered', label = '$\\epsilon=0.1$', markersize = 8)
axs[0].plot(p, numpropt1eps1, 'o', color='firebrick', label = '$\\epsilon=1$', markersize = 8)

axs[0].plot(np.array([1]), 1**(2)/(0.001), 'o', color='seagreen', markersize = 8)
axs[0].plot(np.array([1]), 1**(2)/(0.01), 'o', color='royalblue', markersize = 8)
axs[0].plot(np.array([1]), 1**(2)/(0.1), 'o', color='orangered', markersize = 8)
axs[0].plot(np.array([1]), 1**(2)/(1), 'o', color='firebrick', markersize = 8)

axs[0].set_xlabel('$p$', fontsize = 21)
axs[0].set_yscale('log')
axs[0].xaxis.set_tick_params(labelsize=15)
axs[0].yaxis.set_tick_params(labelsize=15)
axs[0].set_ylabel('$\\propto$ # of local propagation operators', fontsize = 19)
axs[0].set_ylim(0.0005,10**6)
axs[0].set_xlim(0.4,15)
axs[0].grid(alpha = 0.4, which="both")
axs[0].legend(loc = 'lower right',prop={'size': 16})
axs[0].set_title('$\\tau = t = 1$', fontsize = 20)

axs[1].plot(p, numpropt001eps01, 'o', color='seagreen',label = '$\\tau = t = 0.01$', markersize = 8)
axs[1].plot(p, numpropt01eps01, 'o', color='royalblue',label = '$\\tau = t = 0.1$', markersize = 8)
axs[1].plot(p, numpropt1eps01, 'o', color='orangered', label = '$\\tau = t = 1$', markersize = 8)
axs[1].plot(p, numpropt10eps01, 'o', color='firebrick', label = '$\\tau = t = 10$', markersize = 8)

axs[1].plot(np.array([1]), 0.01**(2)/(0.1), 'o', color='seagreen', markersize = 8)
axs[1].plot(np.array([1]), 0.1**(2)/(0.1), 'o', color='royalblue', markersize = 8)
axs[1].plot(np.array([1]), 1**(2)/(0.1), 'o', color='orangered', markersize = 8)
axs[1].plot(np.array([1]), 10**(2)/(0.1), 'o', color='firebrick', markersize = 8)

axs[1].set_xlabel('$p$', fontsize = 21)
axs[1].set_yscale('log')
axs[1].xaxis.set_tick_params(labelsize=15)
axs[1].yaxis.set_tick_params(labelsize=15)
axs[1].set_ylim(0.0005,10**6)
axs[1].set_xlim(0.4,15)
axs[1].grid(alpha = 0.4, which="both")
axs[1].set_yticklabels([])
axs[1].legend(loc = 'lower right',prop={'size': 16})
axs[1].set_title('$\\epsilon = 0.1$', fontsize = 20)
plt.savefig('trotterlocprop.png', bbox_inches='tight')
plt.savefig('trotterlocprop.eps', format='eps', bbox_inches='tight')
'''