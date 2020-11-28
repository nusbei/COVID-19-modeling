import numpy as np
import progressbar
import matplotlib.pyplot as plt
from warnings import warn


class lbinv:
    r'''invert for l and beta to establish the relationship between R0 and s'''
    def __init__(self,dmg,Hd,s,gn,fx,sig,wG,NI_y):
        r'''initialization'''
        self.dmg = dmg # static population distribution (2D float array: (ny-by-nx))
        self.Hd = Hd # infection event history (infevnt class)
        self.s = s # stringency index (1D float array: (nt-))
        self.gn = gn # normalized infectability time sequence (1D float array: (ng-))
        self.ng = len(gn) # length of gn (int scalar)
        self.fx = fx # basic spatial influence (fxxi class)
        self.sig = sig # gaussian distribution parameter varying with time (1D float array: (nt-))
        self.wG = wG # weight of the gaussian distribution varying with time (1D float array: (nt-))
        self.nt = len(sig) # total modeling days (int scalar)
        self.F = self.Fcal() # diagnal forward modeling operator
        self.NI = NI_y # fitted data
        
    def Fcal(self):
        r'''calculate the F sequence that directly apply on R0'''
        I = np.zeros(self.fx.shape)
        F = np.zeros(self.nt)
        barF = progressbar.ProgressBar(maxval=self.nt, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        print(f'Fcal progress:')
        barF.start()
        for i in range(self.nt):
            barF.update(i+1)
            if i>0:
                idxi, Nii = self.Hd.evnt_t(i-1)
                subi = tuple(idxi.transpose())
                I[subi] += Nii/self.dmg[subi]
            y = np.zeros(self.fx.shape)
            for j in range(self.ng):
                indt = i-j-1
                if indt<0:
                    continue
                indx, Ni = self.Hd.evnt_t(indt)
                subx = tuple(indx.transpose())
                y[subx] += Ni*self.gn[j]
            nI = (1-I)*self.fx.convplus(y,self.sig[i],self.wG[i])
            nI[~self.fx.effarea] = 0
            F[i] = np.sum(nI)
        barF.finish()
        return F   
    
    def R0cal(self,l,b):
        r'''calculate R0 using s according to given l and b'''
        R0 = l*self.s**b
        return R0
    
    def dR0cal(self,l,b):
        r'''calculate derivative of R0 w.r.t l and b, respectively'''
        dR0l = self.s**b
        dR0b = l*np.log(self.s)*self.s**b
        return dR0l,dR0b
        
    def Fapp(self,adj,x):
        r'''apply F or its adjoint on input x:
            inputs:
                adj--adjoint boolean (Ture or False):
                     True: x--data residual (FR0-NI) (1D float array: (nt-))
                     Flase: x--R0 (1D float array: (nt-))
            output:
                adj:
                    True: y--F^T r (1D float array: (nt-))
                    False: y--FR0 (1D float array: (nt-))'''
        
        return self.F*x
    
    def iterupdate(self,l0,b0,Nl=100,Nb=100,tol=1e-3,itermax=100):
        r'''gradient-based iteration to invert for better l and b starting from initial l0 and b0:
            inputs:
                l0,b0: initial solution for l0 and b0 (float scalar)
                Nl,Nb: scaling for the intervted m, i.e., the final l=m0*Nl, b=m1*Nb (float scalar)
                tol: tolerance of convergence (float scalar)
            output:
                m0,m1: inverted model for scaled l and b (float scalar)'''
        #predetermined parameters for iteration
        ss0 = 1e-4 # initial tentative step size
        nss = 10 # times of shrinking for searching appropriate tentative step size (shrink rate=2)
        #initialization
        m0 = l0/Nl
        m1 = b0/Nb
        R0 = self.R0cal(l0,b0)
        d = self.Fapp(0,R0) # forward F
        r = d-self.NI # residual
        J0 = np.sum(r**2) # obj
        #iteration
        J = np.zeros(itermax)
        for i in range(itermax):
            #**gradient calculation**#
            pdl,pdb = self.dR0cal(m0*Nl,m1*Nb) # gredients for l and b
            Fr = self.Fapp(1,r) # adjoint F
            gl = np.sum(pdl*Fr)/Nl # gradient for m0
            gb = np.sum(pdb*Fr)/Nb # gradient for m1
            #**step size optimization**#
            ss = ss0 # initialize tentative step size
            nsc = 0 # count for shrinking time
            while(nsc<nss):
                m0t = m0-gl*ss # tentative m0
                m1t = m1-gb*ss # tentative m1
                R0t = self.R0cal(m0t*Nl,m1t*Nb) # tentative R0
                rt = self.Fapp(0,R0t)-self.NI # tentative residual
                if np.sum(rt**2)<J0:
                    break
                else:
                    nsc += 1 # count for 1 more shrinking
                    ss /= 2 # shrink the tentative step size
            if nsc>=nss:
                warn(f'At {i+1}th iteration, the tentative step size--{ss} has shrinked by 2 for {nsc} times but still not achieved decreased obj!')
            dr = (r-rt)/ss # delta residual
            alp = -np.sum(dr*r)/np.sum(dr**2) # step size
            #**update model and perform forward modeling**#
            m0 += gl*alp # update m0
            m1 += gb*alp # update m1
            R0 = self.R0cal(m0*Nl,m1*Nb) # calculate new R0
            d = self.Fapp(0,R0) # forward F
            r = d-self.NI # residual
            J1 = np.sum(r**2) # obj
            if J1>J0:
                m0 -= gl*alp
                m1 -= gb*alp # back to previous model
                print(f'The inversion stopped at the {i+1}th iteration due to residual increasing!')
                break
            else:
                J[i] = J1 # record obj
                if (J0-J1)/J0 <= tol:
                    print(f'The inversion converges at the {i+1}th iteration according to relative tolerance of {tol}!')
                    break
                else:
                    J0 = J1 # update previous obj
        # obtain finally optimized k and a           
        l = m0*Nl
        b = m1*Nb
        return l,b
        
    def testlb(self,k,a):            
        r'''test given l and b against NI'''
        R0 = self.R0cal(k,a)
        d = self.Fapp(0,R0)
        
        fig,ax = plt.subplots(1,1,figsize=(8,4))
        ax.plot(self.NI,label='infected number')
        ax.plot(d,label='predicted infected number')
        ax.set_xlabel('Time (day)')
        ax.set_ylabel('$N_I$')
        ax.legend()
        
        return d   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        