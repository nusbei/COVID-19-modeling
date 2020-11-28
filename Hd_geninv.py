import numpy as np
from warnings import warn
from ETAS_fun import infevnt
import matplotlib.pyplot as plt

def nipdf(Nmean,Nsig,Nmax):
    r'''Define the pdf for infected number of each infection event:
        inputs:
            Nmean--mean of the Gaussian distribution (int scalar)
            Nsig--sigma of the Gaussian distribution (int scalar)
            Nmax--maximum cutoff of the Gaussian distribution (int scalar)
        output:
            y--pdf of the each positive integer smaller than Nmax (1D float array: (Nmax-))'''
    x = np.arange(Nmax)+1
    y = np.exp(-0.5*((x-Nmean)/Nsig)**2)
    
    return y
    

def rdHdgen(NI,dmg,Hd0=None):
    r'''generate random infection events with toal infected number being consistent with given NI
        input:
            NI--total infected number along time (1D int array: (nt-))
            dmg--population density (ny-by-nx)
        output:
            evt--infevnt class'''
    Nmax = 10
    pni = nipdf(4,2,10)
    nt = len(NI) # total number of days
    Np = dmg.size # total element number
    # background PDF
    Pn = np.array(dmg)
    Pn[Pn<1] = 1
    Pn = np.log(Pn)
    Ns = np.array(dmg) # susceptable populatin
    if Hd0 is None:
        Hd0 = infevnt() # infection events with priority (i.e., already assigned)
    Hd = infevnt() # objective infection events
    # generate event number and infected number for each event
    for i in range(nt):
        # check prior infection events
        ind0, Ni0 = Hd0.evnt_t(i)
        sub0 = tuple(ind0.transpose())
        if len(Ni0)!=0:
            Ns[sub0] -= Ni0 # decrease susceptable population
        Ni = NI[i] # infected number at day i
        if Ni == 0:
            continue
        if Ni == 1:
            # only one infection event with one infected
            # sample position for this one infected
            P = np.array(Pn) # initialize background PDF
            mask = Ns<1
            P[mask] = 0 # eleminate positions without enough susceptable population
            P = P*Ns/dmg # scale P with susceptability
            P = P.flatten() # flatten P
            P[np.isnan(P)] = 0 # replace nan with 0
            ind = np.random.choice(Np,size=1,replace=False,p=P/np.sum(P)) # sample one position
            sub = np.unravel_index(ind,Pn.shape) # transform to sub
            indx = np.array(sub).transpose() # compose 2D spatial indx array
            indt = np.array([i]) # compose 1D time array
            Hd.evnt_add(indx,indt,np.array([1])) # addin infection event
            Ns[sub] -= 1 # reduce susceptable population
        else:
            #Nmax = 10#max(3,np.ceil(50*(Ni/50)**(1/4)/5))
            y = 0
            while(y < Ni):
                Nu = min(Nmax,Ni-y)
                if Nu == 1:
                    x = 1
                else:
                    Pu = pni[:Nu]
                    x = np.random.choice(Nu,size=1,replace=False,p=Pu/np.sum(Pu)) # sample one infected number
                    x = x[0]+1
                y += x
                # sample position for this one infected
                P = np.array(Pn) # initialize background PDF
                mask = Ns<x
                P[mask] = 0 # eleminate positions without enough susceptable population
                P = P*Ns/dmg # scale P with susceptability
                P = P.flatten() # flatten P
                P[np.isnan(P)] = 0 # replace nan with 0
                ind = np.random.choice(Np,size=1,replace=False,p=P/np.sum(P)) # sample one position
                sub = np.unravel_index(ind,Pn.shape) # transform to sub
                indx = np.array(sub).transpose() # compose 2D spatial indx array
                indt = np.array([i]) # compose 1D time array
                Hd.evnt_add(indx,indt,np.array([x]))
                Ns[sub] -= x
    # return Hd
    return Hd
        
class invka:
    r'''invert for k_d and alpha_d according to given Ld and HdNi'''
    def __init__(self,gd,Ld,HdNi,Ldy,N0=1):
        r'''initialization'''
        self.gd = gd # temporal function for the infectability of an infecious disease (1D float array: (ng-))
        self.ng = len(self.gd) # length of gd (int scalar)
        self.Ld = Ld # infection event number along time (1D int array: (nt-))
        self.cLd = np.cumsum(Ld) # cumsum of Ld (1D int array: (nt-))
        self.HdNi = HdNi # infected number for each event (1D int array: (Nt-))
        self.Nt = len(HdNi) # total number of infection events (int scalar)
        self.nt = len(Ld) # time sample number (int scalar)
        self.N0 = N0 # infection number threshold that being considered as an infection event (int saclar)
        self.Ldy = Ldy # fitting objective infection event number (1D int array: (nt-))
        self.Hdn = self.HdNi/self.N0 # normalized infection event history (1D int array: (Nt-))
    
    def hdcal(self,k,a):
        r'''calculate spatial influence according to infected number:
            inputs:
                k,a--parameters for the ETAS model (float scalar)
            output:
                hdt--spatial magnitude influence for each infection event (1D float array: (Nt-))'''
        hdt = k*self.Hdn**a
        return hdt
    
    def dhdcal(self,k,a):
        r'''calculate partial derivatives of hdt w.r.t k and a respectively:
            output:
                pdk--hdt's derivative w.r.t k (1D float array: (Nt-))
                pda--hdt's derivative w.r.t a (1D float array: (Nt-))'''
        pdk = self.Hdn**a
        pda = k*self.Hdn**a*np.log(self.Hdn)
        return pdk,pda
    
    def Gcal(self,adj,x):
        r'''convolution of gd or its adjoint on input x:
            inputs:
                adj--adjoint boolean (Ture or False):
                     True: x--data residual (GH0-Ld) (1D float array: (nt-))
                     Flase: x--hdt (1D float array: (Nt-))
            output:
                adj:
                    True: y--gradient with the same size as hdt (1D float array: (Nt-))
                    False: y--predicted infection event number along time (1D float array: (nt-))'''
        if adj:
            y = np.zeros(self.Nt)
        else:
            y = np.zeros(self.nt)
        for i in range(self.nt):
            for j in range(self.ng):
                I = i-j-1
                if I<0:
                    continue
                id0 = self.cLd[I]-self.Ld[I]
                for k in range(self.Ld[I]):
                    if adj:
                        y[id0+k] += self.gd[j]*x[i]
                    else:
                        y[i] += self.gd[j]*x[id0+k]
                        
        return y
    
    
            
    def iterupdate(self,k0,a0,Nk=1,Na=1,tol=1e-3,itermax=100):
        r'''gradient-based iteration to invert for better k and a starting from initial k0 and a0:
            inputs:
                k0,a0: initial solution for k0 and a0 (float scalar)
                Nk,Na: scaling for the intervted m, i.e., the final k=m0*Nk, a=m1*Na (float scalar)
                tol: tolerance of convergence (float scalar)
            output:
                m0,m1: inverted model for scaled k and a (float scalar)'''
        #predetermined parameters for iteration
        ss0 = 1e-4 # initial tentative step size
        nss = 10 # times of shrinking for searching appropriate tentative step size (shrink rate=2)
        #initialization
        m0 = k0/Nk
        m1 = a0/Na
        hdt = self.hdcal(k0,a0)
        d = self.Gcal(0,hdt) # forward G
        r = d-self.Ldy # residual
        J0 = np.sum(r**2) # obj
        #iteration
        J = np.zeros(itermax)
        for i in range(itermax):
            #**gradient calculation**#
            pdk,pda = self.dhdcal(m0*Nk,m1*Na) # gredients for k and a
            Gr = self.Gcal(1,r) # adjoint G
            gk = np.sum(pdk*Gr)*Nk # gradient for m0
            ga = np.sum(pda*Gr)*Na # gradient for m1
            #**step size optimization**#
            ss = ss0 # initialize tentative step size
            nsc = 0 # count for shrinking time
            while(nsc<nss):
                m0t = m0-gk*ss # tentative m0
                m1t = m1-ga*ss # tentative m1
                hdtt = self.hdcal(m0t*Nk,m1t*Na) # tentative hdt
                rt = self.Gcal(0,hdtt)-self.Ldy # tentative residual
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
            m0 += gk*alp # update m0
            m1 += ga*alp # update m1
            hdt = self.hdcal(m0*Nk,m1*Na) # calculate new hdt
            d = self.Gcal(0,hdt) # forward G
            r = d-self.Ldy # residual
            J1 = np.sum(r**2) # obj
            if J1>J0:
                m0 -= gk*alp
                m1 -= ga*alp # back to previous model
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
        k = m0*Nk
        a = m1*Na
        return k,a
                    
    def testka(self,k,a):            
        r'''test given k and a against Ldy'''
        hdt = self.hdcal(k,a)
        Ldt = self.Gcal(0,hdt)
        
        fig,ax = plt.subplots(1,1,figsize=(8,4))
        ax.plot(self.Ldy,label='infection event number')
        ax.plot(Ldt,label='predicted infection event number')
        ax.set_xlabel('Time (day)')
        ax.set_ylabel('$\lambda_d$')
        ax.legend()
        
        return Ldt
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
