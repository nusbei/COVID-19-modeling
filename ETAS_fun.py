import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy import signal

def pdegconv(a,s):
    r'''pad given 1D sequence a using edge values and convolve it with s:
        inputs:
            a--1D sequence (1D float(int) array: (na-))
            s--smoother (1D float(int) array: (ns-))
        output:
            smoothed 1D sequence (1D float array: (na-))'''
    ns = len(s)
    if ns % 2:
        pl = int((ns-1)/2) # ns must be odd
    else:
        raise ImportError('The smoother length must be odd!')
    apeg = np.pad(a,(pl,pl),'edge')
    y = np.convolve(apeg,s,'valid')
    return y

class infevnt:
    def __init__(self):
        r'''intialize an infection event'''
        self.indx = np.zeros((0,2),dtype=np.int16)
        self.indt = np.zeros(0,dtype=np.int16)
        self.Ni = np.zeros(0,dtype=np.int16)
        self.ne = 0
        
    def evnt_add(self,indx,indt,Ni):
        r'''add extra events into current history:
            inputs:
                indx--infection event position grid indices (2D int array: (ne-by-2))
                indt--infection event time (1D int array: (ne-))
                Ni--infection number of this event (1D int array: (ne-))'''
        self.indx = np.concatenate((self.indx,indx),axis=0)
        self.indt = np.concatenate((self.indt,indt),axis=0)
        self.Ni = np.concatenate((self.Ni,Ni),axis=0)
        self.ne += len(Ni)
        
    def evnt_del(self,Hd_d):
        r'''reduce events from current history:
            inputs:
                Hd_d--event histroy being deleted (infevnt class)'''
        mask = np.zeros(self.ne,dtype=bool)
        for i in range(Hd_d.ne):
            mask += (Hd_d.indx[i,0]==self.indx[:,0])\
            *(Hd_d.indx[i,1]==self.indx[:,1])\
            *(Hd_d.indt[i]==self.indt)\
            *(Hd_d.Ni[i]==self.Ni)
        self.indx = self.indx[~mask]
        self.indt = self.indt[~mask]
        self.Ni = self.Ni[~mask]
        self.ne = len(self.Ni)

    def evnt_t(self,indt):
        r'''find infection events at time indt:
            input:
                indt--time of the infection event (int scalar)'''
        mask = self.indt==indt
        return self.indx[mask], self.Ni[mask]
    
    def NIcal(self,tr):
        r'''calculate total infected number along time:
            input:
                tr--time indices (1D int array: (nt-))'''
        nt = len(tr)
        NI = np.zeros(nt)
        for i in range(nt):
            _, Ni = self.evnt_t(tr[i])
            NI[i] = np.sum(Ni)
        return NI
    
    def Ical(self,dmg,t):
        r'''calculate the immunity ratio at time t:
            input:
                dmg--population density (2D float array: (ny-by-nx))
                t--time index (int scalar)
                '''
        I = np.zeros_like(dmg)
        for i in range(t):
            indx, Ni = self.evnt_t(i)
            if len(Ni) > 0:
                sub = tuple(indx.transpose())
                I[sub] += Ni/dmg[sub]
        return I
    
    def LdHdNi_cal(self,nt):
        r'''calculate Ld and HdNi up to time t:
            inputs:
                nt--time index (int scalar)
            outputs:
                Ld--infection event number at each day (1D int array: (nt-))
                HdNi--infected number for each event, aligned from day 0 to t (1D int array: (Nt-))'''
        Ld = np.zeros(nt,dtype=np.int16)
        HdNi = np.zeros(0,dtype=np.int16)
        for i in range(nt):
            _, Ni = self.evnt_t(i)
            Ld[i] = len(Ni)
            if Ld[i] != 0:
                HdNi = np.concatenate((HdNi,Ni))
        return Ld, HdNi

def gd(t,tau,sig,cf=None,norm=True):
    r'''create temporal function for the infectability of an infecious disease:
        inputs:
            t--time sequence (1D float array: (nt-))
            tau--time of peak symptoms (float scalar)
            sig--standard deviation of the infectability distribution along time (float scalar)
            cf--cutoff time for the infectability function, due to quarantine or hospitalization (float scalar)
        output:
            y--infectability time sequence (1D float array: (nt-))'''
    y = np.exp(-0.5*((t-tau)/sig)**2)
    if norm:
        y = y/np.sum(y)
    if cf is not None:
        y = y[t<=cf]
    return y    
    
class fxxi:
    r'''spatial intensity distribution: Gaussian plus stationary background (proportional to log scale population distribution)'''
    def __init__(self, shape, effarea=None, bkd=None):
        r'''shape--structure modeling area (int tuple: (2-))
            effarea--effective area mask (2D boolean array, (ny-by-nx))
            bkd--background distribution (2D float array, (ny-by-nx)'''
        self.shape = shape
        if effarea is None:
            self.effarea = np.ones(shape, dtype=bool) # default effarea is the total structure area
        else:
            self.effarea = effarea # effarea is the given mask
        # global background
        if bkd is None:
            self.bkd = np.zeros(shape)
            self.bkd[self.effarea] = 1/np.sum(effarea) # default bkd is homogeneous
        else:
            self.bkd = bkd/np.sum(bkd[self.effarea])
            self.bkd[~effarea] = 0 # bkd is the given bkd normalized within the effarea
        
    def _gaussfun(self,sig):
        r'''Gaussian possibility function:
            input:
                sig--standard deviation of the Gaussian possiblity function (float scalar)
            output:
                local Gaussian distribution (2D float array: (ngy-by-ngx))'''
        # the size of the gaussian filter is ny-by-ny
        ny = int(2*np.around(3*sig)+1)
        nx = ny
        G = np.zeros((ny,nx))
        # generate grid and center
        [Y, X] = np.meshgrid(range(ny),range(nx), indexing='ij')
        xc = (ny/2,nx/2)
        r2 = (X-xc[1])**2+(Y-xc[0])**2
        G = 1/(sig**2*2*pi)*np.exp(-0.5*r2/sig**2)
        G = G/np.sum(G)
        return G
    
    def convplus(self, gNi_map, sig, wG):
        r'''caculate the spatial intensity distribution according to given infection map:
            input:
                gNi_map--total infection number for all infection events that have influence upto time t (2D int array: (ny-by-nx))
                wG--Gassian distribution weight (float scalar)
            output:
                expected infection intensity (2D float array: (ny-by-nx)) '''
        gx = self._gaussfun(sig)
        y = wG*signal.convolve2d(gNi_map, gx, mode='same') # local Gaussian
        y += (1-wG)*np.sum(gNi_map)*self.bkd # global background
        y[~self.effarea] = 0 # mute outside the valid area
        return y

class ifds_ETAS:
    r'''create infectious disease spreading model and predict future infected distribution based on ETAS'''
    def __init__(self,dmg,gn,N0,k,a,fx,Hd=None):
        r'''initialization:
            inputs:
                dmg--population distribution (2D float array: (ny-by-nx))
                gn--normalized infectability time sequence (1D float array: (ng-))
                ng--length of gn (int scalar)
                N0--infected number threshold for single infection event (int scalar)
                k,a--hd parameters (float scalar)
                fx--spatial infection PDF (fxxi class)'''
        self.dmg = dmg
        self.gn = gn
        self.ng = len(gn)
        self.N0 = N0
        self.k = k
        self.a = a
        self.fx = fx
        self.size = fx.bkd.size
        if Hd is None:
            self.Hd = infevnt()
        else:
            self.Hd = Hd
    
    def LdnI_cal(self,t,R0,I,sig,wG):
        r'''According to infection event history and current R0, sig, wG, as well as fxxi, predict next day's infection:
            inputs:
                t--current day index (int scalar)
                R0--current reproduction number (int scalar)
                sig--current short-distance human movement range (float scalar)
                wG--current weight for short-distance human movement within total movement (including short-distance and long-distance)
            output:
                nI--next-day expected infection distribution (2D float array: (ny-by-nx))
                Ld--infection event number at day t (int scalar)'''
        y = np.zeros(self.fx.shape)
        Ld = 0
        for i in range(self.ng):
            indt = t-i-1
            if indt<0:
                continue
            indx, Ni = self.Hd.evnt_t(indt)
            subx = tuple(indx.transpose())
            y[subx] += Ni*self.gn[i]
            Ld += np.sum(self.k*(Ni/self.N0)**self.a*self.gn[i])
        if np.all(y==0):
            nI = y
        else:
            nI = R0*(1-I)*self.fx.convplus(y,sig,wG)
        return Ld,nI

    def formod(self,R0t,sigt,wGt,t0=0,I0=None,Hdi=None):
        r'''modeling infecious disease spreading from day t0, according to given parameters:
            inputs:
                R0--basic reproduction number for the modeled nt days (float scalar: (nt-))
                sigt--Gassian distribution (short-distance human movement) parameter varying within nt days (1D float array: (nt-))
                wGt--Gassian distribution (short-distance human movement) weight varying within nt days (1D float array: (nt-))
                t0--starting day index (int scalar)
                I0--immunity at the starting day t0 (2D float array: (ny-by-nx))
                Hdi--imported infection events (infevnt class)
            output:
                self.Hd--updated community infection events (infevnt class)
                nIt--expected infected distributions of modeled nt days (nt-by-ny-by-nx)'''
        
        ################################
        # some random perturbation on predicted Ld and Ni
        ptbN = np.random.normal(scale=0.2,size=100000)
        ptbN[ptbN<-0.95] = -0.95
        ptbN[ptbN>0.95] = 0.95
        c = 0
        ################################
        
        nt = len(R0t)
        nIt = np.zeros((nt,self.fx.shape[0],self.fx.shape[1]))
        if I0 is None:
            I0 = np.zeros(self.fx.shape)
        I = I0
        if Hdi is None:
            Hdi = infevnt()
        # step forward from t0
        for i in range(nt):
            print(f'{i}/{nt}')
            t = i+t0
            # calculate susceptible population
            Ns = self.dmg*(1-I)
            # import infection events
            indx, Ni = Hdi.evnt_t(t)
            indt = np.zeros_like(Ni)+t
            self.Hd.evnt_add(indx,indt,Ni)
            # calculate nI and Ld
            Ld, nI = self.LdnI_cal(t,R0t[i],I,sigt[i],wGt[i])
            ### perturb and int Ld ###
            Ld = int(Ld*(1+ptbN[c]))
            c += 1
            ### perturb and int Ld ###
            nIt[i] = nI
            if Ld != 0:
                nIf = nI.flatten()
                NI = np.sum(nIf)
                # sampling Ld positions    
                ind = np.random.choice(self.size,size=Ld,replace=False,p=nIf/NI)
                sub = np.unravel_index(ind,self.fx.shape)
                indx = np.array(sub).transpose()
                # add new infection events into Hd
                indt = np.zeros(Ld,dtype=np.int16)+t
                Ni = NI*nI[sub]/np.sum(nI[sub])
                ### perturb and int Ni ###
                Ni = np.array(Ni*(1+ptbN[c:c+Ld]),dtype=np.int16)
                c += Ld
                ### perturb and int Ni ###
                Nss = np.array(Ns[sub],dtype=np.int16)
                Ni[Ni>Nss] = Nss[Ni>Nss]
                self.Hd.evnt_add(indx,indt,Ni)
                # update I
                I[sub] += Ni/self.dmg[sub]
                I[I>1] = 1
        return nIt
            
class dispift:
    r'''display infection events and expected infection distribution'''
    def __init__(self,fx,dpi=80,outpath=None):
        r'''input:
                fx--fxxi class'''
        self.bkd = fx.bkd
        self.bkd[~fx.effarea] = float("nan")
        self.shape = fx.shape
        [Y,X] = np.meshgrid(range(self.shape[0]),range(self.shape[1]), indexing='ij')
        self.Y = Y
        self.X = X
        self.dpi = dpi
        self.path = outpath
        
    def setscale(self,ax):
        r'''plot the scale indicator on ax'''
        x1 = 300
        x2 = 380
        x3 = 330
        y1 = 220
        y2 = 215
        y3 = 210
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([x1,x2],[y1,y1],'r',linewidth=3,zorder=100)
        ax.plot([x1,x1],[y1,y2],'r',linewidth=3,zorder=100)
        ax.plot([x2,x2],[y1,y2],'r',linewidth=3,zorder=100)
        _ = ax.text(x3,y3,'10 km',color='r',fontsize=15,zorder=100)
        
    def setinform(self,ax,t,Ld,NI):
        x0 = 20
        y0 = 20
        x1 = 360
        y1 = 40
        ax.text(x0,y0,f'Day: {t}',color='k',fontsize=15,zorder=100)
        ax.text(x1,y0,f'$\lambda_d={Ld}$',color='k',fontsize=15,zorder=100)
        ax.text(x1,y1,f'$N_I={NI}$',color='k',fontsize=15,zorder=100)
        
    def dispevnt_history(self,t,Hdc,Hdi=None):
        r'''display the infection events along time from day 0 to day t:
            inputs:
                t: last day being plot (int scalar)
                Hdc: community infection event history (infevnt class)
                Hdi: imported infection event history (infevnt class)'''
        yc = np.zeros(self.shape)
        yi = np.zeros(self.shape)
        for i in range(t):
            # plot background
            fig, ax = plt.subplots(1,1,figsize=(20,9),frameon=True)
            ax.imshow(self.bkd,cmap='gray',zorder=10)
            ne = 0
            NI = 0
            indx, Ni = Hdc.evnt_t(i)
            subx = tuple(indx.transpose())
            yc[subx] += Ni
            mask = yc!=0
            ne += len(Ni)
            NI += np.sum(Ni)
            ax.scatter(self.X[mask],self.Y[mask],s=yc[mask], alpha=0.7, c='blue', zorder=30)
            if Hdi is not None:
                indx, Ni = Hdi.evnt_t(i)
                subx = tuple(indx.transpose())
                yi[subx] += Ni
                mask = yi!=0
                ne += len(Ni)
                NI += np.sum(Ni)
                ax.scatter(self.X[mask],self.Y[mask],s=yi[mask], alpha=0.7, c='orange', zorder=40)    
            self.setscale(ax)
            self.setinform(ax,i,ne,NI)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(4)
            plt.show()
            if self.path is not None:
                BB = ax.get_position()
                BB.x0 = 4.2
                BB.x1 = 16.3
                BB.y0 = 1
                BB.y1 = 8
                fig.savefig(f'{self.path}/evnt_history_day_0-{i}.png',dpi=self.dpi,bbox_inches=BB)
            
    
    def dispevnt_current(self,t,Hdc,Hdi=None):
        r'''display the infection events at day t:
            inputs:
                t: the day being plot (int scalar)
                Hdc: community infection event history: (infevnt class)
                Hdi: imported infection event history: (infevnt class)'''
        yc = np.zeros(self.shape)
        yi = np.zeros(self.shape)
        ne = 0
        nI = 0
        indx, Ni = Hdc.evnt_t(t)
        subx = tuple(indx.transpose())
        yc[subx] = Ni
        mask = yc!=0
        ne += len(Ni)
        NI += np.sum(Ni)
        # plot background
        fig, ax = plt.subplots(1,1,figsize=(20,9),frameon=True)
        ax.imshow(self.bkd,cmap='gray',zorder=10)
        ax.scatter(self.X[mask],self.Y[mask],s=yc[mask], alpha=0.7, c='blue', zorder=30)
        if Hdi is not None:
            indx, Ni = Hdi.evnt_t(t)
            subx = tuple(indx.transpose())
            yi[subx] = Ni
            mask = yi!=0
            ne += len(Ni)
            NI += np.sum(Ni)
            ax.scatter(self.X[mask],self.Y[mask],s=yi[mask], alpha=0.7, c='orange', zorder=40)
        self.setscale(ax)
        self.setinform(ax,t,ne,NI)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(4)
        plt.show()
        if self.path is not None:
            BB = ax.get_position()
            BB.x0 = 4.2
            BB.x1 = 16.3
            BB.y0 = 1
            BB.y1 = 8
            fig.savefig(f'{self.path}/infevnt_day_{t}.png',dpi=self.dpi,bbox_inches=BB)
        
    def dispnI(self,t,nI,indx,Ni):
        r'''display the infection events and the expected infected distribution at day t:
            inputs:
                t--day index (int scalar)
                nI--expected infected distribution at day t (2D float array: (ny-by-nx))
                indx--infection event positions at day t (2D int array: (ne-by-2))
                Ni--infected number of infection events at day t (1D int array: (ne-))'''
        y = np.zeros(self.shape)
        subx = tuple(indx.transpose())
        y[subx] = Ni
        mask = y!=0
        NI = np.sum(nI)
        if NI==0:
            nI[0,0] = 1
            NI = 1
        # plot background
        fig, ax = plt.subplots(1,1,figsize=(20,9),frameon=True)
        ax.imshow(self.bkd,cmap='gray',zorder=10)
        ax.imshow(-nI/NI,cmap='hot',alpha=0.9,zorder=20)
        #ax.scatter(self.X[mask],self.Y[mask],s=y[mask], alpha=0.7, c='green', zorder=30)
        self.setscale(ax)
        self.setinform(ax,t,len(Ni),np.sum(Ni))
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        if self.path is not None:
            BB = ax.get_position()
            BB.x0 = 4.2
            BB.x1 = 16.3
            BB.y0 = 1
            BB.y1 = 8
            fig.savefig(f'{self.path}/expinfdis_day_{t}.png',dpi=self.dpi,bbox_inches=BB)
        plt.show()


