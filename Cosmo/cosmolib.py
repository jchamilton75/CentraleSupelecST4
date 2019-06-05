import scipy.integrate
import numpy as np
from matplotlib import *
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter1d
from scipy import integrate
from scipy import interpolate
from scipy import ndimage
import iminuit


###############################################################################
################################### Fitting ###################################
###############################################################################
### Generic polynomial function ##########
def thepolynomial(x,pars):
    f=np.poly1d(pars)
    return(f(x))
    
### Generic fitting function ##############
def dothefit(x,y,covarin,guess,functname=thepolynomial,method='minuit'):
    if method == 'minuit':
        print('Fitting with Minuit')
        return(do_minuit(x,y,covarin,guess,functname))
    else:
        print('method must be among: minuit')
        return(0,0,0,0)

### Class defining the minimizer and the data
class MyChi2:
    def __init__(self,xin,yin,covarin,functname):
        self.x=xin
        self.y=yin
        self.covar=covarin
        self.invcov=np.linalg.inv(covarin)
        self.functname=functname
            
    def __call__(self,*pars):
        val=self.functname(self.x,pars)
        chi2=np.dot(np.dot(self.y-val,self.invcov),self.y-val)
        return(chi2)
        
### Call Minuit
def do_minuit(x,y,covarin,guess,functname=thepolynomial, fixpars = False, hesse=False):
    # check if covariance or error bars were given
    covar=covarin
    if np.size(np.shape(covarin)) == 1:
        err=covarin
        covar=np.zeros((np.size(err),np.size(err)))
        covar[np.arange(np.size(err)),np.arange(np.size(err))]=err**2
                                    
    # instantiate minimizer
    chi2=MyChi2(x,y,covar,functname)
    # variables
    ndim=np.size(guess)
    parnames=[]
    for i in range(ndim): parnames.append('c'+np.str(i))
    # initial guess
    theguess=dict(zip(parnames,guess))
    # fixed parameters
    dfix = {}
    if fixpars:
        for i in range(len(parnames)): dfix['fix_'+parnames[i]]=fixpars[i]
    else:
        for i in range(len(parnames)): dfix['fix_'+parnames[i]]=False
    #stop
    # Run Minuit
    print('Fitting with Minuit')
    theargs = dict(theguess.items())
    theargs.update(dfix.items())
    if theargs is None:
        m = iminuit.Minuit(chi2,forced_parameters=parnames,errordef=1.)
    else:
        m = iminuit.Minuit(chi2,forced_parameters=parnames,errordef=1.,**theargs)
    m.migrad()
    if hesse: m.hesse()
    # build np.array output
    parfit=[]
    for i in parnames: parfit.append(m.values[i])
    errfit=[]
    for i in parnames: errfit.append(m.errors[i])
    ndimfit = int(np.sqrt(len(m.covariance)))
    covariance=np.zeros((ndimfit,ndimfit))
    if fixpars:
        parnamesfit = []
        for i in range(len(parnames)):
            if fixpars[i] == False: parnamesfit.append(parnames[i])
            if fixpars[i] == True: errfit[i]=0
    else:
        parnamesfit = parnames
    for i in range(ndimfit):
        for j in range(ndimfit):
            covariance[i,j]=m.covariance[(parnamesfit[i],parnamesfit[j])]

    print('Chi2=',chi2(*parfit))
    print('ndf=',np.size(x)-ndim)
    return(m,np.array(parfit), np.array(errfit), np.array(covariance))

###############################################################################
###############################################################################





###############################################################################
############################# Cosmology Functions #############################
###############################################################################
def e_z(z,cosmo):
    omegam=cosmo['omega_M_0']
    omegax=cosmo['omega_lambda_0']
    w0=cosmo['w0']
    h=cosmo['h']
    omegak=1.-omegam-omegax
    omegaxz=omegax*(1+z)**(3+3*w0)
    e_z=np.sqrt(omegak*(1+z)**2+omegaxz+omegam*(1+z)**3)
    return(e_z)

def inv_e_z(z,cosmo):
    return(1./e_z(z,cosmo))

def hz(z,cosmo):
    return(cosmo['h']*e_z(z,cosmo))

### Proper distance in Mpc
def propdist(z,cosmo,zres=0.001,accurate=False):
    ### z range for integration
    zmax=np.max(z)
    if zmax < zres:
        nb=101
    else:
        nb=zmax/zres+1
    zvals=np.linspace(0.,zmax,nb)
    ### integrate
    cumulative=np.zeros(int(nb))
    cumulative[1:]=scipy.integrate.cumtrapz(1./e_z(zvals,cosmo),zvals)
    ### interpolation to input z values
    propdist=np.interp(z,zvals,cumulative)
    ### curvature
    omega=cosmo['omega_M_0']+cosmo['omega_lambda_0']
    k=np.abs(1-omega)
    if omega == 1:
        propdist=propdist
    elif omega < 1:
        propdist=np.sinh(np.sqrt(k)*propdist)/np.sqrt(k)
    elif omega > 1:
        propdist=np.sin(np.sqrt(k)*propdist)/np.sqrt(k)
    ### returning
    return(propdist*2.99792458e5/100/cosmo['h'])

### Luminosity distance in Mpc
def lumdist(z,cosmo,zres=0.001,accurate=False):
    return(propdist(z,cosmo,zres=zres,accurate=accurate)*(1+z))

### Angular distance in Mpc
def angdist(z,cosmo,zres=0.001,accurate=False):
    return(propdist(z,cosmo,zres=zres,accurate=accurate)/(1+z))

### SNIa distance modulus
def musn1a(z, cosmo):
    dlum = lumdist(z, cosmo)*1e6
    return(5*np.log10(dlum)-5+5*np.log10(cosmo['h']/0.7))
### Age
def lookback(z,cosmo,zres=0.001):
    ### z range for integration
    zmax=np.max(z)
    if zmax < zres:
        nb=101
    else:
        nb=zmax/zres+1
    zvals=np.linspace(0.,zmax,nb)
    ### integrate
    cumulative=np.zeros(int(nb))
    cumulative[1:]=scipy.integrate.cumtrapz(1./e_z(zvals,cosmo)/(1+zvals),zvals)
    ### interpolation to input z values
    age=np.interp(z,zvals,cumulative)
    ### Age in Gyr
    return age/100/cosmo['h'] * (3.26 * 1e6 * 365 * 24 * 3600 *3e5) / (365 * 24 * 3600) / 1e9

### Eisenstein & Hu 1998 modelization of decoupling, sound horizon and so on... avoids calling CAMB but is not very accurate...
def rs(cosmo,zd=1059.25):
    o0=cosmo['omega_M_0']-cosmo['omega_n_0']     ### need to remove omega_neutrino as they are relativistic (massless at high z)
    h=cosmo['h']
    ob=cosmo['omega_b_0']
    theta=2.7255/2.7
    zeq=2.5*1e4*o0*h**2*theta**(-4)
    keq=7.46*0.01*o0*h**2*theta**(-2)
    b1=0.313*(o0*h**2)**(-0.419)*(1+0.607*(o0*h**2)**0.674)
    b2=0.238*(o0*h**2)**0.223
    # This is E&H zdrag
    #zd=1291.*(o0*h**2)**0.251/(1+0.659*(o0*h**2)**0.828)*(1+b1*(ob*h**2)**b2)
    # We use instead the value coming from CAMB and for Planck+WP+Highl Cosmology as suggested by J.Rich (it depends mostly on atomic physics) => zd=1059.25
    req=RR(zeq,ob,h,theta)
    rd=RR(zd*1.,ob,h,theta)
    rs=(2./(3*keq))*np.sqrt(6./req)*np.log((np.sqrt(1+rd)+np.sqrt(rd+req))/(1+np.sqrt(req)))
    return(rs)

def RR(z,ob,h,theta):
    return(31.492*ob*h**2*theta**(-4)*((1+z)/1000)**(-1))

def thetastar(cosmo,zstar=1090.49):
        rsval=rs(cosmo,zd=zstar)
        da=angdist(zstar,cosmo,zres=0.001)
        return rsval/(1+zstar)/da
        
###############################################################################
###############################################################################


###############################################################################
########################## Miscellaneous Functions ############################
###############################################################################
def progress_bar(i,n):
    if n != 1:
        ntot=50
        ndone=ntot*i/(n-1)
        a='\r|'
        for k in range(ndone):
            a += '#'
        for k in range(ntot-ndone):
            a += ' '
        a += '| '+str(int(i*100./(n-1)))+'%'
        sys.stdout.write(a)
        sys.stdout.flush()
        if i == n-1:
            sys.stdout.write(' Done \n')
            sys.stdout.flush()
###############################################################################
###############################################################################
            






