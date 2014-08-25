import numpy as np
import pylab as plt
import matplotlib as mpl
from scipy.stats import scoreatpercentile as sap
from scipy import stats
import pickle,os

__all__ = ['getColors','errorbar','pystanErrorbar',
           'saveStanFit','loadStanFit','printCI',
           'figure','subplot','subplot_annotate',
           'hist','histCI','plotCIttest1','plotCIttest2']
CLR=(0.2, 0.5, 0.6)
# size of figure columns
FIGCOL=[3.27,4.86,6.83] # plosone
FIGCOL=[3.3,5, 7.1] # frontiers
# TODO custom ppl style histogram
def getColors(N):
    ''' creates set of colors for plotting

        >>> len(getColors(5))
        5
        >>> getColors(5)[4]
        (0.69411766529083252, 0.3490196168422699, 0.15686275064945221, 1.0)
        >>> N=14
        >>> c=getColors(N)
        >>> plt.scatter(range(N),range(N),color=c)
        >>> plt.show()
    '''
    clrs=[]
    cm = plt.get_cmap('Paired')
    for i in range(N+1):
        clrs.append(cm(1.*i/float(N))) 
        #plt.plot(i,i,'x',color=clrs[i])
    clrs.pop(-2)
    return clrs

def imshow(*args,**kwargs):
    plt.imshow(*args,**kwargs)

#plt.ion()
#imshow(np.array([[1,2,3],[2,1,2],[3,1,2]]))
def figure(**kwargs):
    ''' wrapper around matplotlib.figure
        additionally supports following kwargs
        size - 1,2 or 3 respectively for small, medium, large width
        aspect - [0,inf] height to width ratio 
    '''
    if not kwargs.has_key('figsize'):
        if kwargs.has_key('size'): w= FIGCOL[kwargs.pop('size')-1]
        else: w=FIGCOL[0]
        if kwargs.has_key('aspect'): h=kwargs.pop('aspect')*w
        else: h=w
        kwargs['figsize']=(w,h)
    plt.figure(**kwargs)
    ax=plt.gca()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    plt.grid(False,axis='x')
    plt.grid(True,axis='y')

def hist(*args,**kwargs):
    '''
        >>> dat=np.random.randn(1000)*10+5
        >>> x=np.linspace(-30,30,60)
        >>> hist(dat,bins=x)
    '''
    if not kwargs.has_key('facecolor'): kwargs['facecolor']=CLR
    if not kwargs.has_key('edgecolor'): kwargs['edgecolor']='w'
    plt.hist(*args,**kwargs)

def histCI(*args,**kwargs):
    '''
        >>> x=np.random.randn(1000)
        >>> bn=np.linspace(-3,3,41)
        >>> histCI(x,bins=bn)
    '''
    if kwargs.has_key('plot'): plot=kwargs.pop('plot')
    else: plot=True
    if kwargs.has_key('alpha'): alpha=kwargs.pop('alpha')
    else: alpha=0.05
    a,b=np.histogram(*args,**kwargs)
    m=b.size
    n=args[0].shape[0]
    c=stats.norm.ppf(alpha/(2.*m))/2.*(m/float(n))**0.5
    l=np.square(np.maximum(np.sqrt(a)-c,0))
    u=np.square(np.sqrt(a)+c)
    if plot: plothistCI(a,b,l,u)
    return a,b,l,u

def plothistCI(a,b,l,u):
    '''
        >>> x=np.random.randn(1000)
        >>> bn=np.linspace(-3,3,41)
        >>> a,b,l,u=histCI(x,bins=bn)
        >>> plothistCI(a,b,l,u)
    '''
    b=b[:-1]+np.diff(b)/2.
    plt.plot(b,a,color=CLR)
    x=np.concatenate([b,b[::-1]])
    ci=np.concatenate([u,l[::-1]])
    plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                alpha=0.2,fill=True,fc='red',ec='red'))


def subplot(*args):
    plt.subplot(*args)
    ax=plt.gca()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    plt.grid(False,axis='x')
    plt.grid(True,axis='y')
def subplot_annotate(loc='nw',nr=None):
    if type(loc) is list and len(loc)==2: ofs=loc
    elif loc is 'nw': ofs=[0.1,0.9]
    elif loc is 'sw': ofs=[0.1,0.1]
    elif loc is 'se': ofs=[0.9,0.1]
    elif loc is 'ne': ofs=[0.9,0.9]
    ax=plt.gca()
    if nr is None: nr=ax.colNum*ax.numRows+ax.rowNum
    plt.text(plt.xlim()[0]+ofs[0]*(plt.xlim()[1]-plt.xlim()[0]),
            plt.ylim()[0]+ofs[1]*(plt.ylim()[1]-plt.ylim()[0]), 
            str(unichr(65+nr)),horizontalalignment='center',verticalalignment='center',
            fontdict={'weight':'bold'},fontsize=12)

def _errorbar(out,x,clr='k'):
    plt.plot([x,x],out[1:3],color=clr)
    plt.plot([x,x],out[3:5],
        color=clr,lw=3,solid_capstyle='round')
    plt.plot([x],[out[0]],mfc=clr,mec=clr,ms=8,marker='_',mew=2)

def plotCIttest1(y,x=0,alpha=0.05,clr='k'):
    m=y.mean();df=y.size-1
    se=y.std()/y.size**0.5
    cil=stats.t.ppf(alpha/2.,df)*se
    cii=stats.t.ppf(0.25,df)*se
    out=[m,m-cil,m+cil,m-cii,m+cii]
    _errorbar(out,x=x,clr=clr)
    return out
    
def plotCIttest2(y1,y2,x=0,alpha=0.05,clr='k'):
    n1=float(y1.size);n2=float(y2.size);
    v1=y1.var();v2=y2.var()
    m=y2.mean()-y1.mean()
    s12=(((n1-1)*v1+(n2-1)*v2)/(n1+n2-2))**0.5
    se=s12*(1/n1+1/n2)**0.5
    df= (v1/n1+v2/n2)**2 / ( (v1/n1)**2/(n1-1)+(v2/n2)**2/(n2-1)) 
    cil=stats.t.ppf(alpha/2.,df)*se
    cii=stats.t.ppf(0.25,df)*se
    out=[m,m-cil,m+cil,m-cii,m+cii]
    _errorbar(out,x=x,clr=clr)
    return out
    
def errorbar(y,clr=CLR,x=None,labels=None):
    ''' customized error bars
        y - NxM ndarray containing results of
            N simulations of M random variables
        x - array with M elements, position of the bars on x axis 
        clr - bar color
        labels - array with xtickslabels

        >>> errorbar(np.random.randn(1000,10)+1.96)    
    '''
    out=[]
    d=np.array(y);
    if x is None: x=np.arange(d.shape[1])
    elif np.array(x).ndim!=1 or np.array(x).shape[0]!=y.shape[1]:
        x=np.arange(0,y.shape[1])
    ax=plt.gca()
    for i in range(d.shape[1]):
        out.append([d[:,i].mean(),sap(d[:,i],2.5),sap(d[:,i],97.5),
                    sap(d[:,i],25),sap(d[:,i],75)])
        _errorbar(out[-1],x=x[i],clr=clr)
    ax.set_xticks(x)
    if not labels is None: ax.set_xticklabels(labels)
    plt.xlim([np.floor(x[0]-1),np.ceil(x[-1]+1)])
    return np.array(out)

def pystanErrorbar(w,keys=None):
    """ plots errorbars for variables in fit
        fit - dictionary with data extracted from Pystan.StanFit instance 
    """
    kk=0
    ss=[];sls=[]
    if keys is None: keys=w.keys()[:-1]
    for k in keys:
        d= w[k]
        if d.ndim==1:
            ss.append(d);sls.append(k)
            continue
            #d=np.array(d,ndmin=2).T
        d=np.atleast_3d(d)
        for h in range(d.shape[2]):
            kk+=1; figure(num=kk)
            #ppl.boxplot(plt.gca(),d[:,:,h],sym='')
            errorbar(d[:,:,h])
            plt.title(k)
    #ss=np.array(ss)
    for i in range(len(ss)):
        print sls[i], ss[i].mean(), 'CI [%.3f,%.3f]'%(sap(ss[i],2.5),sap(ss[i],97.5)) 
def printCI(w,var=None,decimals=3):
    def _print(b):
        d=np.round([b.mean(), sap(b,2.5),sap(b,97.5)],decimals).tolist()
        print var+' %.3f, CI %.3f, %.3f'%tuple(d) 
    if var is None: d=w;var='var'
    else: d=w[var]
    if d.ndim==2:
        for i in range(d.shape[1]):
            _print(d[:,i])
    elif d.ndim==1: _print(d)
                
def saveStanFit(fit,fname='test'):
    path = os.getcwd()+os.path.sep+'standata'+os.path.sep+fname
    w=fit.extract()
    f=open(path,'w')
    pickle.dump(w,f)
    f.close()
def loadStanFit(fname):
    path = os.getcwd()+os.path.sep+'standata'+os.path.sep+fname
    f=open(path,'r')
    out=pickle.load(f)
    f.close()
    return out

def ndsamples2latextable(data,decim=2):
    ''' data - 3D numpy.ndarray with shape (rows,columns,samples)'''
    elem='{: .%df} [{: .%df}, {: .%df}]'%(decim,decim,decim)
    ecol=' \\\\\n'
    out='\\begin{table}\n\\centering\n\\begin{tabular}{|l|'+data.shape[1]*'c|'+'}\n\\hline\n'
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out+=elem.format(data[i,j,:].mean(),
                stats.scoreatpercentile(data[i,j,:],2.5),
                stats.scoreatpercentile(data[i,j,:],97.5))
            if j<data.shape[1]-1: out+=' & '
        out+=ecol
    out+='\\hline\n\\end{tabular}\n\\end{table}'
    print out


def ndarray2gif(path,array,duration=0.1,addblank=False):
    '''
    path - file path, including filename, excluding .gif
    array - 3d numpy array, zeroth dim is the time axis,
            dtype uint8 or float in [0,1]
    duration - frame duration

    Example:
    
    >>> im = np.zeros((200,200), dtype=np.uint8)
    >>> im[10:30,:] = 100
    >>> im[:,80:120] = 255
    >>> im[-50:-40,:] = 50
    >>> images = [im*1.0, im*0.8, im*0.6, im*0.4, im*0]
    >>> images = [im, np.rot90(im,1), np.rot90(im,2), np.rot90(im,3), im*0]
    >>> ndarray2gif('test',np.array(images), duration=0.5) 
    '''
    if array.dtype.type is np.float64 or array.dtype.type is np.float32:
        array=np.uint8(255*array)
    from PIL import Image
    if addblank:
        temp=np.zeros((array.shape[0]+1,array.shape[1],array.shape[2]),dtype=np.uint8)
        temp[1:,:,:]=array
        array=temp
    for k in range(array.shape[0]):
        I=Image.fromarray(array[k,:,:])
        I.save('temp%04d.png'%k)
    os.system('convert -delay %f temp*.png %s.gif'%(duration,path))
    for k in range(array.shape[0]):
        os.system('rm temp%04d.png'%k)

def plotGifGrid(dat,fn='test'):
    offset=8 # nr pixels for border padding
    rows=len(dat); cols=len(dat[0])
    h=dat[0][0].shape[0];w=dat[0][0].shape[1];t=dat[0][0].shape[2]
    R=np.ones((t+1,(h+offset)*rows,(w+offset)*cols),dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            i=((offset+h)*row+offset/2,(offset+w)*col+offset/2)
            temp=np.rollaxis(dat[row][col],2)
            R[1:,i[0]:i[0]+h,i[1]:i[1]+w]=temp
    ndarray2gif(fn,np.uint8(R*255),duration=0.1)
    
