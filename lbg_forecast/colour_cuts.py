import numpy as np
    
def SelectDropouts(dropout, dropout_data):
        
        sps_params, colours = dropout_data

        n_sources = len(sps_params)
        umg = colours[:, 0]
        gmr = colours[:, 1]
        rmi = colours[:, 2]
        imz = colours[:, 3]
    
        u_dropouts = []
        g_dropouts = []
        r_dropouts = []
        
        if(dropout == 'u'):
            i=0
            while i < n_sources:
                if(u_drop(umg[i], gmr[i])==True):
                    u_dropouts.append(sps_params[i])              
                i+=1

            return np.vstack(np.asarray(u_dropouts))
            
        if(dropout == 'g'):
            i=0
            while i < n_sources:
                if(g_drop(gmr[i], rmi[i])==True):
                    g_dropouts.append(sps_params[i])                   
                i+=1

            return np.vstack(np.asarray(g_dropouts))
            
        if(dropout == 'r'):
            i=0
            while i < n_sources:
                if(r_drop(rmi[i], imz[i])==True):
                    r_dropouts.append(sps_params[i])
                i+=1

            return np.vstack(np.asarray(r_dropouts))

def u_drop(ug,gr):
    if(ug>1.5 and gr>-1 and gr<1.2 and ug > (1.5*gr+0.75)):
        return True
    else:
        return False

def u_cut1(gr):
    cutl=[]
    for i in gr:
        cutl.append(1.5)
    return cutl

def u_cut2(ug):
    cutl=[]
    for i in ug:
        cutl.append(-1.0)
    return cutl

def u_cut3(ug):
    cutl=[]
    for i in ug:
        cutl.append(1.2)
    return cutl

def u_cut4(gr):
    cutl=[]
    for i in gr:
        cutl.append(1.5*i+0.75)
    return cutl
    
def g_drop(gr,ri):
    if(gr>1.0 and ri<1.0 and ri>-1.0 and gr > (1.5*ri+0.8)):
        return True
    else:
        return False
            
def g_cut1(ri):
    cutl=[]
    for i in ri:
        cutl.append(1.0)
    return cutl

def g_cut2(gr):
    cutl=[]
    for i in gr:
        cutl.append(1.0)
    return cutl
    
def g_cut3(gr):
    cutl=[]
    for i in gr:
        cutl.append(-1.0)
    return cutl

def g_cut4(ri):
    cutl=[]
    for i in ri:
        cutl.append(1.5*i+0.8)
    return cutl
    
def r_drop(ri,iz):
    if(ri>1.2 and iz<0.7 and iz>-1.0 and ri > (1.5*iz+1.0)):
        return True
    else:
        return False
            
def r_cut1(iz):
    cutl=[]
    for i in iz:
        cutl.append(1.2)
    return cutl

def r_cut2(ri):
    cutl=[]
    for i in ri:
        cutl.append(0.7)
    return cutl
    
def r_cut3(ri):
    cutl=[]
    for i in ri:
        cutl.append(-1.0)
    return cutl

def r_cut4(iz):
    cutl=[]
    for i in iz:
        cutl.append(1.5*i+1.0)
    return cutl    
