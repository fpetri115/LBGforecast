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
    
    
    
def SelectDropouts(dropout, colour_list_full):
    
        umg_list_full, gmr_list_full, rmi_list_full, imz_list_full, z_list_full, index_list = colour_list_full
    
        u_dropouts = []
        u_index = []
        g_dropouts = []
        g_index = []
        r_dropouts = []
        r_index = []
        
        if(dropout == 'u'):
            i=0
            while i < len(umg_list_full):
                if(u_drop(umg_list_full[i], gmr_list_full[i])==True):
                    
                    u_dropouts.append(z_list_full[i])
                    u_index.append(i)
                    
                i+=1
            return u_dropouts, u_index
            
        if(dropout == 'g'):
            i=0
            while i < len(gmr_list_full):
                if(g_drop(gmr_list_full[i], rmi_list_full[i])==True):
                    
                    g_dropouts.append(z_list_full[i])
                    g_index.append(i)
                    
                i+=1
            return g_dropouts, g_index
            
        if(dropout == 'r'):
            i=0
            while i < len(rmi_list_full):
                if(r_drop(rmi_list_full[i], imz_list_full[i])==True):
                    
                    r_dropouts.append(z_list_full[i])
                    r_index.append(i)
                    
                i+=1
            return r_dropouts, r_index