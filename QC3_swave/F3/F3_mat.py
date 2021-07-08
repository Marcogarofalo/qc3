import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from F2_mov import  Fmat00
from H_mat import Hmat00
from defns import chop,truncate
import projections as proj
from numba import jit,njit
import G_mov, K2i_mat
import time




import pickle
import os
import time
class database_Fmat00:
    filename='Fmat00.data'
    size_in=0
    def add(self,E,L,alpha,nnP,IPV):
        self.db.append([E,L,alpha,nnP,IPV, self.compute(E,L,alpha,nnP,IPV)])
        
        #self.values.append(self.compute(E,L,alpha,nnP,IPV))
        
    def get_value(self,E,L,alpha,nnP,IPV):
        found=False
        #start_db = time.time()
        for i in range(0,len(self.db)):
            if (E==self.db[i][0]):
                if (L==self.db[i][1]):
                    if (alpha==self.db[i][2]):
                        if (np.all(nnP==self.db[i][3])):
                            if (IPV==self.db[i][4]):
                                #print("value found in the database")
                                found=True
                                return self.db[i][5]
         
         
        #end_db = time.time()
        if(not found):
            #start = time.time()
            self.add(E,L,alpha,nnP,IPV)
            #end = time.time()
            #print('value not found: time scanning database:', end_db - start_db, ' s   time compute:', end - start,  'database size=',len(self.db))
            
            return self.db[-1][5]
                
    def compute(self,E,L,alpha,nnP,IPV):
        #print("computing new value: len=", len(self.values))
        print('value not found computing and adding to database, size=',len(self.db) )
        return Fmat00(E,L,alpha,nnP,IPV)
        
    
    def write(self):
        if(len(self.db)>self.size_in or (not os.path.isfile(self.filename)) ):
            with open(self.filename, 'wb') as file:
            # store the data as binary data stream
                print("writing a list of dimension ",len(self.db))
                pickle.dump(self.db , file)
                #setting size_in to the current database size
                self.size_in=len(self.db)
                
                
    def read(self):
        with open(self.filename, 'rb') as file:
        # read the data as binary data stream
            self.db = pickle.load(file)
            self.size_in=len(self.db)
            print("reading a database of dimension=",self.size_in)
            self.db=sorted(self.db)
    
    def __init__(self):
        if(os.path.isfile(self.filename) and  os.path.getsize(self.filename) > 0 ):
            print("found database file")
            self.read()
        else:
            print("creating a new database")
            self.db = []
            self.size_in=0 
            


class database_Fmat00_new:
    file_nnP='Fmat00_nnP.data'
    file_L='Fmat00_L.data'
    file_E='Fmat00_E.data'
    file_v='Fmat00_v.data'
    alpha=0.5
    IPV=0
    
        
    def add_nnP(self,E,L,alpha,nnP,IPV, v):
        self.nnP_list.append(nnP)
        self.L_list.append([L])
        self.E_list.append([[E]])
        self.v_list.append( [[ v ]]    )
        self.size_now+=1
        
    def add_L(self,E,L,alpha,nnP,n,IPV, v):   
        self.L_list[n].append(L)
        self.E_list[n].append([E])
        self.v_list[n].append( [  v ]    )
        self.size_now+=1
    def add_E(self,E,L,nL,alpha,nnP,n,IPV, v):   
        self.E_list[n][nL].append(E)
        self.v_list[n][nL].append(   v     )
        self.size_now+=1
        
    def get_value(self,E,L,alpha,nnP,IPV):
        found_nnP=-1
        found_L=-1
        found_E=-1
        for n in range(0,len(self.nnP_list)):
            if (np.all(nnP==self.nnP_list[n])):
                found_nnP=n
                for nL in range(0,len(self.L_list[n])):
                    if (L==self.L_list[n][nL]):
                        found_L=nL
                        for nE in range(0,len(self.E_list[n][nL])):
                            if (E==self.E_list[n][nL][nE]):
                                found_E=nE
                                return self.v_list[n][nL][nE]
        v=self.compute(E,L,alpha,nnP,IPV)    
        if ( found_nnP==-1 ):
            self.add_nnP(E,L,alpha,nnP,IPV ,v)
            return  v
        if ( found_L==-1):
            self.add_L(E,L,alpha,nnP, found_nnP ,IPV ,v)
            return  v
        if ( found_E==-1):
            self.add_E(E,L, found_L,alpha,nnP, found_nnP ,IPV ,v)
            return  v
            
    
                
    def compute(self,E,L,alpha,nnP,IPV):
        print("computing new value: ")
        return Fmat00(E,L,alpha,nnP,IPV)
        #return E+L
    
    def write(self):
        #print(self.size_now,self.size_read)
        if(self.size_now>self.size_read ):
            print("writing database")
            with open(self.file_nnP, 'wb') as file:
                pickle.dump(self.nnP_list , file)
            with open(self.file_L, 'wb') as file:
                pickle.dump(self.L_list , file)
            with open(self.file_E, 'wb') as file:
                pickle.dump(self.E_list , file)
            with open(self.file_v, 'wb') as file:
                pickle.dump(self.v_list , file)
                      
                
    def read(self):
        with open(self.file_nnP, 'rb') as file:
            self.nnP_list = pickle.load(file)
        with open(self.file_L, 'rb') as file:
            self.L_list = pickle.load(file)
        with open(self.file_E, 'rb') as file:
            self.E_list = pickle.load(file)
        with open(self.file_v, 'rb') as file:
            self.v_list = pickle.load(file)
        self.size_read=0
        for n in range(0,len(self.v_list)):
            for nL in range(0,len(self.v_list[n])):
                for nE in range(0,len(self.v_list[n][nL])):
                    self.size_read+=1
        self.size_now=self.size_read
        print("reading a database of dimension=",self.size_read)

        
        
    def __init__(self):
        tf =         os.path.isfile(self.file_nnP) and  os.path.getsize(self.file_nnP) > 0 
        tf = tf and  os.path.isfile(self.file_L)   and  os.path.getsize(self.file_L) > 0 
        tf = tf and  os.path.isfile(self.file_E)   and  os.path.getsize(self.file_E) > 0 
        tf = tf and  os.path.isfile(self.file_v)   and  os.path.getsize(self.file_v) > 0 
        if(tf):
            print("found database files")
            self.read()
        else:
            print("creating a new database")
            self.size_in=0 
            self.nnP_list=[]
            self.L_list=[]
            self.E_list=[]
            self.v_list=[]
            self.size_read=0
            self.size_now=0

            
       
 
db_Fmat00=database_Fmat00()
db_Fmat00_new=database_Fmat00_new()

def write_db_Fmat00():
    db_Fmat00.write()
    db_Fmat00_new.write()


###########################
import pandas as pd
class dataframe_Fmat00:
   
    def add(self,E,L,alpha,nnP,IPV):
        #self.db.append([E,L,alpha,nnP,IPV])
        self.df.loc[ self.df.index.max() + 1]=[E,L,alpha,nnP,IPV,  self.compute(E,L,alpha,nnP,IPV)  ]
        #self.values.append(self.compute(E,L,alpha,nnP,IPV))
        
    def get_value(self,E,L,alpha,nnP,IPV):
        found=False
        #start_db = time.time()
        print( self.df.index.max()  )
        for i in range(0,self.df.index.max() ):
            if (E==self.df['E'][i]):
                if (L==self.df['L'][i]):
                    if (alpha==self.df['alpha'][i]):
                        if (np.all(nnP==self.df['nnP'][i] )):
                            if (IPV==self.df['IPV'][i]):
                                #print("value found in the database")
                                found=True
                                return self.df['value'][i]
                            
        #end_db = time.time()
        if(not found):
            #start = time.time()
            self.add(E,L,alpha,nnP,IPV)
            #end = time.time()
            #print('time scanning database:', end_db - start_db, ' s   time compute:', end - start,  'database size=',len(self.db))
            return self.df['value'][self.df.index.max()]
                
    def compute(self,E,L,alpha,nnP,IPV):
        #print("computing new value: len=", len(self.values))
        return Fmat00(E,L,alpha,nnP,IPV)
    #    return E+L
    
    
    def __init__(self):
        data = {'E':[0],'L':[0],'alpha':[0],'nnP':[0],'IPV':[0],'value':[0]}
        self.df = pd.DataFrame(data)

        

df_Fmat00=dataframe_Fmat00()
###################################

##############################################################
# Compute full matrix F3 = 1/L**3 * (Ft/3 - Ft@Hi@Ft)
# Uses new structure w/ faster Fmat, Hmat
##############################################################
#@jit(fastmath=True,cache=True)
def F3mat00(E,L,alpha,nnP,kcot,IPV=0):
  # F00 = truncate(Fmat00(E,L,alpha,nnP,IPV))
  # Gt00 = truncate(G_mov.Gmat00_nnP(E,L,nnP))
  # K2it00 = truncate(K2i_mat.K2inv_mat00_nnP(E,L,nnP,kcot,IPV))
  
  
  #F00 = Fmat00(E,L,alpha,nnP,IPV)   # TB: changed list_nnk_nnP so that truncate is unnecessary
  
  #start_db = time.time()
  #F00 = db_Fmat00.get_value(E,L,alpha,nnP,IPV)
  #end_db = time.time()
  
  #start_df = time.time()
  F00 = db_Fmat00_new.get_value(E,L,alpha,nnP,IPV)
  #end_df = time.time()
  #print('time database:', end_db - start_db, ' s   time dataframe:', end_df - start_df,  'database size=',len(db_Fmat00.db))# , 'diff=',F00-F00_new

  
  #start = time.time()
  #F00 = Fmat00(E,L,alpha,nnP,IPV)
  #end = time.time()
  #print('time database:', end_db - start_db, ' s   time compute:', end - start,  'database size=',len(db_Fmat00.db))

  Gt00 = G_mov.Gmat00_nnP(E,L,nnP)
  K2it00 = K2i_mat.K2inv_mat00_nnP(E,L,nnP,kcot,IPV)

  Hi00 = chop(LA.inv( K2it00 + F00 + Gt00  ))
  return 1/L**3 * chop((1/3*F00 - F00@Hi00@F00))
#  return Hi00

@njit(fastmath=True,cache=True)
def LAinv(x):
  return LA.inv(x)


def isoprojector(E,L,nnP):
  nnk_vec = list_nnk_nnP(E,L,nnP)


@jit(fastmath=True,cache=True)
def F3mat00iso(E,L,Lista0,r0,P0,a2,alpha,IPV=0):
  F00 = Fmat00(E,L,alpha,IPV)
  F00o3 = 1./3*F00
  Gt00 = Gmatrix.Gmat00(E,L)
  res = []
  ones = np.ones(len(F00))
  for a0 in Lista0:
    K2it00 = K2i_mat.K2inv_mat00(E,L,a0,r0,P0,IPV)
    Hi00 = chop(LAinv( K2it00 + F00 + Gt00  ))
    f3mat = 1/L**3 * chop((F00o3 - F00@Hi00@F00))
    res.append(1./(ones@f3mat@ones))
  return res
