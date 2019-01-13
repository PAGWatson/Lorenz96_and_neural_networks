"""This contains code to perform simulations of the Lorenz '96 system (see Lorenz E., 1996, Predictability: a problem partly solved. In Predictability. Proc 1995. ECMWF Seminar, 1-18). It can simulate the full system, the truncated model or use a neural network either alone or in combination with the truncated model, chosen with the "param" argument.
Modified from code at https://github.com/jswhit/L96/blob/master/L96.py"""

"""Initialise model as model = L96(*args)
Arguments:
members - no. of models
nx - no. of X variables
ny - no. of Y variables per X-variable.
dt - time step
fc_length - no. of time units to run for.
h,b,c -parameters of model
F - forcing value in dX/dt
param - the parameterisation to use: None to run the truth model with simulation of the Y terms; 'cubic' to use truncated model with a cubic function of X to approximate the tendency from the Y terms; 'NN_full' to use a neural network to predict the full tendency; 'NN_err' to use a neural network to correct the error of the truncated model.
x - set this to the initial conditions. If set, the array must have a shape with the last two dimensions being members by nx - runs are done for all indices of the preceding dimensions simulataneously, allowing runs starting from multiple initial conditions to be done simultaneously. If x is None, the code will generate random (but reproducible) initial conditions.
rs - optional, a random variable for initialising the model
nn - give neural network object to use in making predictions
nn_x_train_mean - give mean of training data used to train neural network
nn_x_train_std - give standard deviation of training data used to train neural network
nn_delta_x_input - the maximum distance of the neural network input data in X-space
"""

import fnmatch
import numpy as np

class L96:

    def __init__(self,members=1,nx=8,ny=32,dt=0.001,h=1,b=10,c=4,F=20,param=None,x=None,rs=None,
                 nn=None,nn_x_train_mean=None,nn_x_train_std=None,nn_delta_x_input=None):
        
        self.param = param
        self.nx = nx 
        if self.param is None:
            self.ny=ny 
        self.dt = dt
        self.h=h  
        self.b=b  
        self.c=c  
        self.F = F
        self.members = members
        if rs is None:
            rs = np.random.RandomState(1)  #to get reproducible results.
        self.rs = rs
        
        if x is not None:
            assert x.shape[-2:]==(members,nx)
            self.x=x
        else:
            self.x = F+0.1*self.rs.standard_normal(size=(members,nx))
            
        param_is_allowed= param in [None,'cubic'] or fnmatch.fnmatch(param,'NN*')
        assert param_is_allowed, 'L96: param '+param+' not recognised.'
        if self.param is None:
            self.y = np.zeros(self.x.shape+(ny,))  #added Y-variables
            self.ywrap = np.zeros(self.y.shape[:-1]+(self.ny+3,), np.float)
        elif fnmatch.fnmatch(self.param,'NN*'):
            self.nn=nn
            self.nn_x_train_mean=nn_x_train_mean
            self.nn_x_train_std=nn_x_train_std
            self.nn_delta_x_input=nn_delta_x_input
        
        if self.param=='NN_full':
            self.xwrap=np.zeros(self.x.shape[:-1]+(self.nx+2*self.nn_delta_x_input,), np.float)
        else:
            self.xwrap = np.zeros(self.x.shape[:-1]+(self.nx+3,), np.float)
            if self.param=='NN_err':
                self.xwrap_nn=np.zeros(self.x.shape[:-1]+(self.nx+2*self.nn_delta_x_input,), np.float)  #wrapped array for NN inputs

        self.forcing = self.F
        
        self.t=0  #for saving the integration time, used for saving data.
        self.nsteps=0  #counter for number of time steps

    def shiftx(self,x):
        xwrap = self.xwrap
        xwrap[...,2:self.nx+2] = x
        xwrap[...,1]=x[...,-1]
        xwrap[...,0]=x[...,-2]
        xwrap[...,-1]=x[...,0]
        
        xm2=xwrap[...,0:self.nx]
        xm1=xwrap[...,1:self.nx+1]
        xp1=xwrap[...,3:self.nx+3]
        return xm2,xm1,xp1

    def shifty(self,y):
        ywrap = self.ywrap
        ywrap[...,1:-2] = y

        ywrap[...,1:,0]=y[...,:-1,-1]
        ywrap[...,0,0]=y[...,-1,-1]

        ywrap[...,:-1,-2:]=y[...,1:,:2]
        ywrap[...,-1,-2:]=y[...,0,:2]

        ym1=ywrap[...,0:self.ny]
        yp1=ywrap[...,2:self.ny+2]
        yp2=ywrap[...,3:self.ny+3]

        return ym1,yp1,yp2

    #PW - function to calculate the x-dependent term in the X-evolution equations.
    def x_term(self,x):
        
        if self.param=='NN_full':
            #Make array with x-values wrapped around
            self.xwrap[...,self.nn_delta_x_input:-self.nn_delta_x_input]=self.x[...,:]
            self.xwrap[...,:self.nn_delta_x_input]=self.x[...,-self.nn_delta_x_input:]
            self.xwrap[...,-self.nn_delta_x_input:]=self.x[...,:self.nn_delta_x_input]
            
            nn_input=np.zeros(self.x.shape+(2*self.nn_delta_x_input+1,))
            for i in range(self.nx):
                nn_input[...,i,:]=self.xwrap[...,i:i+2*self.nn_delta_x_input+1]
            nn_input=nn_input.reshape(-1,2*self.nn_delta_x_input+1)

            nn_input_scaled = (nn_input-self.nn_x_train_mean)/self.nn_x_train_std  #rescaling inputs to NN
            
            x_term=self.nn.predict(nn_input_scaled)
            x_term=x_term.reshape(self.x.shape)

        else:
            xm2,xm1,xp1 = self.shiftx(x)
            x_term=(xp1-xm2)*xm1 - x + self.forcing

        return x_term
        
    #PW - function to calculate the y-dependent term in the X-evolution equations, calling this "U", following Arnold et al. (2013).
    def U_term_true(self,y):
        return float(self.h)*self.c/self.b*np.sum(y, axis=-1)

    def dxdt(self):
        x_term=self.x_term(self.x)
        
        if self.param is None:
            U_term=self.U_term_true(self.y)
        elif self.param in ['cubic','NN_err']:
            U_term=-0.207 + 0.577*self.x - 0.00553*self.x**2 - 0.000220*self.x**3
        elif self.param=='NN_full':
            U_term=0  #there is no separate term relating to small-scale processes when using a NN trained to reproduce output of the full model.

        if self.param=='NN_err':
            #Code similar to that for using the NN computing the full tendency in x_term()
            self.xwrap_nn[...,self.nn_delta_x_input:-self.nn_delta_x_input]=self.x[...,:]
            self.xwrap_nn[...,:self.nn_delta_x_input]=self.x[...,-self.nn_delta_x_input:]
            self.xwrap_nn[...,-self.nn_delta_x_input:]=self.x[...,:self.nn_delta_x_input]
            
            nn_input=np.zeros(self.x.shape+(2*self.nn_delta_x_input+1,))
            for i in range(self.nx):
                nn_input[...,i,:]=self.xwrap_nn[...,i:i+2*self.nn_delta_x_input+1]
            nn_input=nn_input.reshape(-1,2*self.nn_delta_x_input+1)

            nn_input_scaled = (nn_input-self.nn_x_train_mean)/self.nn_x_train_std  #rescaling inputs to NN
            
            NN_term=self.nn.predict(nn_input_scaled)
            NN_term=NN_term.reshape(self.x.shape)
        else:
            NN_term=0
        
        return x_term - U_term + NN_term

    #PW - added dydt function
    def dydt(self):
        ym1,yp1,yp2 = self.shifty(self.y)
        x_term=float(self.h)*self.c/self.b*np.tile(self.x[...,np.newaxis], (1,)*len(self.x.shape)+(self.ny,))
        return self.b*self.c*(ym1-yp2)*yp1 - self.c*self.y + x_term


    def advance(self):
        h = self.dt
        hh = 0.5*h
        h6 = h/6.
        
        x = self.x
        if self.param is None:
            y = self.y
        
        dxdt1 = self.dxdt()
        if self.param is None:
            dydt1 = self.dydt()
        self.x = x + hh*dxdt1
        if self.param is None:
            self.y = y + hh*dydt1
        
        dxdt2 = self.dxdt()
        if self.param is None:
            dydt2 = self.dydt()
        self.x = x + hh*dxdt2
        if self.param is None:
            self.y = y + hh*dydt2
        
        dxdt = self.dxdt()
        if self.param is None:
            dydt = self.dydt()
        self.x = x + h*dxdt
        if self.param is None:
            self.y = y + h*dydt
        
        dxdt2 = 2.0*(dxdt2 + dxdt)
        dxdt = self.dxdt()
        if self.param is None:
            dydt2 = 2.0*(dydt2 + dydt)
            dydt = self.dydt()
        self.x = x + h6*(dxdt1 + dxdt + dxdt2)
        if self.param is None:
            self.y = y + h6*(dydt1 + dydt + dydt2)
        
        self.nsteps+=1
        self.t=self.nsteps*self.dt  #calculating the time elapsed
    
    #function to save the model state
    def save_state(self):
        save_ind=int(round(((self.t-self.t0)-self.discard)/self.save_int)) + self.save_ind_start
        self.x_save[save_ind,...]=self.x
        if self.param is None:
            self.y_save[save_ind,...]=self.y
        
    #function to do a run of length fc_length, saving output at the end of every time interval of length save_int, discarding the first discard time units.
    #Note, the run cannot be restarted.
    def run(self, fc_length=1, save_int=None, discard=0):
        self.fc_length=fc_length
        self.save_int=save_int
        self.discard=discard
        
        fc_steps=int(np.ceil(float(self.fc_length)/self.dt))

        #Create arrays for saving data, if save_int is set.
        if self.save_int:
            assert self.save_int/self.dt %1 == 0, "save_int should be a multiple of dt."
            assert self.discard/self.dt %1 == 0, "discard should be a multiple of dt."
            
            
            self.steps_between_saves=int(self.save_int/self.dt)
            self.steps_discard=int(self.discard/self.dt)
            n_saves=(fc_steps-self.steps_discard)/self.steps_between_saves+1
            x_save_new=np.zeros((n_saves,)+(self.x.shape))
            if self.param is None:
                y_save_new=np.zeros((n_saves,)+(self.y.shape))
            
            #If the model has previously been run, concatenate the new array onto the existing array
            if hasattr(self,'x_save'):
                self.save_ind_start=self.x_save.shape[0]
                self.x_save=np.concatenate((self.x_save,x_save_new[:-1]))
                if self.param is None:
                    self.y_save=np.concatenate((self.y_save,y_save_new[:-1]))
            else:
                self.x_save=x_save_new
                if self.param is None:
                    self.y_save=y_save_new
                self.save_ind_start=0
            
            self.t0=self.t  #save start time of run

        for i in range(fc_steps):
            
            #Print the time every 10 time units
            if fc_length>10 and self.t>self.dt and ( 5<(self.t-0.9*self.dt)%10<10 and 0.1*self.dt<(self.t+0.9*self.dt)%10<5 ):  #this is written so as not to be prone to machine precision errors
                print 't =',self.t
            
            #Do saving first, so the initial values are saved if discard is 0.
            if self.save_int and self.nsteps>=self.steps_discard and (self.nsteps-self.steps_discard)%self.steps_between_saves==0:
                self.save_state()

            self.advance()
            
            #Save the final state, if this occurs at a save point.
            if self.save_int and self.nsteps==fc_steps and (self.nsteps-self.steps_discard)%self.steps_between_saves==0:
                self.save_state()
            