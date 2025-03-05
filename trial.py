import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from joblib import Parallel, delayed
from time import time

def generate_data(trials,n,d,q):
    
    Data = np.empty((trials,n,d))
    
    Z = np.random.binomial(1,0.5,(trials,n))
    B = np.random.binomial(1,q,(trials,n))
    iid = np.random.binomial(1,0.5,(trials,n,d))
    
    Data = np.repeat((B*Z)[:,:,np.newaxis],d,axis=2) + np.repeat((1-B)[:,:,np.newaxis],d,axis=2)*iid
    
    return 2*Data - 1 


def LDP(x,eps):
    ## assumes x has 2-norm leq \sqrt{d}

    d = len(x)
    r = np.sqrt(d)
    B = np.sqrt(np.pi)*d*r*(np.exp(eps)+1)*gamma((d+1)/2)/((np.exp(eps)-1)*gamma(1+(d/2)))
    K = np.random.binomial(1,np.exp(eps)/(1+np.exp(eps)))
    S = np.random.binomial(1, 0.5 + (np.linalg.norm(x,2)/(2*r)))
    tilde_x = ((2*S)-1)*r*x/np.linalg.norm(x,2)
    
    while True:
        z = np.random.normal(0,1,d)
        z = (B)*z/np.linalg.norm(z,2)
        if (2*K-1)*np.dot(z,tilde_x) > 0:
            return z
        

def bdp_improved_vk(delta,eps,q,Data,l, seed = None):
    n = Data.shape[0]
    d = Data.shape[1]

    
    if seed is not None:
        np.random.seed(seed)
    
    for i in range(d):
        delta[i] = min(delta[i],eps)
    
    c = np.zeros(d)
    c[-1] = min(delta[-1],np.log((np.exp((0.5+q/2)*delta[0])+q-1)/q))
    for i in range(d-1):
        if c[-1] < delta[i]:
            c[i] = c[-1]
        else:
            c[i] = delta[i]-np.log(1-q+(q*np.exp(c[-1]))) 

    
    diff_ind = []
    diff_ind.append(0)
    c_diff = []
    c_diff.append(c[0])
    
    last_val = c[0]
    for i in range(d-1):
        if c[i+1] > 1e-5 + last_val:
            c_diff.append(c[i+1]-last_val)
            last_val = c[i+1]
            diff_ind.append(i+1)
            
    print(q,c,c_diff)
    
    dim_samples = np.zeros(d)
    for i in range(len(diff_ind)):
        dim_samples[diff_ind[i]:] += np.ones(d-diff_ind[i])
    
    c_diff = np.array(c_diff)
    num_stages = len(c_diff)

    
    Y = np.zeros((n,num_stages,d))
    mu = np.zeros((num_stages,d))
    
    w = np.zeros(num_stages)
    for i in range(num_stages):
        w[i] = c_diff[i]**2/(d - diff_ind[i])

    
    for t in range(num_stages):
        #temp = np.zeros(d - diff_ind[t])
        temp = 0
        for person in range(n):
            #Y[person,t,diff_ind[t]:d] = LDP(Data[person,diff_ind[t]:d],c_diff[t]) 
            #temp += Y[person,t,diff_ind[t]:d]
            temp += LDP(Data[person,diff_ind[t]:d],c_diff[t]) 
        mu[t,diff_ind[t]:d] = temp/n
    
    '''
    for t in range(num_stages):
        if np.linalg.norm(mu[t],2) > np.sqrt(d-diff_ind[t]):
            mu[t] = np.sqrt(d-diff_ind[t])*mu[t]/np.linalg.norm(mu[t],2)
    '''
    avg = np.zeros(d)
    for i in range(d):
        temp_num = 0 
        temp_den = 0
        for j in range(int(dim_samples[i])):
            if np.linalg.norm(mu[j],2) <= np.sqrt(d-diff_ind[j]):
                temp_num += w[j]*mu[j,i]
                temp_den += w[j]
            else:
                temp_num += w[j]*np.sqrt(d-diff_ind[j])*mu[j,i]/np.linalg.norm(mu[j],2)
                temp_den += w[j]
        avg[i] = temp_num/temp_den
        
        

    
    if np.linalg.norm(avg,2) > np.sqrt(d):
        avg = np.sqrt(d)*avg/np.linalg.norm(avg,2)

    '''    
    ###
    print('-----',q,c,c_diff)
    print(w)
    print([np.linalg.norm(mu[i,:],2) for i in range(num_stages)])
    print(np.linalg.norm(avg,2)**2)
    ###
    '''
    
    emp_avg = np.mean(Data,axis=0)

    return np.linalg.norm(avg-emp_avg,2)**2


def LDP_experiment(trials,n,d,q,delta,eps):
    np.random.seed(0)
    Data = generate_data(trials,n,d,q)
    errors = []
    def ldp_mean(Data,delta,seed=None):
      np.random.seed(seed)
      temp = 0
      for x in Data:
        temp += LDP(x,delta)
      temp = temp/n
      emp_avg = np.mean(Data,axis=0)
      return np.linalg.norm(temp-emp_avg,2)**2
    '''
    for i in range(trials):
        avg = 0
        for j in range(n):
             avg += LDP(Data[i,j],delta[0])
        avg = avg/n
        errors.append(np.linalg.norm(avg,2)**2)
    '''
    errors = Parallel(n_jobs=4)(delayed(ldp_mean)(Data[i],delta[0],i) for i in range(trials))
    return errors


def experiment_vk(trials,n,d,q,delta,eps,l):
    np.random.seed(0)
    Data = generate_data(trials,n,d,q)
    errors = Parallel(n_jobs=4)(delayed(bdp_improved_vk)(delta,eps,q,Data[i],l,i) for i in range(trials))
    '''
    errors = []
    for i in range(trials):
        errors.append(bdp_improved_vk(delta,eps,q,Data[i],l))
    '''
    return errors


start = time()

n = 10000
d = 10
eps = 2
trials = 1000
linsp = 10
np.random.seed(0)

delta = np.array([0.2,0.2,eps,eps,eps,eps,eps,eps,eps,eps])
qs =  np.linspace(0.01,1,linsp)


error_bdp = []
error_ldp = []


for q in qs:
    error_bdp.append(experiment_vk(trials,n,d,q,delta,eps,0.5))
    error_ldp.append(LDP_experiment(trials,n,d,q,delta,eps))

end = time()
print(end-start)

x = error_bdp[0]
print(np.mean(np.array(x)),np.mean(np.array(x))-np.std(np.array(x)))



plt.rcParams['text.usetex'] = True

plt.plot(qs,[np.median(np.array(x)) for x in error_bdp],color='blue',label=r'$M_{\mathsf{mean}}(\cdot)$')
plt.fill_between(qs, [np.quantile(np.array(x),0.25) for x in error_bdp], [np.quantile(np.array(x),0.75) for x in error_bdp],
    alpha=0.2, facecolor='blue')

plt.plot(qs,[np.median(np.array(x)) for x in error_ldp],color='red',label=r'\boldmath$\min \delta$-LDP')
plt.fill_between(qs, [np.quantile(np.array(x),0.25) for x in error_ldp], [np.quantile(np.array(x),0.75) for x in error_ldp],
    alpha=0.2, facecolor='red')


plt.xlabel('q')
plt.ylabel('MSE')
plt.legend(loc='upper right')


plt.show()
