from typing import Sized
import numpy as np

# you must use python 3.6, 3.7, 3.8, 3.9 for sourcedefender
import sourcedefender
from HomeworkFramework import Function



class RS_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

        

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        mean_cr = 0.5
        mean_F = 0.5
        pop_N = 8
        p = 0.2
        A = []
        populations = []
        next_populations = []
        min_value = self.optimal_value
        min_solution = self.optimal_solution
        #Generation 0
        for i in range(pop_N):
            solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            populations.append(solution)
        
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)
            F_list = []
            CR_list = []
            #sort populations by f(x)
            pop_tmp = []
            for i in range(pop_N):
                val = self.f.evaluate(func_num, populations[i])
                if val<min_value:
                    min_value = val
                    min_solution = populations[i]
                pair = (i,populations[i],val)
                self.eval_times+=1
                if self.eval_times>FES:
                    print("No more times!")
                    return
                pop_tmp.append(pair)
            pop_tmp.sort(key = lambda x:x[2])
            result_val = []
            for i in range(pop_N):
                populations[i] = pop_tmp[i][1]
                result_val.append(pop_tmp[i][2])
            for i in range(pop_N):
                #Generate CRi = randni (μCR, 0.1), Fi = randci (μF, 0.1)
                CRi = np.random.normal(mean_cr,0.1)
                #print(CRi)
                
                Fi = 0.1*np.random.standard_cauchy()+mean_F
                while(Fi<0):
                    Fi = np.random.standard_cauchy()
                    Fi = 0.1*np.random.standard_cauchy()+mean_F
                    if Fi>1: Fi = 1
                if Fi>1: Fi = 1
                #print(Fi)
                #choose Xr1g , Xr2,g
                seed = np.random.randint(int(pop_N*p),size = 1)[0]
                Xp_best = populations[seed]
                
                seed = np.random.randint(int(pop_N),size = 1)[0]
                Xr1_g = populations[seed]
                while(list(Xr1_g)==list(populations[i])):
                    #print(seed,Xr1_g,populations[i])
                    seed = np.random.randint(int(pop_N*p),size = 1)[0]
                    Xr1_g = populations[seed]
                #===
                populations_l = []
                for j in populations:
                    populations_l.append(list(j))
                #===
                if A!=[]:
                    for j in A:
                        #print("j= \n",j)
                        if list(j) not in populations_l:
                            populations_l.append(j)
                seed = np.random.randint(int(pop_N),size = 1)[0]
                Xr2_g = populations[seed]
                #print("i== ",i)
                while(list(Xr2_g)==list(populations[i]) or list(Xr2_g)==list(Xr1_g)):
                    seed = np.random.randint(int(pop_N),size = 1)[0]
                    Xr2_g = populations[seed]
                Xig = populations[i]
                #print("Fi: ",Fi)
                #print("Xp_best: ",Xp_best)
                #print("Xr1_g: ",Xr1_g)
                #print("Xr2_g: ",Xr2_g)
                #print("Xig: ",Xig)
                Xig = np.array(Xig)
                Xp_best = np.array(Xp_best)
                Xr1_g = np.array(Xr1_g)
                Xr2_g = np.array(Xr2_g)
                Vig = Xig + Fi * (Xp_best-Xig) + Fi * (Xr1_g-Xr2_g)
                for j in range(self.dim):
                    if Vig[j]<self.lower:
                        Vig[j] = self.lower
                    elif Vig[j]>self.upper:
                        Vig[j] = self.upper
                #print("XVig: ",Vig)

                j_seed = np.random.randint(self.dim,size = 1)[0]
                Uig = []
                for j in range(self.dim):
                    Uji_g = None  
                    rand_seed = np.random.rand()
                    if j==j_seed or rand_seed<CRi:
                        Uji_g = Vig[j]
                    else:
                        Uji_g = Xig[j]
                    Uig.append(Uji_g)
                #calculate and arrange next generation
                val_u = self.f.evaluate(func_num,Uig)
                self.eval_times += 1
                if(self.eval_times>FES):
                    print("No more times")
                    return
                val_x = result_val[i]
                if val_x<=val_u:
                    next_populations.append(Xig)
                else:
                    #print("change!!",val_u)
                    if(val_u<min_value):
                        min_value  = val_u
                        min_solution = Uig
                    next_populations.append(Uig)
                    A.append(list(Xig))
                    if CRi not in CR_list:
                        CR_list.append(CRi)
                    if Fi not in F_list:
                        F_list.append(Fi)
            #Randomly remove sols from A to make |A|<=pop_N
            #print("A:",len(A))
            #print("CR_list: ", CR_list)
            #print("F_list: ",F_list)
            #print(A)
            while(len(A)>pop_N/2):
                seed = np.random.randint(len(A),size=1)[0]
                A.remove(A[seed])
            if(len(CR_list)!=0 and len(F_list)!=0):
                #print("update")
                c = (pop_N*2)/FES

                mean_SCR = sum(CR_list) / len(CR_list)
                mean_SF = None
                squre_sum = 0
                sum_ = sum(F_list)
                for j in F_list:
                    squre_sum+=j**2
                mean_SF = squre_sum/sum_
                mean_cr = (1-c)*mean_cr + c*mean_SCR
                mean_F = (1-c)*mean_F + c*mean_SF
            populations = next_populations
            #print(next_populations)
            if float(min_value) < self.optimal_value:
                self.optimal_solution[:] = min_solution
                self.optimal_value = float(min_value)
            print("optimal: {}\n".format(self.get_optimal()[1]))
            #return
            

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = RS_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 
