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
        #parameter set
        pop_n = 6

        F1 = 1
        CR1 = 0.1

        F2 = 1
        CR2 = 0.9

        F3 = 0.8
        CR3 = 0.2

        F_L = [F1,F2,F3]
        CR_L = [CR1,CR2,CR3]
        #initial
        populations = []
        result = []
        for i in range(pop_n):
            solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            value = self.f.evaluate(func_num,solution)
            if float(value)<self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)
            populations.append(solution)
            result.append(value)
        self.eval_times+=pop_n
        flag = 0
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)
            #implement 3 stratege and generate 3 u
            new_populations = []
            new_result = []
            for i in range(pop_n):

                x_val = result[i]

                para = np.random.randint(0,3,size=1)[0]
                #print(para,F_L[para],CR_L[para])

                #先隨機取不同於xi的5個不同vectors
                index_l = [j for j in range(pop_n)]
                index_l.remove(i)
                x_index_l = np.random.choice(index_l, size=5, replace=False)
                #print(x_index_l)
                x1_index = x_index_l[0]
                x2_index = x_index_l[1]
                x3_index = x_index_l[2]
                x4_index = x_index_l[3]
                x5_index = x_index_l[4]
              


                #rand/1/bin
                u1 = []
                Xr1 = populations[x1_index]
                Xr2 = populations[x2_index]
                Xr3 = populations[x3_index]
                Xr4 = populations[x4_index]
                Xr5 = populations[x5_index]
                rand = np.random.rand()
                j_rand = np.random.randint(0,self.dim,size=1)[0]
                #print("rand: ", rand)
                #print("j_rand: ",j_rand)
                for j in range(self.dim):
                    uij = None
                    rand = np.random.rand()
                    #print(rand)
                    j_rand = np.random.randint(0,self.dim,size=1)[0]
                    #print(j_rand)
                    if rand<CR_L[para] or j==j_rand:
                        uij = Xr1[j] + F_L[para] * (Xr2[j] - Xr3[j])
                    else:
                        uij = populations[i][j]
                    #======
                    #if uij<self.lower: uij = populations[i][j]
                    #elif uij>self.upper: uij = populations[i][j]
                    if uij<self.lower:
                        uij = min(self.upper,2*self.lower-uij)
                    elif uij>self.upper:
                        uij = max(self.lower,2*self.upper-uij)
                    #======
                    u1.append(uij)
                u1 = np.array(u1)
                #print(u1)
                
                #rand/2/bin
                u2 = []
                for j in range(self.dim):
                    uij = None
                    rand = np.random.rand()
                    j_rand = np.random.randint(0,self.dim,size=1)[0]
                    if rand<CR_L[para] or j==j_rand:
                        uij = Xr1[j] + F_L[para] * (Xr2[j] - Xr3[j]) + F_L[para]* (Xr4[j] - Xr5[j])
                    else:
                        uij = populations[i][j]
                    if uij<self.lower:
                        uij = min(self.upper,2*self.lower-uij)
                    elif uij>self.upper:
                        uij = max(self.lower,2*self.upper-uij)
                    u2.append(uij)
                u2 = np.array(u2)
                #print(u2)
                
                #current-to-rand/1
                rand = np.random.rand()
                j_rand = np.random.randint(0,self.dim,size=1)[0]
                u3 = populations[i] + rand * (Xr1 - populations[i]) + F_L[para] * (Xr2 - Xr3)
                for j in range(self.dim):
                    #if u3[j]<self.lower: u3[j] = populations[i][j]
                    #elif u3[j]>self.upper: u3[j] = populations[i][j]
                    if u3[j]<self.lower: 
                        u3[j] = min(self.upper,2*self.lower-u3[j])
                    elif u3[j]>self.upper: 
                        u3[j] = max(self.lower,2*self.upper-u3[j])
                #print(u3)
                #return
                #==========================================#
                #==========================================#
                u1_val = self.f.evaluate(func_num,u1)
                u2_val = self.f.evaluate(func_num,u2)
                u3_val = self.f.evaluate(func_num,u3)
                if(u1_val=="ReachFunctionLimit" or u2_val=="ReachFunctionLimit" or u3_val=="ReachFunctionLimit"):
                    print("No more time!")
                    flag = 1
                    break
                #print(u1,u1_val)
                #print(u2,u2_val)
                #print(u3,u3_val)
                self.eval_times+=3
                u_solution = None
                sol_val = None
                min_solution = None
                min_value = None
                if u1_val <= u2_val and u1_val <= u3_val:
                    u_solution = u1
                    sol_val = u1_val
                elif u2_val < u1_val and u2_val <= u3_val:
                    u_solution = u2
                    sol_val = u2_val
                elif u3_val < u1_val and u3_val < u2_val:
                    u_solution = u3
                    sol_val = u3_val
                #print("u_sol:")
                #print(u_solution,sol_val)
                #print("x_sol:")
                #print(populations[i],x_val)
                if sol_val<x_val:
                    min_solution = u_solution
                    min_value = sol_val
                else:
                    min_solution = populations[i]
                    min_value = x_val
                #print("choose:")
                #print(min_solution,min_value)
                
                new_populations.append(np.array(min_solution))
                new_result.append(min_value)

                if float(min_value)<self.optimal_value:
                    self.optimal_solution[:] = min_solution
                    self.optimal_value = float(sol_val)
            if flag:break
            populations = new_populations
            result = new_result
            print("optimal: {}\n".format(self.get_optimal()[1]))
            

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
