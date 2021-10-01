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
        #print(self.lower,self.upper)
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        print(self.lower,self.upper)
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)

            solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            print(type(solution))
            value = self.f.evaluate(func_num, solution)
            self.eval_times += 1

            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break            
            if float(value) < self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)

            print("optimal: {}\n".format(self.get_optimal()[1]))
            return



def show_model(vhm_models,dims,M):
    for i in range(dims):
        print("dim: ",i+1)
        print(vhm_models[i])
    return
def get_index(bins,lower_bd,upper_bd,ai_1,ai_M_minus_one,width,M,x):
    index = None
    if x<ai_1:return ai_1 # Ai,1
    elif x>=ai_M_minus_one:return upper_bd #Ai,M

    for i in bins.keys():
        if(x>=i):
            index = i
            break
    if index==None or index>upper_bd or index<lower_bd:
        print("Sth Wrong No index!",x,ai_1,ai_M_minus_one,width)
        exit(0)
    return index
def get_probability_from_model(bins):
    sum = 0
    bins_tmp = bins.copy()
    for i in bins_tmp.values():
        sum+=i
    for i in list(bins_tmp.keys()):
        bins_tmp[i] = bins_tmp[i]/sum
    return bins_tmp
def initial_vhm_models(dims,lower_bd,upper_bd,pop_n,populations_T):
    vhm_models = []
    pro_models = []

    for i in populations_T:#pop_n
        #find 1st,2nd min and 1st,2nd max
        M = 10
        first_min = i[0]
        sendcond_min = i[1]
        first_max = i[pop_n-1]
        sendcond_max = i[pop_n-2]
        ai_0 = lower_bd
        ai_M = upper_bd
        ai_1 = max(first_min-0.5*(sendcond_min-first_min),ai_0)
        ai_M_minus_one = min(first_max+0.5*(first_max-sendcond_max),ai_M)
        if ai_M_minus_one==ai_M:
            print("spec case")
            M-=1
        if ai_1==ai_0:
            print("spec case2")
            M-=1
        width = (ai_M_minus_one-ai_1)/(M-2)
        bins = {}

        #initial model bins
        bins[ai_1] = 0
        bins[ai_M] = 0
        for j in range(M-2):
            if ai_1+width*(j+1)>upper_bd:print("wrong width")
            bins[ai_1+width*(j+1)] = 0
        for k in list(bins.keys()):
            if k>upper_bd:
                print("initial wrong2!",list(bins.keys()),width)
                exit(1)
        print('---')
        #put xj into model
        for j in i:
            #get_m
            m = get_index(bins,lower_bd,upper_bd,ai_1,ai_M_minus_one,width,M,j) 
            if ai_1<=m and m<ai_M_minus_one: bins[m]+=1
            elif (m<ai_1 or m>=ai_M_minus_one) and ai_M_minus_one<ai_M: bins[m] = 0.1
            elif (m<ai_1 or m>=ai_M_minus_one) and ai_M_minus_one==ai_M: bins[m] = 0
            else:
                print("parsing index error!",m)
                exit(1)     
        vhm_models.append(bins)
        #get propability model
        pro_model = get_probability_from_model(bins)
        pro_models.append(pro_model)
    #show_model(vhm_models,dims,M)
    #print("pro_md:")
    #show_model(pro_models,dims,M)
    return vhm_models,pro_models
def sample(vhm_models,pro_modles,dims,lower_bd,upper_bd):
    x = []
    M = 10
    for i in range(dims):
        pro_modle = pro_modles[i]
        vhm_model = vhm_models[i]
        #print(pro_modle.keys())
        #print(vhm_model.keys())
        keys_l = list(pro_modle.keys())
        #print(pro_modle)
        keys_l.sort()
        #print(keys_l)
        
        width = keys_l[1]-keys_l[0]
        ai_1 = keys_l[0]
        ai_m_minus_one = keys_l[-2]
        #print(limit_min,limit_max)
        #print(pro_modle)
        keys_pro = []
        for i in keys_l:
            keys_pro.append(pro_modle[i])
        rand_key = np.random.choice(keys_l,1,keys_pro)[0]
        limit_max = rand_key
        #print(limit_max)
        limit_min = None
        if limit_max==ai_1:
            limit_min = lower_bd
        elif limit_max==upper_bd:
            limit_min = ai_m_minus_one
        else:
            limit_min = keys_l[keys_l.index(limit_max)-1]
        xi =  np.random.uniform(limit_min,limit_max,1)[0]
        if(xi<lower_bd or xi>upper_bd):
            print("out of range_eee!",xi,upper_bd)
            print(list(vhm_model.keys()))
            exit(1)
        x.append(float(xi))
    #print(x)
    return x
def cheap_ls(pair1,pair2,pair3):
    z_min = pair1[0]
    ε = 1.0 * (1/10)**50
    flag = 0
    if(abs(pair1[0]-pair2[0])>ε and abs(pair2[0]-pair3[0])>ε and abs(pair3[0]-pair1[0])>ε):

        a = 1/(pair2[0]-pair3[0])
        b = (pair1[1]-pair2[1])/(pair1[0]-pair2[0])
        c = (pair1[1]-pair3[1])/(pair1[0]-pair3[0])
        d = (pair1[0]+pair2[0])/(pair1[0]-pair3[0])
        c1 = a*(b-c)
        c2 = b-c1*d
        if(abs(c1)>ε):
            z_min = -1*(c2/(2*c1))
        flag = 1
    #print(z_min)
    #print("cheap_ls!!!")

    return z_min
class my_optimizer(Function): # need to inherit this class "Function"
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
        #print(self.dim, self.lower, self.upper)
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation

        #generate populations
        populations = []
        pop_N = 100
        for i in range(pop_N):
            solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            populations.append(solution)
        populations = np.array(populations)
        #print(populations)
        #print("=====")
        #print(populations_T)
        #print(FES)
        #print(self.lower,self.upper)
        min_solution = None
        min_value = self.optimal_value
        next_result = []
        new_populations0 = []
        for i in populations:
            pair = ()
            value = self.f.evaluate(func_num,i)
            pair = (i,value)
            new_populations0.append(pair)
            if value < min_value:
                min_solution = i
                min_value = value
        new_populations0.sort(key = lambda x:x[1])
        populations = []
        for i in new_populations0:
            populations.append(i[0])
            next_result.append(i[1])

        populations = np.array(populations)
        self.eval_times+=pop_N
        
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)
            
            #detect all pops
            #self.detect(populations)
            
            #sort population_T
            populations_T = populations.transpose()
            for i in populations_T:
                i.sort()
            populations_T = np.array(populations_T)
            #print(populations_T)
            
            #build models
            vhm_model = None
            pro_models = None
            width = 0
            (vhm_models,pro_models) = initial_vhm_models(self.dim,self.lower,self.upper,pop_N,populations_T)
            
            #Sample new candidates by sampling
            #show_model(pro_models,6,5)
            y = []
            eval_cnt = 0
            for i in range(pop_N):
                yi = sample(vhm_models,pro_models,self.dim,self.lower,self.upper)
                Pb = 0.2
                Pc = 0.2
                max_int = int(Pb*pop_N)-1
                #print(max_int)
                k = np.random.randint(2,max_int)
                for j in range(self.dim):
                    seed = float(np.random.rand())
                    if seed<Pc:
                        #print(k)
                        x1 = populations[k-1][j]
                        y1 = next_result[k-1]
                        x2 = populations[k][j]
                        y2 = next_result[k]
                        x3 = populations[k+1][j]
                        y3 = next_result[k+1]
                        yi[j] = cheap_ls((x1,y1),(x2,y2),(x3,y3))
                    if yi[j]<self.lower: yi[j] = 0.5*(populations[i][j]+self.lower)
                    elif yi[j]>self.upper: yi[j] = 0.5*(populations[i][j]+self.upper)
                    else: yi[j] = yi[j]
                y.append(yi)
           
            #self.detect(y)
            #update populations
            #pop ← select({x1, x2, . . . , xN} ∪ {y1, y2, . . . , yN}).
            #print(np.size(y))
            new_populations = []
            y_result = []
            for i in range(pop_N):
                val = self.f.evaluate(func_num,y[i])
                pair_l1 = (i,y[i],val)
                if val<min_value:
                    print("what!?")
                eval_cnt+=1
                pair_l2 = (i,populations[i],next_result[i])
                y_result.append(pair_l1)
                y_result.append(pair_l2)
            #print(y_result)
            
            y_result.sort(key=lambda x:x[2])

            next_result = []
            for i in range(pop_N):
                next_result.append(y_result[i][2])
                new_populations.append(y_result[i][1]) 
            #for i in range(pop_N):
             #   print(i,y_result[i][2])
            #print(next_result)
            if next_result[0]<min_value:
                #print("update!")
                min_value = next_result[0]
                min_solution = new_populations[0]
            
            populations = np.array(new_populations)
            
            #print(min_solution)
            #print(min_value)
            value = min_value
            self.eval_times += eval_cnt
            if float(min_value) < self.optimal_value:
                self.optimal_solution[:] = min_solution
                self.optimal_value = float(min_value)
            print("optimal: {}\n".format(self.get_optimal()[1]))

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 2:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = my_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print("==============================================")
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 
