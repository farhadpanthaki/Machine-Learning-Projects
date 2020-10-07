#To find minima of himmelblau function using GA in the range (0,6)
import numpy as np
import pandas as pd
u_lim = 6
l_lim = 0
#Fuction definitions:
#1.Fitness Calc
def himb_fitness(X,l_lim,u_lim):#Takes input as the binary encoded population and out put as the output of the himmelblau function
    X1 = b_to_d(X[0:int(len(X.index)/2),].array,l_lim,u_lim)
    #print(X1)
    X2 = b_to_d(X[int(len(X.index)/2):len(X.index),].array,l_lim,u_lim)
   # print(X2)
    return(np.absolute(himmelblau (X1,X2) ))
#2.Swap
def swap_at_k(a,b,k):
    s1 = a.copy()
    s2 = b.copy()
   # print(s1)
   # print(s2)
    temp = s1.iloc[k:len(s1)].copy()#if copy method is not used then temp refers to the same data in memory. 
   # print(temp)
    s1[k:len(s1)] = s2[k:len(s1)]
  #  print(s1)
    s2[k:len(s1)] = temp
   # print(s2)
    return [s1,s2]
#res[1].name for the index number in the fitness array
#s1 = population[0]
#print(s1)

#s2 = population[1]
#print(s2)

#temp = s1.iloc[3:len(s1)].copy() 
#s1[3:len(s1)] = s2[3:len(s1)]
#s2[3:len(s1)] = temp
#print(s1)
#print(s2)

#3.Mutation
def mutation(cross_over_populationf):
    pop_size = len(cross_over_population.columns)
    number_of_bits = len(cross_over_population.index)
    P = 1 /(pop_size)
     #The probability of uniform random number between 0 and 1 having value less than a certain x is x. SO if a random unifomr number comes less than P when its probability is P!
    mutated_population = pd.DataFrame(index =range(len(cross_over_population.index)), columns = range(pop_size),dtype = "float64")
    for i in range(pop_size):
        #for j in cross_over_population.index:
        mutated_population[i] = cross_over_populationf.iloc[:,i].where(np.random.uniform(0,1,len(cross_over_populationf.index))< P ,np.logical_not)
        #print(mutated_population)
    return(mutated_population)

#4.Mapping Function
def b_to_d(b_number,lower_l,upper_l):
    bits = len(b_number)
    
    inc = ((upper_l - lower_l )/(np.power(2,bits)-1)) * np.sum(np.power(2,np.arange(bits -1 , -1, -1)) * np.array(b_number))
    return(lower_l + inc)
    

###########




def himmelblau (x1,x2):
    if np.all(np.logical_and(np.logical_and(np.less_equal(x1,6), np.greater_equal(x1,0)), np.logical_and(np.less_equal(x2,6),np.greater_equal(x2,0)))):
        return(np.power((np.power(x1,2) + x2 -11),2) + np.power((x1 + np.power(x2,2) - 7),2))
    
##
##from sklearn.preprocessing import scale, LabelEncoder
##X1 = np.linspace(0,6,100)
##X2 = np.linspace(0,6,100)
##X = pd.DataFrame({"X1" : X1,
##                  "X2" : X2})
##Y = pd.Series(index=X.index)
##for i in X.index:
##    Y[i] = himmelblau(X.iloc[i,0],X.iloc[i,1])
##
number_of_bits = 30 #per variable ie 10 for X1 and X2
pop_size = 2 #should be divisible by 2

population = pd.DataFrame()
epochs = 1

##
##
##
fitness = []
##X_current = pd.DataFrame()
for i in range(pop_size):
    #population[i] =  np.random.choice(2,2 * number_of_bits)
    population[i] = np.zeros(2* number_of_bits)
##    X_current = X.loc[:,population[i] == 1] #use population as on/off switch along each col to select features for accuracy evaluation
    fitness.append( himb_fitness(population[i],l_lim,u_lim))#fitness is a pandas series containing the accuracy values for the different chromosomes in the population
fitness = pd.Series(fitness) # The index of fitness is the index of the population with that fitness value
##
for le in range(epochs):
##    #Tournament Selection
    sorted_fitness = fitness.sort_values()

    sf_sum = sorted_fitness.sum()
    length  = len(sorted_fitness)


    prob =   (sf_sum - sorted_fitness) / (length * sf_sum - sf_sum) #smaller vaues have higher prob
    ##
    tournament_result = np.random.choice(sorted_fitness.index,size = pop_size, p = prob ,replace= True) #This has the indices of the population which are selected for crossover. This is the result of the tournament of size  = pop_size.
    ##
    intermediate_population = population[tournament_result]
    ##
    ##    #X_intermediate = X.iloc[:,intermediate_population.iloc[7]] To get the X df corresponding to the current chromosome
    ##
    ##    #Crossover
    ##
    ##    #First create pairs for crossover
    pairs = np.random.choice(pop_size,pop_size,replace=False)
    ##  for i in range(1,pop_size,2):        #Good way to include all pairs using simple indices i and i-1.
    ##    crossover_population = swap_at_k( intermediate_population[i] intermediate_population[i-1],k)
    ##    #>>> (population[1] == population[0]).sum()
    ##
    shuffled_population = intermediate_population.iloc[::,pairs] #initialize dataframe for crossover.
    cross_over_population = pd.DataFrame()
    for i in range(1,pop_size,2):
            crossver_list_temp = swap_at_k(shuffled_population.iloc[::,i-1],shuffled_population.iloc[::,i],np.random.randint(0,len(population[i-1])))
            cross_over_population[i-1] = crossver_list_temp[0]
            cross_over_population[i] = crossver_list_temp[1]
    ##        #basic two point crossover. Function defined above.
    ##
    ##    #Mutation. Each bit in every population has a probability of being mutated. but that number is very small. P = (1/(np.log(pop_size))) * (np.power(10,-2))
    ##
    ##
    mutated_population = mutation(cross_over_population) #The function as defined returns boolean values
    mutated_population = mutated_population.astype(int) # Convert boolean array to int array
    ##
    ##    #Evaluate fitness for mutated_population
    fitness_children = []
    for i in mutated_population.columns:
        fitness_children.append(himb_fitness(mutated_population.loc[::,i],l_lim,u_lim))
    fitness_children = pd.Series(fitness_children) #The index of mutated_population and fitness_children are coupled.
    sorted_fitness_children = fitness_children.sort_values()# i of mutated_population and fitness_children is "coupled"
    sorted_mutated_population  = mutated_population[sorted_fitness_children.index]
    parent_retain_ratio = 0.3
    parents_retained = np.floor(parent_retain_ratio * pop_size)

    selected_parents = intermediate_population.iloc[:,0:parents_retained.astype(int)]
    selected_children = sorted_mutated_population.iloc[:,0:(pop_size - parents_retained.astype(int))] #mutated_population
    new_generation = pd.concat([selected_parents,selected_children],axis = 1,ignore_index= True)
    new_fitness = pd.concat([sorted_fitness.loc[selected_parents.columns.array],sorted_fitness_children.iloc[0:(pop_size - parents_retained.astype(int))]],axis = 0,ignore_index = True) #Top 3 of tournament winners and top pop_size - 3 from the sorted mutated population
    sorted_new_generation = new_generation.iloc[::,new_fitness.sort_values().index]
    population  = sorted_new_generation
    population.columns = range(pop_size)
    fitness = new_fitness.sort_values(ignore_index=True)
    
#best_X = X.loc[:,new_generation.iloc[:,0] == 1 ]
#print(best_X)
for i in range(pop_size): 
#best_fitness_index = new_fitness.sort_values().index[0]

    best_fitness_index = i
    best_min =  {'x1' :b_to_d(population[best_fitness_index][0:int(len(new_generation.index)/2),],l_lim,u_lim),
                 'x2' : b_to_d(population[best_fitness_index][int(len(new_generation.index)/2):len(new_generation.index),],l_lim,u_lim)}
    print(best_min)
    print(fitness.loc[best_fitness_index])
   

##number_of_bits = 5
##
##population = pd.DataFrame()
##population[0] =  np.random.choice(2,2 * number_of_bits)
##print(himb_fitness(population[0],0,6))

for le in range(epochs):
##    #Tournament Selection
    sorted_fitness = fitness.sort_values()

    sf_sum = sorted_fitness.sum()
    length  = len(sorted_fitness)


    prob =   (sf_sum - sorted_fitness) / (length * sf_sum - sf_sum) #smaller vaues have higher prob
    ##
    tournament_result = np.random.choice(sorted_fitness.index,size = pop_size, p = prob ,replace= True) #This has the indices of the population which are selected for crossover. This is the result of the tournament of size  = pop_size.
    ##
    intermediate_population = population[tournament_result]
    ##
    ##    #X_intermediate = X.iloc[:,intermediate_population.iloc[7]] To get the X df corresponding to the current chromosome
    ##
    ##    #Crossover
    ##
    ##    #First create pairs for crossover
    pairs = np.random.choice(pop_size,pop_size,replace=False)
    ##  for i in range(1,pop_size,2):        #Good way to include all pairs using simple indices i and i-1.
    ##    crossover_population = swap_at_k( intermediate_population[i] intermediate_population[i-1],k)
    ##    #>>> (population[1] == population[0]).sum()
    ##
    shuffled_population = intermediate_population.iloc[::,pairs] #initialize dataframe for crossover.
    cross_over_population = pd.DataFrame()
    for i in range(1,pop_size,2):
            crossver_list_temp = swap_at_k(shuffled_population.iloc[::,i-1],shuffled_population.iloc[::,i],np.random.randint(0,len(population[i-1])))
            cross_over_population[i-1] = crossver_list_temp[0]
            cross_over_population[i] = crossver_list_temp[1]
    ##        #basic two point crossover. Function defined above.
    ##
    ##    #Mutation. Each bit in every population has a probability of being mutated. but that number is very small. P = (1/(np.log(pop_size))) * (np.power(10,-2))
    ##
    ##
    mutated_population = mutation(cross_over_population) #The function as defined returns boolean values
    mutated_population = mutated_population.astype(int) # Convert boolean array to int array
    ##
    ##    #Evaluate fitness for mutated_population
    fitness_children = []
    for i in mutated_population.columns:
        fitness_children.append(himb_fitness(mutated_population.loc[::,i],l_lim,u_lim))
    fitness_children = pd.Series(fitness_children) #The index of mutated_population and fitness_children are coupled.
    sorted_fitness_children = fitness_children.sort_values()# i of mutated_population and fitness_children is "coupled"
    sorted_mutated_population  = mutated_population[sorted_fitness_children.index]
    parent_retain_ratio = 0.3
    parents_retained = np.floor(parent_retain_ratio * pop_size)

    selected_parents = intermediate_population.iloc[:,0:parents_retained.astype(int)]
    selected_children = sorted_mutated_population.iloc[:,0:(pop_size - parents_retained.astype(int))] #mutated_population
    new_generation = pd.concat([selected_parents,selected_children],axis = 1,ignore_index= True)
    new_fitness = pd.concat([sorted_fitness.loc[selected_parents.columns.array],sorted_fitness_children.iloc[0:(pop_size - parents_retained.astype(int))]],axis = 0,ignore_index = True) #Top 3 of tournament winners and top pop_size - 3 from the sorted mutated population
    sorted_new_generation = new_generation.iloc[::,new_fitness.sort_values().index]
    population  = sorted_new_generation
    population.columns = range(pop_size)
    fitness = new_fitness.sort_values(ignore_index=True)

for i in range(pop_size): 
#best_fitness_index = new_fitness.sort_values().index[0]

    best_fitness_index = i
    best_min =  {'x1' :b_to_d(population[best_fitness_index][0:int(len(new_generation.index)/2),],l_lim,u_lim),
                 'x2' : b_to_d(population[best_fitness_index][int(len(new_generation.index)/2):len(new_generation.index),],l_lim,u_lim)}
    print(best_min)
    print(fitness.loc[best_fitness_index])
for le in range(epochs):
##    #Tournament Selection
    sorted_fitness = fitness.sort_values()

    sf_sum = sorted_fitness.sum()
    length  = len(sorted_fitness)


    prob =   (sf_sum - sorted_fitness) / (length * sf_sum - sf_sum) #smaller vaues have higher prob
    ##
    tournament_result = np.random.choice(sorted_fitness.index,size = pop_size, p = prob ,replace= True) #This has the indices of the population which are selected for crossover. This is the result of the tournament of size  = pop_size.
    ##
    intermediate_population = population[tournament_result]
    ##
    ##    #X_intermediate = X.iloc[:,intermediate_population.iloc[7]] To get the X df corresponding to the current chromosome
    ##
    ##    #Crossover
    ##
    ##    #First create pairs for crossover
    pairs = np.random.choice(pop_size,pop_size,replace=False)
    ##  for i in range(1,pop_size,2):        #Good way to include all pairs using simple indices i and i-1.
    ##    crossover_population = swap_at_k( intermediate_population[i] intermediate_population[i-1],k)
    ##    #>>> (population[1] == population[0]).sum()
    ##
    shuffled_population = intermediate_population.iloc[::,pairs] #initialize dataframe for crossover.
    cross_over_population = pd.DataFrame()
    for i in range(1,pop_size,2):
            crossver_list_temp = swap_at_k(shuffled_population.iloc[::,i-1],shuffled_population.iloc[::,i],np.random.randint(0,len(population[i-1])))
            cross_over_population[i-1] = crossver_list_temp[0]
            cross_over_population[i] = crossver_list_temp[1]
    ##        #basic two point crossover. Function defined above.
    ##
    ##    #Mutation. Each bit in every population has a probability of being mutated. but that number is very small. P = (1/(np.log(pop_size))) * (np.power(10,-2))
    ##
    ##
    mutated_population = mutation(cross_over_population) #The function as defined returns boolean values
    mutated_population = mutated_population.astype(int) # Convert boolean array to int array
    ##
    ##    #Evaluate fitness for mutated_population
    fitness_children = []
    for i in mutated_population.columns:
        fitness_children.append(himb_fitness(mutated_population.loc[::,i],l_lim,u_lim))
    fitness_children = pd.Series(fitness_children) #The index of mutated_population and fitness_children are coupled.
    sorted_fitness_children = fitness_children.sort_values()# i of mutated_population and fitness_children is "coupled"
    sorted_mutated_population  = mutated_population[sorted_fitness_children.index]
    parent_retain_ratio = 0.5
    parents_retained = np.floor(parent_retain_ratio * pop_size)

    selected_parents = intermediate_population.iloc[:,0:parents_retained.astype(int)]
    selected_children = sorted_mutated_population.iloc[:,0:(pop_size - parents_retained.astype(int))] #mutated_population
    new_generation = pd.concat([selected_parents,selected_children],axis = 1,ignore_index= True)
    new_fitness = pd.concat([sorted_fitness.loc[selected_parents.columns.array],sorted_fitness_children.iloc[0:(pop_size - parents_retained.astype(int))]],axis = 0,ignore_index = True) #Top 3 of tournament winners and top pop_size - 3 from the sorted mutated population
    sorted_new_generation = new_generation.iloc[::,new_fitness.sort_values().index]
    population  = sorted_new_generation
    population.columns = range(pop_size)
    fitness = new_fitness.sort_values(ignore_index=True)

for i in range(pop_size): 
#best_fitness_index = new_fitness.sort_values().index[0]

    best_fitness_index = i
    best_min =  {'x1' :b_to_d(population[best_fitness_index][0:int(len(new_generation.index)/2),],l_lim,u_lim),
                 'x2' : b_to_d(population[best_fitness_index][int(len(new_generation.index)/2):len(new_generation.index),],l_lim,u_lim)}
    print(best_min)
    print(fitness.loc[best_fitness_index])
    



    
