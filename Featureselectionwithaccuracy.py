#Feature selection using GeneticAlgorithms(Using randomForest)
import numpy as np
import pandas as pd
#Fuction definitions:
#1.Fitness Calc
def rf_fitness(X,y,k):
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(criterion = 'gini',max_depth = None , n_estimators = 100,max_features =  0.01, random_state = 42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(Y_test,y_pred)
    #acc = accuracy_score(Y_test,y_pred)
    #print("MCC : ",mcc)
    #print("Accuracy :",acc)
    return(acc - k * np.floor(len(X.columns)/7129))
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
    number_of_features = len(cross_over_population.index)
    P = 1 /pop_size
     #The probability of uniform random number between 0 and 1 having value less than a certain x is x. SO if a random unifomr number comes less than P when its probability is P!
    mutated_population = pd.DataFrame(index =range(len(cross_over_population.index)), columns = range(pop_size),dtype = "float64")
    for i in range(pop_size):
        #for j in cross_over_population.index:
        mutated_population[i] = cross_over_populationf.iloc[:,i].where(np.random.uniform(0,1,len(cross_over_populationf.index))< P ,np.logical_not)
        #print(mutated_population)
    return(mutated_population)
###########

#Initialize data
data = pd.read_csv("leukemia.csv",header = None)
from sklearn.preprocessing import scale, LabelEncoder
X = scale(data.iloc[:,:-1])
X = pd.DataFrame(X)
Y = data.iloc[:,-1]
label = LabelEncoder()
label.fit(Y)
Y = label.transform(Y)
number_of_features = len(X.columns)
pop_size = 10

#Initialize for GA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
best_param = {'criterion': 'gini', 'max_depth': None, 'max_features': 0.01, 'n_estimators': 100}
scorer = make_scorer(accuracy_score)
start_pop = pd.DataFrame()
for i in range(pop_size):
    start_pop[i] = np.random.choice(2,number_of_features)


epochs = 2
weights = [0.1,0.2,0.5,0.7,0.9]
#weights = [0.2]
acc_cv = []
no_attr = []
#weights = [0.5]
for tune_para in weights:
    population = pd.DataFrame()
    fitness = []
    X_current = pd.DataFrame()
    for i in range(pop_size):
        #population[i] =  np.random.choice(2,number_of_features)
        population[i] = start_pop[i]
        X_current = X.loc[:,population[i] == 1] #use population as on/off switch along each col to select features for accuracy evaluation
        fitness.append( rf_fitness(X_current,Y,tune_para))#fitness is a pandas series containing the accuracy values for the different chromosomes in the population
    fitness = pd.Series(fitness) # The index of fitness is the index of the population with that fitness value

    for le in range(epochs):
        #Tournament Selection
        sorted_fitness = fitness.sort_values(ascending= False)


        prob = sorted_fitness / sorted_fitness.sum()

        tournament_result = np.random.choice(sorted_fitness.index,size = pop_size, p = prob ,replace= True) #This has the indices of the population which are selected for crossover. This is the result of the tournament of size  = pop_size.

        intermediate_population = population[tournament_result]

        #X_intermediate = X.iloc[:,intermediate_population.iloc[7]] To get the X df corresponding to the current chromosome

        #Crossover

        #First create pairs for crossover
        pairs = np.random.choice(pop_size,pop_size,replace=False)
        #for i in range(1,pop_size,2):
         #  crossover_population = swap_at_k( intermediate_population[i] intermediate_population[i-1],k)
        #>>> (population[1] == population[0]).sum()

        shuffled_population = intermediate_population.iloc[::,pairs]
        cross_over_population = pd.DataFrame()
        for i in range(1,pop_size,2):
            crossver_list_temp = swap_at_k(shuffled_population.iloc[::,i-1],shuffled_population.iloc[::,i],np.random.randint(0,len(population[i-1])))
            cross_over_population[i-1] = crossver_list_temp[0]
            cross_over_population[i] = crossver_list_temp[1]
            #basic two point crossover. Function defined above.

        #Mutation. Each bit in every population has a probability of being mutated. but that number is very small. P = (1/(np.log(pop_size))) * (np.power(10,-2))


        mutated_population = mutation(cross_over_population)
        mutated_population = mutated_population.astype(int)

        #Evaluate fitness for mutated_population
        fitness_children = []
        for i in mutated_population.columns:
            fitness_children.append(rf_fitness(X.loc[:,mutated_population[i].astype(bool)],Y,tune_para))
        fitness_children = pd.Series(fitness_children)
        sorted_fitness_children = fitness_children.sort_values(ascending= False)# i of mutated_population and fitness_children is "coupled"
        sorted_mutated_population  = mutated_population[sorted_fitness_children.index]
        parent_retain_ratio = 0.3
        parents_retained = np.floor(parent_retain_ratio * pop_size)

        selected_parents = intermediate_population.iloc[:,0:parents_retained.astype(int)]
        selected_children = sorted_mutated_population.iloc[:,0:(pop_size - parents_retained.astype(int))] #mutated_population
        new_generation = pd.concat([selected_parents,selected_children],axis = 1,ignore_index= True)
        new_fitness = pd.concat([sorted_fitness.loc[selected_parents.columns.array],sorted_fitness_children.iloc[0:(pop_size - parents_retained.astype(int))]],axis = 0,ignore_index = True) #Top 3 of tournament winners and top pop_size - 3 from the sorted mutated population
        sorted_new_generation = new_generation.iloc[::,new_fitness.sort_values(ascending = False)]
        population  = sorted_new_generation
        population.columns = range(pop_size)
        fitness = new_fitness.sort_values(ascending = False)
    best_X = X.loc[:,sorted_new_generation.iloc[:,0] == 1 ]
    print(best_X)
    #Using the best X in each iteration calculate the accuracy values
    
    X_train,X_test,Y_train,Y_test = train_test_split(best_X,Y,test_size = 0.2)
    #best param for the classifier used (RF) are {'criterion': 'gini', 'max_depth': None, 'max_features': 0.01, 'n_estimators': 100}
    rfc = RandomForestClassifier(criterion = 'gini',max_depth = None,n_estimators = 100,random_state = 42)
    acc_cv.append(cross_val_score(rfc,best_X,Y,cv=5,scoring = scorer).mean()) #the indices of weights and acc_cv are coupled
    no_attr.append(len(best_X.columns))
    
    
    





    
