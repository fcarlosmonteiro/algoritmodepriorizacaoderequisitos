#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random

from random import randint
from deap import base
from deap import creator
from deap import tools


from random import sample
from functools import partial

import pandas as pd
bd = pd.read_csv('base.csv')

import matplotlib.pyplot as plt
import numpy as np

bd1 = bd.iloc[0:5,0:4]
bd2 = bd.iloc[5:10,0:4]
bd3 = bd.iloc[10:15,0:4]
bd4 = bd.iloc[15:20,0:4]

xglobal = []
yglobal = []
maxGlobal = 0
geracaoGlobal = 0
gglobal = 0
gglobalTemp = 0

####### Variaveis Importantes #######

# Populacao Total
populacao = 50
# Probabilidade De Um Individuo Sofrer Mutacao
probmut = 0.5
# Probabilidade De Dois Individuos Cruzarem
probcross = 0.7
# Quantidade de Geracoes
numgeracoes = 500

#####################################

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def getId():
    v = randint(1,100)
    return v

def validaFilho(vetor):
    for i, e in enumerate(vetor):
        for j, f in enumerate(vetor):
                if i is j:
                    pass
                elif e == f:
                    while e == f:
                        tmp = 0
                        f = randint(1,20)
                        for g2 in vetor:
                            if f == g2:
                                tmp = 1
                            if tmp == 1:
                                f = e
                    vetor[i] = f
    print vetor
    return vetor

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)

gen_idx = partial(sample, range(100), 5)

toolbox.register("inputs", gen_idx)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initIterate, creator.Individual, 
    toolbox.inputs)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalOneMax(individual):

    #print individual

    dado1 = bd.iloc[individual[0],0:4]
    dado2 = bd.iloc[individual[1],0:4]
    dado3 = bd.iloc[individual[2],0:4]
    dado4 = bd.iloc[individual[3],0:4]
    dado5 = bd.iloc[individual[4],0:4]

    grau_depen = []
    grau_depen.append(dado1.loc['Grau de dependencia'])
    grau_depen.append(dado2.loc['Grau de dependencia'])
    grau_depen.append(dado3.loc['Grau de dependencia'])
    grau_depen.append(dado4.loc['Grau de dependencia'])
    grau_depen.append(dado5.loc['Grau de dependencia'])

    gpd = 0
    for g in grau_depen:
        if pd.isnull(g):
            pass
        else:
            gpd += g

    gpd = gpd/5
    gpd = gpd * 0.2 # 0.1, 0.2, 0.3

    funcao = dado1.loc['Prioridade'] + dado2.loc['Prioridade'] + dado3.loc['Prioridade'] + dado4.loc['Prioridade'] + dado5.loc['Prioridade']
    funcao = funcao - gpd
    #print funcao

    return funcao,

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

#----------

def main():

    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=populacao)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = probcross, probmut
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while max(fits) < 25 and g < numgeracoes:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        gglobalTemp = g
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                child1 = validaFilho(child1)                        
                child2 = validaFilho(child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        if max(fits) > maxGlobal:
            global maxGlobal
            global gglobal
            maxGlobal = max(fits)
            gglobal = gglobalTemp

        xglobal.append(max(fits))
        yglobal.append(g)
    
    print("-- End of (successful) evolution --")
    print("Chupa essa mangaaaaaaaa!")
    
    best_ind = tools.selBest(pop, 1)[0]

    fig, ax = plt.subplots()
    ax.plot(xglobal, yglobal, 'b--')
    style = dict(size=10, color='gray')
    plt.scatter(xglobal, yglobal, color='green')
    txtTemp = '{} individuos\nMelhor solucao: {}\nGeracao {}\n{}'.format(populacao, best_ind.fitness.values[0], gglobal, best_ind)
    ax.annotate(txtTemp, xy=(best_ind.fitness.values[0], gglobal), xytext=(best_ind.fitness.values[0]-3, gglobal+200),arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel('Funcao')
    plt.ylabel('Geracao')
    plt.show()

    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    
if __name__ == "__main__":
    main()