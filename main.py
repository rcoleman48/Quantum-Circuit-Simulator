# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:04:26 2021

@author: Robin Coleman

This code can simulate a quantum circuit for any N amount of qubits
There are 3 possible gates that are currently implemented
including the parametric general single qubit rotation gate
The circuit will be run over a set multi-shot amount
for each shot the final state will be selected using the weighted probabilities from the final state vector

"""
import numpy as np
from numpy.random import choice
import sympy as sym

#defining symbols to evaluate for parametric gates
#NOTE using lam for lambda since lambda is already a keyword
theta, phi, lam = sym.symbols('theta phi lam')
#Defining the parametric gate here so that I can use lambdify in order to convert it to a numpy array 
u3 = sym.Array([[sym.cos(theta/2),-sym.exp(1j*lam)*sym.sin(theta/2)],[sym.exp(1j*phi)*sym.sin(theta/2),sym.exp(1j*lam+1j*phi)*sym.cos(theta/2)]])
s = (theta, phi, lam)
u3_func = sym.lambdify(s,u3, modules='numpy')

#Now creating the gates dictionary that can be used in the simulator
#Gates are h:Hadamard, cx:CNOT, u3:General single qubit rotation gate
gates = [
    {"gate": "h", "unitary": (1/np.sqrt(2))*np.array([[1,1],[1,-1]])},
    {"gate": "cx", "unitary": np.array([[0,1],[1,0]])},
    {"gate": "u3", "unitary": u3_func}
     ]

def get_ground_state(num_qubits):
    # return vector of size 2**num_qubits with all zeroes except first element which is 1
    ground = np.zeros(2**num_qubits)
    ground[0]=1
    return ground

def get_operator(total_qubits, gate_unitary, target_qubits):
    # return unitary operator of size 2**n x 2**n for given gate and target qubits
    #First create 2x2 matricies to use, P0x0 is |0><0| projection, P1x1 is same for 1 state and I is identity
    P0x0 = np.array([[1, 0],[0, 0]])
    P1x1 = np.array([[0, 0],[0, 1]])
    I = np.identity(2)
    #I don't know if there is a more general way to do it so I have it broken up if it is a one or two qubit gate
    #This is for one qubit gates
    if len(target_qubits)==1:
        #With the kron we have special cases if the target is one of the first two qubits otherwise we can generalize
        if target_qubits[0]==0:
            O = np.kron(gate_unitary,I)
        elif target_qubits[0]==1:
            O = np.kron(I,gate_unitary)
        else:
            O = np.kron(I,I)
        for i in range(2,total_qubits):
            if target_qubits[0]==i:
                O = np.kron(O,gate_unitary)
            else:
                O = np.kron(O,I)
    #Now the two-qubit gates
    elif len(target_qubits)==2:
        #Now we have 7 initial cases that we need special syntax if any targets are in the first two qubits.
        if target_qubits[0]==0:
            O1 = np.kron(P0x0,I)
            if target_qubits[1]==1:
                O2 = np.kron(P1x1,gate_unitary)
            else:
                O2 = np.kron(P1x1,I)
        elif target_qubits[0]==1:
            O1 = np.kron(I,P0x0)
            if target_qubits[1]==0:
                O2 = np.kron(gate_unitary,P1x1)
            else:
                O2 = np.kron(I,P1x1)
        else:
            O1 = np.kron(I,I)
            if target_qubits[1]==0:
                O2 = np.kron(gate_unitary,I)
            elif target_qubits[1]==1:
                O2 = np.kron(I,gate_unitary)
            else:
                O2 = np.kron(I,I)
        for i in range(2,total_qubits):
            if target_qubits[0]==i:
                O1 = np.kron(O1,P0x0)
                O2 = np.kron(O2,P1x1)
            else:
                O1 = np.kron(O1,I)
                if target_qubits[1]==i:
                    O2 = np.kron(O2,gate_unitary)
                else:
                    O2 = np.kron(O2,I)
        O = O1+O2
    #O1 and O2 are just the two parts of the whole operator for the case of a 2 qubit gate
    return O

def run_program(initial_state, program):
    # read program, and for each gate:
    #   - calculate matrix operator
    #   - multiply state with operator
    # return final state
    
    
    state = initial_state
    #numQ is just the number of qubits measured by taking log2 of the length of the state vector
    numQ =int(np.log2(len(initial_state)))
    #We will just loop over the gates in the program and update state each time
    for i in program:
        for j in gates: 
            #since we don't have many gates it's fine to loop over all the gates
            #With more gates it might be worth putting them in a set and getting the intersection
            if i.get("gate") == j.get("gate"):
                #Parametric gates are split from the non-parametric ones by checking if there are parameters defined
                if "params" in i:
                    #Since we used lambdify we can evaluate the expression here
                    expr = j.get("unitary")
                    state = np.dot(state,get_operator(numQ, expr(i.get("params").get("theta"),i.get("params").get("phi"),i.get("params").get("lambda")), i.get("target")))
                else:
                    #If it's not a parametric gate things get a bit easier
                    state= np.dot(state,get_operator(numQ, j.get("unitary"), i.get("target")))
    #Return final state vector after looping over all the gates
    return state

def measure_all(state_vector):
    # choose element from state_vector using weighted random and return it's index
    prob = np.abs(state_vector)**2
    #prob is just a vector of the state squared, this can then be passed to choice as a vector of the probabilities
    element = choice(range(len(state_vector)), p=prob)
    #element just returns the index of the state vector that is returned
    return element

def get_counts(state_vector, num_shots):
    # simply execute measure_all in a loop num_shots times and
    # return object with statistics in following form:
    #   {
    #      element_index: number_of_ocurrences,
    #      element_index: number_of_ocurrences,
    #      element_index: number_of_ocurrences,
    #      ...
    #   }
    # (only for elements which occoured - returned from measure_all)
    numQ =int(np.log2(len(state_vector)))
    # We need numQ for formatting and we create occur to store each measurement of the states
    occur = np.zeros(len(state_vector))
    #occur is a vector that holds all the occurences that we get over the multi-shot
    for i in range(num_shots):
        shot = measure_all(state_vector)
        occur[shot] += 1
    bits = {}
    #We are then taking the occur vector and putting those states into a dictionary 
    #the indices of the vector are translated to a qubit state
    #NOTE The current encoding used is little endian
    for i in range(len(state_vector)):
        bit = bin(i)[2:].zfill(numQ)
        #Here is just some formatting to make the string for the key only the length of the number of qubits
        bits.update({bit: int(occur[i])})
    return bits

'''
Below is some sample code that shows how to run the code
There is an example circuit using one instance of each of the 3 gates coded

'''
my_circuit = [
{ "gate": "h", "target": [1] }, 
{ "gate": "cx", "target": [1, 2] },
{ "gate": "u3", "params": {"theta": 3.1415, "phi": 1.5708, "lambda": -3.1415 }, "target": [2]}
]


# Create "quantum computer" with 2 qubits (this is actually just a vector :) )

my_qpu = get_ground_state(3)


# Run circuit

final_state = run_program(my_qpu, my_circuit)


# Read results

counts = get_counts(final_state, 5000)

print(counts)