# Librerias que necesitamos importar
from __future__ import annotations
from sympy import symbols, And, Xor, Or
from sympy.logic.boolalg import Boolean, to_anf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pennylane as qml
import networkx as nx
import copy
import itertools as it

    #----------------------------------- Propagators operations -----------------------------------#

def make_edges(num_edges: int) -> list_edges: 
    """
        Se crea una lista con el número de edges 
    """
    
    edges = [symbols("S_{}".format(edge), Boolean = True) for edge in range(num_edges)]
    
    return edges


def make_prime_edges(edges: List) -> Prime_edges: 
    """
        Dada la lista de edges, esta regresa la lista de los edges primos
    
    """
    
    prime_edges = [symbols("S^'_{}".format(num_edges), Boolean = True) for num_edges in range(len(edges))]
    
    return prime_edges


def make_set_edges(edges, prime_edges) -> List: 
    """
        Esta función crea la lista con los conjuntos de los edges y prime edges
    
    """
    set_edges = [{edges[id_edge], prime_edges[id_edge]} for id_edge in range(len(edges))]
    
    return set_edges

def edges_to_propagators(edges_list: edge_oriented_positive, num_propagator_edge: int_or_list) -> dict: 

    num_edges = len(edges_list)
    if isinstance(num_propagator_edge, int):
        num_propagator_edge = [num_propagator_edge] * num_edges
    
    num_propagators = sum(num_propagator_edge)
    e = [f'e{num_prop}' for num_prop in range(num_propagators)]
    
    propagator_per_edge = {}
    _ = 0
    for edge, num in zip(edges_list, num_propagator_edge):
        propagator_per_edge[edge] = e[_:_ + num]
        _ += num

    return propagator_per_edge, e

   #----------------------------------- Clauses operations -----------------------------------#


def remove_subexpressions(expr: LogicExpr, set_edges: List[set]) -> Logic_expr:
    """
       Esta función se encarga de remover los terminos que sean de la forma 
       A y A', las cuales sabemos que son falsos
       
    """
    
    new_expr = list(expr.args)
    for sub_expr in expr.args:
        if any(edge.issubset(set(sub_expr.args)) for edge in set_edges):
            new_expr.remove(sub_expr)
    return Xor(*new_expr)

def not_subexpressions(expr: LogicExpr, set_edges: List[set]) -> Logic_expr:
    """
       Esta función se encarga de remover los terminos que sean de la forma 
       A y A', las cuales sabemos que son falsos
       
    """
    
    return not any(edge.issubset(expr.args) for edge in set_edges)
            
            


def transform_expression(expr: LogicExpr, set_edges: List[set]) -> Logic_expr:
    """
        Esta función se encarga de convertir la expresión lógica a una expresión ANF, y 
        posteriormente remover los términos que sean de la forma (A and A').
    
    """
    
    return remove_subexpressions(to_anf(expr), set_edges)




def acyclic_terms(Logic_expr, variables, qfix = None):
    """
       Esta función convierte nuestra proposición lógica en una tabla de verdad 
       y se encarga de devolver en una lista los expresiones que son falsas.
       Esto se traduce en nuestro contexto a diagramas aciclicos
    
    """

    if qfix is not None:
        variables = [True if var == qfix else var for var in variables]
    
    table = np.array(list(truth_table(logic_expr, variables)), dtype='object')
    
    return [row[0] for row in table if not row[1]]


def reverse_clauses(clauses: List_expr, S: Edge, Sp: prime_edge) -> List:
    """
       Makes the reverse of a set of clauses. We are making the asumtion that S_0 has a fix direction. 
       These reverse clauses are attached to the original list.
    """

    reverse_clauses_list = [clause for clause in clauses if S[0] not in clause.args]

    for i, clause in enumerate(reverse_clauses_list):
        reverse_edge = []
        for edge in clause.args:
            reverse_edge.append(Sp[S.index(edge)] if edge in S else S[Sp.index(edge)])
        reverse_clauses_list[i] = And(*reverse_edge)

    return reverse_clauses_list
        

def ext_clause(clauses,num_ext_edges):
    for i in range(len(clauses)):
        sep=separar(str(clauses[i]))    
        
        if str(sep)==str([f'{j}' for j in range (0, num_ext_edges)]):
            return i            
        #----------------------------------- Graphs operations -----------------------------------#


def indepent_clauses(clauses, set_edges, show_expression = False ) -> Boolean:
    """
       Esta función se encarga de comprobar si dos clausulas son independientes con la condición
       A or B = A xor B
       
    """
    
    num_clauses = len(clauses)
    
    xor_expression = transform_expression(Or(*clauses), set_edges)
    
    num_xor_clauses = len(xor_expression.args)
    
    if show_expression == True: 
        return xor_expression
    
    return num_clauses == num_xor_clauses


def dependent_clauses_depth(clauses, set_edges) -> Boolean:
    """
       Esta función se encarga de comprobar si dos clausulas son independientes con la condición
       A or B = A xor B
       
    """
    
    total_args = sum(len(clause.args) for clause in clauses)
    intersec_expression = And(*clauses)
    num_elements_intersec_expr = len(intersec_expression.args)

    return (num_elements_intersec_expr < total_args) and not_subexpressions(intersec_expression, set_edges)
    
def adjacent_matrix_clauses(clauses: list, set_edges: list):

    num_clauses = len(clauses)
    adjacent_matrix = np.zeros((num_clauses,num_clauses))
                
    for i in range(0,num_clauses):
        for j in range(0,num_clauses): 
            if i != j and indepent_clauses([clauses[i], clauses[j]], set_edges):
                adjacent_matrix[i, j] = 1
                
    return adjacent_matrix

def adjacent_matrix_depth(clauses: list, set_edges: list):

    num_clauses = len(clauses)
    adjacent_matrix = np.zeros((num_clauses,num_clauses))
                
    for i in range(0,num_clauses):
        for j in range(0,num_clauses): 
            if i != j and dependent_clauses_depth([clauses[i], clauses[j]], set_edges):
                adjacent_matrix[i, j] = 1
                
    return adjacent_matrix
                
def graph_clauses(clauses: list, set_edges: list, draw = True, name_graph = None):
    
    adyacent_matrix = adjacent_matrix_clauses(clauses, set_edges)
        
    graph = nx.from_numpy_array(adyacent_matrix)
    
    if draw: 
        num_clauses = len(clauses)
        labels = [f'$c_{{{i}}}$' for i in range (0, num_clauses)]
        dict_labels = {i_edge:labels[i_edge] for i_edge in range (0,num_clauses)}
        
        nx.draw(graph, pos=nx.shell_layout(graph), labels = dict_labels,
                node_color='lightblue', node_size = 1400, width=2,font_size=28)
        
        if name_graph != None: 
            plt.savefig(f'{name_graph}.pdf',format="pdf",dpi=300,bbox_inches='tight',pad_inches=0.05)
            
        
    return graph

def graph_clauses_Rev(S,c,clauses: list, set_edges: list, draw = True, name_graph = None):
    
    adyacent_matrix = adjacent_matrix_clauses(clauses, set_edges)
        
    graph = nx.from_numpy_array(adyacent_matrix)
    rev=[]

    for i in range(0,len(c)):
        clause=clauses[i]
        if S[0] not in clause.args:
            rev.append(i)
    if draw: 
        num_clauses = len(clauses)
        labels = [f'$c_{{{i}}}$' for i in range (0, len(c))]
        for i in range (0, len(rev)):
            labels.append(f'$\\overline{{c}}_{{{rev[i]}}}$' )
        dict_labels = {i_edge:labels[i_edge] for i_edge in range (0,num_clauses)}

        if num_clauses >= 16:
            nx.draw(graph, pos=nx.shell_layout(graph), labels = dict_labels,
                node_color='lightblue', node_size = 1400, width=2,font_size=28)
        else:
            nx.draw(graph, pos=nx.shell_layout(graph), labels = dict_labels,
                node_color='lightblue', node_size = 1400, width=4,font_size=28)
        
        if name_graph != None: 
            plt.savefig(f'{name_graph}.pdf',format="pdf",dpi=300,bbox_inches='tight',pad_inches=0.05)
            
        
    return graph
 

def graph_depth(clauses: list, set_edges: list, draw = True,name_graph = None):
    
    adyacent_matrix = adjacent_matrix_depth(clauses, set_edges)
        
    graph = nx.from_numpy_array(adyacent_matrix)
    
    if draw: 
        num_clauses = len(clauses)
        labels = [f'$c_{{{i}}}$' for i in range (0, num_clauses)]
        dict_labels = {i_edge:labels[i_edge] for i_edge in range (0,num_clauses)}
        
        nx.draw(graph, pos=nx.shell_layout(graph), labels = dict_labels,
                node_color='lightgreen',  node_size = 1400, width=2,font_size=24)
        
        if name_graph != None: 
            plt.savefig(f'{name_graph}.pdf',format="pdf",dpi=300,bbox_inches='tight',pad_inches=0.05)
        
    return graph
    
    
def Graph_condition_combination(conditional_graph,clauses,S,c):

    """
    Dado un grado cualquiera se encarga de buscar las combinaciones dada una condición
    por medio de la busqueda de subgrafos completos que contiene el grafo con la condición. 
 
                     # MEJORAR EN UN FUTURO PARA BUSCAR EL MÍNIMO # 
    """
    rev=[]

    for i in range(0,len(c)):
            clause=clauses[i]
            if S[0] not in clause.args:
                rev.append(i)
    clauses_combination = []
    m=len(c)
    while conditional_graph.number_of_nodes() > 0:
        
        max_clique = max(nx.find_cliques(conditional_graph), key=len) # by lenght
        
        clauses_combination.append(max_clique)
        
        conditional_graph.remove_nodes_from(max_clique)


    clauses_combination4 = copy.deepcopy(clauses_combination)
    for i in range(len(clauses_combination4)):
        for j in range(len(clauses_combination4[i])):
            if clauses_combination4[i][j] >= m:
                clauses_combination4[i][j]= -rev[clauses_combination4[i][j]-m]
                        
    return clauses_combination,clauses_combination4     
    






def clauses_auxiliar(clauses, clauses_combination): 

    num_aux = len(clauses_combination)
    a = [f'a{num_aux}' for num_aux in range(num_aux)]
    clauses_per_auxiliar = {} 

    _ = 0 # Variable dummy
    for ids_indepent_clauses, auxiliar in zip(clauses_combination, a): 
        for id_clause in ids_indepent_clauses: 
            clauses_per_auxiliar[clauses[id_clause]] = auxiliar 

    return clauses_per_auxiliar, a


       #----------------------------------- Circuit operations -----------------------------------#
    
def Draw_circuit(circuit, wire_order_list, name_file: str, style = "black_white") -> plot: 
    qml.drawer.use_style("black_white")

    fig, ax = qml.draw_mpl(circuit, wire_order = wire_order_list, style ='black_white')()

    plt.savefig(f'{name_file}.pdf',format="pdf",dpi=300,bbox_inches='tight',pad_inches=0.05)

    plt.show()   

def Draw_circuit2(circuit, wire_order_list, name_file: str, style = "black_white") -> plot: 
    qml.drawer.use_style("black_white")
    plt.rcParams['patch.facecolor'] = 'ghostwhite'
    plt.rcParams['patch.edgecolor'] = 'black'
    plt.rcParams['text.color'] = 'blue'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['patch.linewidth'] = 5
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['lines.color'] = 'black'
    plt.rcParams['lines.linewidth'] = 7
    plt.rcParams['figure.facecolor'] = 'ghostwhite'
    fig, ax = qml.draw_mpl(circuit, wire_order = wire_order_list, style ='rcParams', fontsize=28)()
    
    plt.savefig(f'{name_file}.pdf',format="pdf",dpi=300,bbox_inches='tight',pad_inches=0.05)

    plt.show()      
        
    
def plot_histogram(circuit_counts: dict): 
    
    counts = circuit_counts

    plt.figure(figsize = (15,7))
    plt.bar(counts.keys(), counts.values(), color = [(96/255,137/255,1)], width=0.6);

    plt.xlabel('Configurations', fontsize = 16);
    plt.ylabel('Counts', fontsize = 16);
    plt.tick_params(labelsize=12)
    plt.grid(axis='y', linestyle='--')

    plt.xticks(rotation=70, fontsize = 11);




def get_depth(circuit): 
    
    depth = qml.specs(circuit)()['resources'].depth

    print(f'The depth of the circuit is: {depth}')
    

def num_casual_states(circuit_counts, cut = 10): 
    
    casual_states = sum(1 for count in circuit_counts.values() if count > cut)

    print(f'The number of casual states is: {casual_states}')

    
def Number_shots(n: number_propagators, sigma) -> int:
    
    """
    Calculate the aproximation of the required number of shots using the formula (4.11) from the 
    paper https://arxiv.org/pdf/2105.08703. 
    """

    r = 2**n
    
    return r * (sigma)**2 




def clause_to_elements_multitoffoli(clause: Logic_expr, edges_to_propag: dict, clauses_to_aux: dict, S, Sp: list):
        
        control = []
        target = clauses_to_aux[clause]
        PauliX_propagators = []

        for edge in clause.args: 
            if edge in Sp: 
                edge = S[Sp.index(edge)]
                PauliX_propagators.append(edge)
            
            control.extend([propagator for propagator in edges_to_propag[edge] if propagator != 'e0'])

        return control + [target], PauliX_propagators


def oracle(clauses: Logic_expr, edges_to_propag: dict, clauses_to_aux: dict, S, Sp: list, depth_combination): 
    
    last_xgates_order = set()
    
    for block_depth in depth_combination:
        gates_wires_seq = []
        x_gates_order = set()

        for id_clause in block_depth:
            wires, x_gates = clause_to_elements_multitoffoli(clauses[id_clause], edges_to_propag, clauses_to_aux, S, Sp)
            gates_wires_seq.append(wires)
            x_gates_order.update(x_gates)

        x_gates_propagators = [prog for edge in x_gates_order for prog in edges_to_propag[edge]]

        qml.broadcast(qml.PauliX, x_gates_propagators, 'single')

        for elements_wires in gates_wires_seq:
            qml.MultiControlledX(wires=elements_wires)

        qml.broadcast(qml.PauliX, x_gates_propagators, 'single')


ignorar = ["^", "&"]
items=[]
def separar(algo: str):
    for cosa in ignorar:
        algo = algo.replace(cosa, "")
    items = algo.split("  ")
    for i in range(len(items)):
        items[i] = items[i].split("_")
        wx=[num[1] for num in items]
    return wx



# Initialize an empty set to store seen elements


# List to store duplicates

def norep(a:str):
    cop = []
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if a[i]==a[j] and a[i] not in cop:
                cop.append(a[i])
    if cop == []:
        reps=True
    else:
        reps=False
    return reps

def rep(a:str):
    cop = []
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if a[i]==a[j] and a[i] not in cop:
                cop.append(a[i])
    if cop != []:
        reps=True
    else:
        reps=False
    return reps


def truelist(clauses,i,j):
    yz=separar(str((clauses[i])))
    xz=separar(str((clauses[j])))

    for i in xz:
        yz.append(i)
    return yz

def truelist2(clauses,a):
    yz=separar(str((clauses[a[0]])))
    xz=separar(str((clauses[a[1]])))

    for i in xz:
        yz.append(i)
    return yz

def adjacent_matrix_depth2(clauses: list, set_edges: list):

    num_clauses = len(clauses)
    adjacent_matrix2 = np.zeros((num_clauses,num_clauses))
                
    for i in range(0,num_clauses):
        for j in range(0,num_clauses): 
            if i != j and And(dependent_clauses_depth([clauses[i], clauses[j]], set_edges), rep(truelist(clauses,i,j))):
                adjacent_matrix2[i, j] = 1
    return adjacent_matrix2

def adjacent_matrix_depth3(clauses: list, set_edges: list):

    num_clauses = len(clauses)
    adjacent_matrix3 = np.zeros((num_clauses,num_clauses))
                
    for i in range(0,num_clauses):
        for j in range(0,num_clauses): 
            if i != j and Or(dependent_clauses_depth([clauses[i], clauses[j]], set_edges), norep(truelist(clauses,i,j))):
                adjacent_matrix3[i, j] = 1
    return adjacent_matrix3


    

def adjacent_matrix_depth3(clauses: list, set_edges: list):

    num_clauses = len(clauses)
    adjacent_matrix = np.zeros((num_clauses,num_clauses))
                
    for i in range(0,num_clauses):
        for j in range(0,num_clauses): 
            if i != j and Or(dependent_clauses_depth([clauses[i], clauses[j]], set_edges), norep(truelist(clauses,i,j))):
                adjacent_matrix[i, j] = 1
                
    return adjacent_matrix


def graph_depth3(clauses: list,num_ext_edges, set_edges: list, draw = True,name_graph = None):
    n=ext_clause(clauses,num_ext_edges)
    adyacent_matrix = adjacent_matrix_depth3(clauses, set_edges)
    tr=np.array(adyacent_matrix)
    tr=np.delete(tr,n,axis=0)
    tr=np.delete(tr,n,axis=1)    
    graph = nx.from_numpy_array(tr)
    
    

    if draw: 
        num_clauses = len(clauses)-1
        labels = [f'$c_{{{i}}}$' for i in range (0, n)  ]
        labels2=[f'$c_{{{i+1}}}$' for i in range (n, num_clauses)  ]
        labelsf=np.concatenate((labels,labels2))
       
        dict_labels = {i_edge:labelsf[i_edge] for i_edge in range (0,num_clauses)}
        if num_clauses >= 16:
            nx.draw(graph, pos=nx.shell_layout(graph), labels = dict_labels,
                node_color='lightgreen', node_size = 1400, width=2,font_size=28)
        else:
            nx.draw(graph, pos=nx.shell_layout(graph), labels = dict_labels,
                node_color='lightgreen', node_size = 1400, width=4,font_size=28)

        
        if name_graph != None: 
            plt.savefig(f'{name_graph}.pdf',format="pdf",dpi=300,bbox_inches='tight',pad_inches=0.05)
        
    return graph
    
    


def Graph_condition_combination3(conditional_graph3,clauses,num_ext_edges,S,c):

    """
    Dado un grado cualquiera se encarga de buscar las combinaciones dada una condición
    por medio de la busqueda de subgrafos completos que contiene el grafo con la condición. 
 
                     # MEJORAR EN UN FUTURO PARA BUSCAR EL MÍNIMO # 
    """
    n=ext_clause(clauses,num_ext_edges)
    m=len(c)
    clauses_combination3 = []
    clauses_combination4 = []
    rev=[]

    for i in range(0,len(c)):
            clause=clauses[i]
            if S[0] not in clause.args:
                rev.append(i) 
    
    while conditional_graph3.number_of_nodes() > 0:
        
        max_clique = max(nx.find_cliques(conditional_graph3), key=len) # by lenght
        
        clauses_combination3.append(max_clique)
        
        conditional_graph3.remove_nodes_from(max_clique)
    if len(clauses)>4:   
        for i in range(len(clauses_combination3)):
            for j in range(len(clauses_combination3[i])):
                if clauses_combination3[i][j] >= n:
                    clauses_combination3[i][j]=clauses_combination3[i][j]+1
        clauses_combination3=[[n]]+clauses_combination3
        
        clauses_combination4 = copy.deepcopy(clauses_combination3)
        for i in range(len(clauses_combination4)):
            for j in range(len(clauses_combination4[i])):
                if clauses_combination4[i][j] >= m:
                    clauses_combination4[i][j]= -rev[clauses_combination4[i][j]-m]
                    
    
    return clauses_combination3, clauses_combination4




        



def graph_depthC(S,c,clauses: list,num_ext_edges, set_edges: list, draw = True,name_graph = None):
    graph_depthx=graph_depth3(clauses,num_ext_edges ,set_edges,draw=False)
    depth_combination3 = Graph_condition_combination3(graph_depthx,clauses, num_ext_edges,S,c)
    rev=[]
    n=ext_clause(clauses,num_ext_edges)
    amd=adjacent_matrix_depth3(clauses,set_edges)
    k=2
    for i in range(0,amd.shape[0]) :
        if amd[i][n]==1:
            amd[i][n]=0.5
        if amd[n][i]==1:
            amd[n][i]=0.5    
        
        if i<len(depth_combination3[0]):
            perm=list(it.permutations(depth_combination3[0][i]))
            
            if len(perm)>1:
                for j in range(0,len(perm)):
                    amd[perm[j][0]][perm[j][1]]=k
                    amd[perm[j][1]][perm[j][0]]=k
                k=k+1      
    graph = nx.from_numpy_array(np.matrix(amd),create_using=nx.MultiGraph,parallel_edges=False,edge_attr='weight')
    e=[]

    for i in range(0,len(c)):
        clause=clauses[i]
        if S[0] not in clause.args:
            rev.append(i)    
    if draw: 
        num_clauses = len(clauses)
        labels = [f'$c_{{{i}}}$' for i in range (0, len(c))]
        for i in range (0, len(rev)):
            labels.append(f'$\\overline{{c}}_{{{rev[i]}}}$' )
        dict_labels = {i_edge:labels[i_edge] for i_edge in range (0,num_clauses)}
        eo = [(u, v) for (u, v, d) in graph.edges(data=True) if  d["weight"]   ==1]
        ext_edge = [(u, v) for (u, v, d) in graph.edges(data=True) if  d["weight"] <= 0.5]
        pos=nx.shell_layout(graph)

        if num_clauses >= 23:
            font=16
            node=400
            w=2
            wo=0.2
            we=0.1
        elif 23>=num_clauses >= 16:
            font=22
            node=700
            w=3
            wo=0.5
            we=0.2    
        else:    
            font=28
            node=1400
            w=4
            wo=2
            we=1
        nx.draw_networkx_nodes(graph, pos, node_size=node, node_color='lightgreen')
        nx.draw_networkx_labels(graph,pos, labels= {n:lab for n,lab in dict_labels.items() if n in pos},font_size=font)
        # edgesS
        nx.draw_networkx_edges(graph, pos, edgelist=ext_edge,alpha=0.3 ,width=we, style="dashed")
        nx.draw_networkx_edges(graph, pos, edgelist=eo, width=wo, alpha=0.7, edge_color="black", style="dashed")
       
        for i in range(0,k-1):    
            e.append([(u, v) for (u, v, d) in graph.edges(data=True) if  d["weight"]   ==i+2])
            nx.draw_networkx_edges(graph, pos, edgelist=e[i], width=w, alpha=1, edge_color=cl[i], style="solid") 

        ax = plt.gca()
        ax.margins(0.08)
        #plt.figure(facecolor="none")
        #plt.axis("off")
        ax.set_facecolor("white")
        ax.axis("off")


        plt.tight_layout()
        if name_graph != None: 
            plt.savefig(f'{name_graph}.pdf',format="pdf",dpi=300,bbox_inches='tight',pad_inches=0.05)
        
    return graph,depth_combination3[1]

cl=["r","b","#E0D317","#195C0B","m","c","orange","brown","olive"]





def graph_clauses_C(S,c,clauses: list, set_edges: list, draw = True, name_graph = None):
    
    amd = adjacent_matrix_clauses(clauses, set_edges)
        
    
    rev=[]
    e=[]
    k=2

    graph_clausesx=graph_clauses(clauses,set_edges, draw =False)
    Graph_condition_combinationx=Graph_condition_combination(graph_clausesx,clauses,S,c)

    for i in range(0,amd.shape[0]) :

        if i<len(Graph_condition_combinationx[0]):
            perm=list(it.permutations(Graph_condition_combinationx[0][i]))
            
            if len(perm)>1:
                for j in range(0,len(perm)):
                    amd[perm[j][0]][perm[j][1]]=k
                    amd[perm[j][1]][perm[j][0]]=k
                k=k+1

    for i in range(0,len(c)):
        clause=clauses[i]
        if S[0] not in clause.args:
            rev.append(i)


    graph = nx.from_numpy_array(amd)        
    if draw: 
        num_clauses = len(clauses)
        labels = [f'$c_{{{i}}}$' for i in range (0, len(c))]
        for i in range (0, len(rev)):
            labels.append(f'$\\overline{{c}}_{{{rev[i]}}}$' )
        dict_labels = {i_edge:labels[i_edge] for i_edge in range (0,num_clauses)}

        eo = [(u, v) for (u, v, d) in graph.edges(data=True) if  d["weight"]   ==1]
        ext_edge = [(u, v) for (u, v, d) in graph.edges(data=True) if  d["weight"] <= 0.5]
        pos=nx.shell_layout(graph)
        
        if num_clauses >= 23:
            font=16
            node=400
            w=2
            wo=0.2
            we=0.1
        elif 23>=num_clauses >= 16:
            font=22
            node=700
            w=3
            wo=0.5
            we=0.2    
        else:    
            font=28
            node=1400
            w=4
            wo=2
            we=1
        nx.draw_networkx_nodes(graph, pos, node_size=node, node_color='lightblue')
        nx.draw_networkx_labels(graph,pos, labels= {n:lab for n,lab in dict_labels.items() if n in pos},font_size=font)
        # edgesS
        nx.draw_networkx_edges(graph, pos, edgelist=ext_edge,alpha=0.3 ,width=we, style="dashed")
        nx.draw_networkx_edges(graph, pos, edgelist=eo, width=wo, alpha=0.7, edge_color="black", style="dashed")
        for i in range(0,k-1):    
            e.append([(u, v) for (u, v, d) in graph.edges(data=True) if  d["weight"]   ==i+2])
            nx.draw_networkx_edges(graph, pos, edgelist=e[i], width=w, alpha=1, edge_color=cl[i], style="solid") 
       

        ax = plt.gca()
        ax.margins(0.08)
        #plt.figure(facecolor="none")
        #plt.axis("off")
        ax.set_facecolor("white")
        ax.axis("off")



        if name_graph != None: 
            plt.savefig(f'{name_graph}.pdf',format="pdf",dpi=300,bbox_inches='tight',pad_inches=0.05)
            
        
    return graph,Graph_condition_combinationx[1]