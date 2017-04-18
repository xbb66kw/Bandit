
import numpy as np
from Groups import nodes_generator
from Groups import Groups
from Groups import Graph
from Groups import Distribution
from Groups import Environment
								
												
##Test

M = np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]])#half-triangular
M = M + M.T
nodes = nodes_generator(M, values = [[10.00001],[],[10],[3.01]])
nodes = nodes_generator(M)
for elem in nodes:
    pass#print(elem.values)
g = Graph(nodes)


#print(g.center_value(g.picked()))
#print(g.edge_matrix())
#print(g.argmax_within_keys([0,1]))

G = Groups(g)
G.update(0,3,0)

print(G.argmax_groups())
print(G.argmax_within_group(0))
dist = Distribution()
E = Environment(dist, G)
E.run(20)