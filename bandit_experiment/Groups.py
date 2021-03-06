
import numpy as np
import matplotlib.pyplot as plt


class Graph(object):
    
    def __init__(self, nodes):
        'nodes: list-like'
        self.__nodes = nodes
        self.__len = len(nodes)
        self.q = 0.8 # percentile lower bound that we care
        self.alpha = 0.25 # exploration coefficient
        self.rho = 0.9 # decay rate for estimation
        self.d = 1 # the maximum distant for estimation
        
        
        
        connections = np.zeros(self.__len * self.__len)\
            .reshape(self.__len, self.__len).astype(int)
            
        for i in range(self.__len):
            connections[self.__nodes[i].neighbors,i] = 1
         
        self.__edge_matrix = connections
        
    def edge_matrix(*arg):
        
        self = arg[0]
        
        if len(arg) == 1:
            return self.__edge_matrix
            
        elif len(arg) == 2:
            keys = arg[1]
            return self.__edge_matrix[np.ix_(keys, keys)]
        
    def picked(*arg):
        '''
        Return all nodes picked before in this graph. If keys are given, 
        return the specified nodes that has been picked.
        
        arg[0]: self
        arg[1]: array-like, keys in interests(optional)
        '''
        
        self = arg[0]
        
        if len(arg) == 1:
            return np.asarray([index for (index, node)\
                in enumerate(self.__nodes) if node.picked])
        elif len(arg) == 2:
            keys = arg[1]
            return np.asarray([index for (index, node)\
                in enumerate(self.__nodes) if node.picked and index in keys])
    
    def nodes(*arg):
        '''
        Return all nodes in this graph. If keys are given, return the 
        specified nodes.
        
        arg[0]: self
        arg[1]: array-like, keys in interests(optional)
        '''
        
        self = arg[0]
        
        if len(arg) == 1:
            return np.asarray([node for (index, node)\
                in enumerate(self.__nodes)])
        elif len(arg) == 2:
            keys = arg[1]
            return np.asarray([node for (index, node)\
                in enumerate(self.__nodes) if index in keys])
            
        return np.asarray([node for (index, node) in enumerate(self.__nodes)\
            if index in keys])  
    
    @property
    def length(self):
        return self.__len
   
    def argmax_within_keys(self, keys):
        '''
        keys: array-like. The keys of the specified groups, including both 
        picked and unpicked nodes.
        
        return the selected node, singleton
        '''
        
        labelled = self.picked(keys)
        unlabelled = np.setdiff1d(keys, labelled)
        
        values = np.zeros(len(keys))
        
        for i, key in enumerate(labelled):
            values[i] = self.__nodes[key].average\
                + self.alpha\
                    * np.sqrt(1 / (self.__nodes[key].length + 1))
        
        for i, key in enumerate(unlabelled):
            index = i + len(labelled)
            
            far_ = 0 # less than self.d
            sum_ = 0
            for p in range(self.d):
                mid_ = np.setdiff1d(np.intersect1d(keys, self.nodes_connect(key, p + 1)), unlabelled)
                if mid_.size:
                    far_ += 1
                    counts = 0
                    value = 0
                    for j in mid_:
                        counts += self.__nodes[j].length                        
                        value += np.sum(self.__nodes[j].values)                  
                    sum_ += np.power(self.rho, p) * value / counts
            
            inverse_deflator = 0
            for p in range(far_):
                inverse_deflator += np.power(self.rho, p)         
            deflator =  1 / inverse_deflator 
            
            values[index] = deflator * sum_ + self.alpha
            
        max_ = values.argmax()
        choice = -1
        
        if max_ >= len(labelled):
            choice = unlabelled[max_ - len(labelled)]
        else:
            choice = labelled[max_]
        
        return choice
        
    def nodes_connect(self, key, distant=1):
        M = self.__edge_matrix
        if distant > 1:
            for i in range(distant-1):
                M = np.dot(M, self.__edge_matrix)
            
            return np.setdiff1d(np.array([key for (key, connect) in enumerate(M[key,:]) if connect]), self._nodes_connect_distant(key, distant-1))
                
        else:
            return np.array([key for (key, connect) in enumerate(M[key,:]) if connect])
    
    def _nodes_connect_distant(self, key, distant):
        'Set up for nodes_connect'
        M = self.__edge_matrix
        sum_ = M#np.zeros(self.__len * self.__len).reshpae(self.__len, self.__len)
        
        if distant > 1:
            for i in range(distant-1):
                M = np.dot(M, self.__edge_matrix)
                sum_ += M
                
        return np.array([key for (key, connect) in enumerate(sum_[key,:]) if connect])
    
    
    def center_value(self, keys):
        '''
        Compute the clustering value.
        
        keys: array-like
        '''
        
        _keys = keys
        keys = _keys
        
        '''
        path step
        '''
        path = np.array([])
        original_path_value = 0
        
        while keys.size:
            optimal = keys[np.argmax(self._all_center_values(keys))]
            path = np.append(path, optimal)
            mid = np.append(optimal, self.__nodes[optimal].neighbors)
            keys = np.setdiff1d(keys, mid)
        
        path = path.astype(int)
        original_path_value = self._path_value(_keys, path)
        ave = np.mean(self._all_center_values(_keys))
        
        '''
        trim step
        '''
        trim_set = []
        path_values = np.zeros(len(path))
        idx = np.arange(1, len(path))\
            - np.tri(len(path), len(path) - 1, k=-1, dtype=bool)
        
        for i in range(len(path)):
            path_values[i] = self._path_value(_keys, path[idx[i,:]])
        
        
        
        if len(path) > 1:
            'Then we need trimming'
            trim_set = (path_values + ave) > original_path_value
            
            path = np.asarray([value for (index, value) in enumerate(path)\
                if not trim_set[index]])
        
        
        
        
        return self._path_value(_keys, path) / len(path)
    
    def _all_center_values(self, keys):
        '''
        Compute the value of each node in keys.
        
        keys: array-like
        '''
        len_ = len(keys)
        
        results = np.zeros(len_)
        
        for i, key in enumerate(keys):
            
            consider = [_ for _ in self.__nodes[key].neighbors if _ in keys\
                and self.__nodes[_].picked]
            
            
            
            values = np.append([self.__nodes[elem].average\
                for elem in consider], self.__nodes[key].average)
            
            
            results[i] = np.sort(values)[np.floor(len(values) * self.q)\
                .astype(int)] * np.log(np.e + len(values) * (1 - self.q))

        return results
    
    
       
    def _path_value(self, keys, path):
        '''
        Given the path and the keys where it belong to, _path_value compute 
        the total value resulfed from this keys, path.
        
        keys: array-like        
        '''
        len_ = len(path)
        
        results = np.zeros(len_)
        
        
        for i, step in enumerate(path):

            
            consider = np.asarray(np.append(step,
                [_ for _ in self.__nodes[step].neighbors\
                    if _ in keys and self.__nodes[_].picked])).astype(int)

            values = np.asarray([self.__nodes[elem].average\
                for elem in consider])
            
            results[i] = np.sort(values)[np.floor(len(values) * self.q)\
                .astype(int)] * np.log(np.e + len(values) * (1 - self.q))
                
            keys = np.setdiff1d(keys, consider)

        
        return np.sum(results)
        
    
class Nodes(object):
    
    
    def __init__(self, key, neighbors, values=[]):
        '''
        neighbors, values: array-like
        '''
        self.__key = key
        self.__neighbors = neighbors
        
        
        self.__values = values
        self.__count = len(values)
        
    
    @property
    def key(self):
        return self.__key
    
    @property
    def values(self):
        if self.__count > 0:
            return self.__values
        else:
            return None
    
    @property
    def average(self):
        if self.__count > 0:
            return np.mean(self.__values)
        else:
            return None        

    
    @property
    def neighbors(self):
        return self.__neighbors
    
    @property
    def length(self):
        return self.__count
    
    @property
    def picked(self):
        if self.__count > 0:
            return 1
        else:
            return 0
    
    def add(self, value):
        self.__values = np.append(self.__values, value)
        self.__count += 1

def nodes_generator(edges_matrix, values=None):
    '''
    M = np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,1,0]])
    M = M + M.T
    nodes_generator(M, values = [[0],[0],[0],[0]])
    
    values: list-like. Each element is the corresponding historical record
    of the nodes. Can be [].
    '''
    len_ = len(np.diag(edges_matrix))
    nodes = []
        
    if values is None:
        for i in range(len_):
            nodes = np.append(nodes, Nodes(i, [index for (index, edge)\
                in enumerate(edges_matrix[:,i]) if edge]))
            
    else:
        for i in range(len_):
            nodes = np.append(nodes, Nodes(i, [index for (index, edge)\
                in enumerate(edges_matrix[:,i]) if edge], values[i]))
    
    return nodes


class Groups(object):
    
    def __init__(self, graph):
        self.__graph = graph
        self.__group_list = [np.arange(graph.length)]
        self.__class_number = 1
        self.k = 1 # For the unknown
        
    def argmax_groups(self):
        if self.__class_number == 1:
            return 0
        else:
            results = np.zeros(self.__class_number)
            
            
            
            for i in range(self.__class_number)[1:]:
                if self.__group_list[i].size:
                    results[i] = self.__graph.center_value(self.__graph.picked(self.__group_list[i]))
                else:
                    results[i] = -100 # for example
                    
            
            
            if self.__group_list[0].size:
                results[0] = np.mean(results[1:]) + self.k * np.std(results[1:])
            else:
                results[0] = -100 # for example
            return np.argmax(results)
            

    def argmax_within_group(self, group):
        '''group: scalar'''
        
        group_list = self.__group_list
        
        
        if group == 0:
            len_ = len(group_list[group])
            
            return group_list[group][(np.random.random_integers(len_, size=1)[0] - 1)]
        else:
            return self.__graph.argmax_within_keys(group_list[group])

    def update(self, pick, value, group):
        '''
        value is added into pick
        
        pick: scalar
        value: scalar
        group: scalar
        
        return if there's newly created group
        '''
        
        self.__graph.nodes(np.array([pick]))[0].add(value)
        
        related_nodes = np.array([]).astype(int)
        
        for d in range(self.__graph.d):
            related_nodes = np.union1d(related_nodes,\
                self.__graph.nodes_connect(pick, d + 1))
        
        check_create = True
        if not self.__group_list[0].size:            
            check_create = False

        
        
        
        if group is not 0:            
            for index, class_list in enumerate(self.__group_list):
                if index != group and index != 0 and\
                    np.intersect1d(class_list, related_nodes):
                        
                    check_create = False
                    self.__group_list[group] = np.asarray(np.append(self.__group_list[group], class_list))
                    self.__group_list[index] = np.array([])                    
                    
        else:
            combined = []
            for index, class_list in enumerate(self.__group_list):
                if index != 0 and\
                    np.intersect1d(self.__group_list[index], related_nodes):
                        
                    check_create = False
                    combined = np.append(combined, index)
                    
            if combined:        
                
                head = combined.pop(0)
                
                self.__group_list[head] = np.asarray(np.append(self.__group_list[head], related_nodes))
                for i in combined:
                    self.__group_list[head] = np.asarray(np.append(self.__group_list[head], self.__group_list[i]))
                    self.__group_list[i] = np.array([])
                    
        if check_create:
            self.__group_list.append(np.asarray((np.append(related_nodes, pick))))
            
        
        self.__group_list[0] = np.setdiff1d(self.__group_list[0], np.append(related_nodes, pick))
        
        self.__class_number = len(self.__group_list)
        
        return check_create

class Distribution(object):
    def __init__(self):
        pass
    def play(self, key):
        return np.random.normal(0, 1, 1)[0]
        
        
class Environment(object):
    
    def __init__(self, distribution, groups):
        self.__distribution = distribution
        self.__groups = groups
        
    def run(self, T):
        groups = self.__groups
        distribution = self.__distribution
        played_path = np.array([])
        recorded_values = np.array([])
        
        for t in range(T):
            group = groups.argmax_groups()
            
            pick = groups.argmax_within_group(group)
            
            
            
            'Drawing'
            value = distribution.play(pick)
            
            played_path = np.append(played_path, pick)
            recorded_values = np.append(recorded_values, value)
            
            groups.update(pick, value, group)
        
        print(played_path)
        
        plt.plot(recorded_values / (np.arange(T) + 1))
        plt.show()
        
       	
