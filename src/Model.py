#Graph.py
import numpy as np
import networkx as nx

from HelperFunctions import Vector, Geometry

class Graph():
    
    #-----------------------------------
    #      C O N S T R U C T O R
    #-----------------------------------
    
    def __init__(self, nodal_coordinates, adj_list, weights, load_components=None, anchored_nodes=None, load_labels=None):
        
        self._nodal_coordinates = nodal_coordinates
        self._connectivity = self._set_connectivity(adj_list)
        
        if anchored_nodes==None:
            self._nodal_labels = [-1 for _ in range(len(self._nodal_coordinates))] 
        else:
            self._nodal_labels = [0 if i in anchored_nodes else 1 for i in range(len(self._nodal_coordinates))]
        self._vertices = self._set_coords(self._nodal_labels)
        self._update_vertices()
        self.dof = len(nodal_coordinates[0])

        self._weights = self._set_weights(weights)
        self._edges = self._set_edges()

        if load_components is not None:
            self._load_components = load_components
            self._anchored_nodes = anchored_nodes
            self._load_labels = load_labels
            self._loads = self._set_forces(self._anchored_nodes, self._load_labels)
    
    def __str__(self):
        return f"Graph Summary:\nNumber of nodes: {self.Order},\nNumber of edges: {self.Size}"
    
    #-----------------------------------
    #        G E T T E R S
    #-----------------------------------
    
    @property
    # The adjacency matrix as a numpy matrix
    def Adjacency(self):
        return self._connectivity_to_adj_matrix()

    @property
    # The incidence matrix as a numpy matrix
    def Incidence(self):
        return self._connectivity_to_inc_matrix()

    @property
    def Coords2D(self):
        coords_ = np.array(self.Vertices, dtype=float)
        
        if np.allclose(coords_[:, 2], 0):  # all z = 0
            # the coords are all in XY-plane
            return coords_[:, [0, 1]].tolist()
        elif np.allclose(coords_[:, 1], 0):  # all y = 0
            # the coords are all in XZ-plane
            return coords_[:, [0, 2]].tolist()
        elif np.allclose(coords_[:, 0], 0):  # all x = 0
            # the coords are all in the YZ-plane
            return coords_[:, [1, 2]].tolist()
        else:
            # the coords are in 3D-space
            return coords_.tolist()

    @property
    def EdgeNodePairs(self):
        return self._connectivity

    @property
    def EdgeObjs(self):
        return self._edges
    
    @property
    def AxialForces(self):
        return self._weights
    
    @property
    def FaceObjs(self):
        faces = []
        for i,cycle in enumerate(nx.cycle_basis(self.NetworkXGraph)):
            if self._cycle_orientation(cycle) == 1:
                cycle = list(reversed(cycle))
            faces.append(Face(cycle, i, self))

        return faces

    @property
    def Lengths(self):
        return [edge.Length for edge in self.EdgeObjs]
    
    @property
    def LoadObjs(self):
        return self._loads

    @property
    def MaxCompressionForce(self):
        return min(self.AxialForces)

    @property
    def MaxTensionForce(self):
        return max(self.AxialForces)

    @property
    def MaxLength(self):
        return max(self.Lengths)
    
    @property
    def MinCompressionForce(self):
        return max([f for f in self.AxialForces if f < 0])
    
    @property
    def MinTensionForce(self):
        return min([f for f in self.AxialForces if f > 0])

    @property
    def MinLength(self):
        return min(self.Lengths)

    @property
    def NetworkXGraph(self):
        G = nx.Graph()
        for i, [x,y,_] in enumerate(self.Vertices):
            G.add_node(i, pos=[x,y])
        for (u, v), w in zip(self._connectivity,self._weights):
            G.add_edge(u, v, weight=w)
        return G
    
    @property
    def Order(self):
        return len(self._vertices)
    
    @property
    def Size(self):
        return len(self._connectivity)
    
    @property
    def Vertices(self):
        return [v.Coords for v in self._vertices]

    @property
    def VertexObjs(self):
        return self._vertices

    @property
    def VertexCoords(self):
        return [vertex.Coords for vertex in self._vertices]

    @property
    def IsPlanar(self):
        for edge in self.EdgeObjs:
            s = edge.StartNode.Coords
            e = edge.EndNode.Coords
            adj_edges = set(edge.StartNode_AdjEdges() + edge.EndNode_AdjEdges())
            for other_edge in self.EdgeObjs:
                if other_edge.Id not in adj_edges:
                    s_other = other_edge.StartNode.Coords
                    e_other = other_edge.EndNode.Coords
                    check = Geometry.SegSegIntersection(s, e, s_other, e_other, True)
                    if check[0] == True:
                        return False
                else:
                    continue
        return True
    
    @property
    def PlanarityDegree(self):
        degree = 0
        # Remove duplicates and ensure (a,b) == (b,a)
        unique_pairs = [tuple(sorted(pair)) for pair in set(tuple(sorted(p)) for p in self.EdgeNodePairs)]

        for i in range(len(unique_pairs)):
            pair_1 = unique_pairs[i]
            edge_1 = self.EdgeByEndIds(list(pair_1))
            p1 = edge_1.StartNode
            p2 = edge_1.EndNode

            for j in range(i + 1, len(unique_pairs)):
                pair_2 = unique_pairs[j]
                edge_2 = self.EdgeByEndIds(list(pair_2))
                if edge_1.Id == edge_2.Id:
                    continue

                q1 = edge_2.StartNode
                q2 = edge_2.EndNode

                if (p1.Id == q1.Id or p1.Id == q2.Id or p2.Id == q1.Id or p2.Id == q2.Id):
                    continue

                check, _ = Geometry.SegSegIntersection(p1.Coords, p2.Coords, q1.Coords, q2.Coords)
                if check:
                    degree += 1

        return degree
    
    #-----------------------------------
    #        S E T T E R S
    #-----------------------------------
    
    @AxialForces.setter
    def AxialForces(self, lst):
        self._weights = lst
    
    #---------------------------------------------------
    #        P R I V A T E   M E T H O D S
    #---------------------------------------------------

    def _set_coords(self, labels):
        if type(self._nodal_coordinates) is list or type(self._nodal_coordinates) is tuple:
            return [Vertex(coords[0], coords[1], coords[2], i, label) for i,(coords, label) in enumerate(zip(self._nodal_coordinates, labels))]
        else:
            return "Set coords: I don't know what you expect me to do..."
    
    def _set_edges(self):
        edges = []
        if type(self._connectivity) is list or type(self._connectivity) is tuple:
            for i,pair in enumerate(self._connectivity):
                # Start node
                s =  next((obj for obj in self.VertexObjs if obj.Id == pair[0]), None)
                
                # End node
                e =  next((obj for obj in self.VertexObjs if obj.Id == pair[1]), None)
                
                edges.append(Edge(s, e, i, self._weights[i]))
                
        else:
            return "Set edges: I don't know what you expect me to do..."  
        return edges
    
    def _set_forces(self, anchored_nodes, labels):
        if type(self._load_components) is list or type(self._load_components) is tuple:
            return [Load(load_comp[0], load_comp[1], load_comp[2],anchor_node, i, label) for i,(load_comp,anchor_node, label) in enumerate(zip(self._load_components, anchored_nodes, labels))]
        else:
            return "Set forces: I don't know what you expect me to do..."  

    def _set_connectivity(self, adj_list):
        if type(adj_list) is list or type(adj_list) is tuple:
            return [(pair[0], pair[1]) for pair in adj_list]
        else:
            return "Set connectivity: I don't know what you expect me to do..."

    def _set_weights(self, weights):
        if type(weights) is list or type(weights) is tuple:
            #return [weights[i] for i in range(len(weights))]
            return list(weights)
        else:
            return "Set weights: I don't know what you expect me to do..."       
    
    def _connectivity_to_adj_matrix(self):
        n_nodes = self.Order
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

        for (node, neighbor) in self._connectivity:
            adj_matrix[node][neighbor] = 1
            adj_matrix[neighbor][node] = 1

        return adj_matrix

    def _connectivity_to_inc_matrix(self):
        nodes = sorted(set(node for edge in self._connectivity for node in edge))
        node_to_index = {node: i for i, node in enumerate(nodes)}
        n_edges = self.Size
        n_nodes = self.Order
        
        incidence_matrix = np.zeros((n_edges, n_nodes), dtype=int)

        for idx, (tail, head) in enumerate(self._connectivity):
            i = node_to_index[tail]
            j = node_to_index[head]
            incidence_matrix[idx, i] = 1   # tail
            incidence_matrix[idx, j] = -1  # head

        return incidence_matrix

    def _update_vertices(self):
        for vertex in self.VertexObjs:
            vertex.AdjacentNodes = list(map(lambda x: int(x), np.where(self.Adjacency[vertex.Id] != 0)[0]))
            vertex.IncidentEdges = list(map(lambda x: int(x), np.where(self.Incidence[:, vertex.Id] != 0)[0]))

    def _cycle_orientation(self, cycle):
        """
        Determine if a polygon (defined by node IDs) is clockwise or counterclockwise.

        Parameters
        ----------
        cycle : list[int]
            Ordered node IDs forming a closed polygon.

        Returns
        -------
        0 : "CW"
        1 : "CCW"
        """
        if cycle[0] != cycle[-1]:
            cycle = cycle + [cycle[0]]

        area = 0.0
        for i in range(len(cycle) - 1):
            x1, y1 = self.Coords2D[cycle[i]]
            x2, y2 = self.Coords2D[cycle[i + 1]]
            area += (x2 - x1) * (y2 + y1)

        # area > 0 → CW, area < 0 → CCW (for standard Cartesian coordinates)
        return 0 if area > 0 else 1

    def _utilizations(self, A, yield_strength, percent=False):
        U = [abs(f) / (A * yield_strength) for f in self.AxialForces]
        return [u * 100 for u in U] if percent else U
    
    #-------------------------------------------------
    #        P U B L I C   M E T H O D S
    #-------------------------------------------------
    
    def Cycles(self, CCW=True, ID=True, vertices=True):
        """
        Identifies the faces (cycles) of the graph

        Args:
            CCW (bool, optional): The vertices/edges sequence orientation. Defaults to True - orientates cycles CCW.
            vertices (bool, optional): The type of sequence - edges or vertices. Defaults to True - returns vertices.
            ID (bool, optional): The type of objects - ID or class instances. Defaults to True - returns ID.
        """
        cycles = []
        for cycle in nx.cycle_basis(self.NetworkXGraph):
            if (self._cycle_orientation(cycle) == 0) == CCW:
                cycle.reverse()

            if vertices == True and ID == True:
                # returns vertex ids
                cycles.append(cycle)
                
            elif vertices == False and ID == True:
                # returns edge ids
                edge_pairs = [[cycle[i], cycle[(i+1) % len(cycle)]] for i in range(len(cycle))]
                edges = [self.EdgeByEndIds(pair).Id for pair in edge_pairs]
                
                cycles.append(edges)
                
            elif vertices == True and ID == False:
                # returns vertex objects
                vs = [self.NodeById(v) for v in cycle]
                
                cycles.append(vs)
            
            elif vertices == False and ID == False:
                # retursn edge objs
                edge_pairs = [[cycle[i], cycle[(i+1) % len(cycle)]] for i in range(len(cycle))]
                edges = [self.EdgeByEndIds(pair) for pair in edge_pairs]
                
                cycles.append(edges)
            
        return cycles 

    def EdgeByEndIds(self, pair):
        """
        Args:
            pair (list): it contains the ids of the end (start and emd) nodes

        Returns:
            Edge: the edge defined by the node ids
        """
        for edge in self._edges:
            if (edge.StartNode.Id in pair) and (edge.EndNode.Id in pair):
                return edge

    def MaxGlobalUtilization(self, A, yield_strength, percent=False):
        utilizations = self._utilizations(A, yield_strength, percent)
        return max(utilizations)
    
    def MaxGlobalDisplacement(self, cm=False):
        disp = -1e-16
        for vertex in self.VertexObjs:
            if abs(vertex.Ux) > disp:
                disp = abs(vertex.Ux)
            if abs(vertex.Uy) > disp:
                disp = abs(vertex.Uy)
            if abs(vertex.Uz) > disp:
                disp = abs(vertex.Uz)
        
        return disp*100 if cm else disp

    def FaceAngles(self):
        """
        Compute node angles for all internal faces of a planar graph (CCW sorted).

        Parameters
        ----------
        self : object
            Graph-like object with:
            - self.Order      → number of nodes
            - self.Cycles()   → list of internal faces (each face = list of nodes)
            - self.Coords2D   → {node: (x, y)} positions

        Returns
        -------
        node_angles : dict
            {node: [angles]} — angles sorted CCW around each node
        node_angles_face_ids : dict
            {node: [face_ids]} — corresponding face IDs in the same CCW order
        angle_map : dict
            {(node, face_id): angle} — flattened lookup
        faces : list[list]
            Internal faces (as ordered node lists)
        """
        pos = self.Coords2D
        faces = self.Cycles()
        node_dirs = {n: [] for n in range(self.Order)}

        # Step 1: compute raw angles and direction angles for sorting
        for face_id, face in enumerate(faces):
            n = len(face)
            for i in range(n):
                v_prev = face[i - 1]
                v = face[i]
                v_next = face[(i + 1) % n]

                p_prev = np.array(pos[v_prev])
                p = np.array(pos[v])
                p_next = np.array(pos[v_next])

                u = p_prev - p
                w = p_next - p

                # Compute the interior angle
                angle = Vector.VecAngleBetween(u, w)
                angle = float(round(angle, 2))

                # Direction of next edge (for CCW sorting)
                dir_angle = np.arctan2(p_next[1] - p[1], p_next[0] - p[0])

                node_dirs[v].append((dir_angle, face_id, angle))

        # Step 2: CCW sort and build final structures
        node_angles = {}
        node_angles_face_ids = {}
        angle_map = {}

        for v in range(self.Order):
            sorted_angles = sorted(node_dirs[v], key=lambda x: x[0])
            node_angles[v] = [a for _, _, a in sorted_angles]
            node_angles_face_ids[v] = [fid for _, fid, _ in sorted_angles]
            for _, fid, a in sorted_angles:
                angle_map[(v, fid)] = a

        return node_angles, node_angles_face_ids, angle_map, faces

    def UpdateAnalyzedGraph(self, axial_forces, displacements, strain, stress):
        
        for id, edge in enumerate(self.EdgeObjs):
            edge.ForceMagnitude = axial_forces[id]
            edge.Strain = strain[id]
            edge.Stress= stress[id]
                    
        for id, vertex in enumerate(self.VertexObjs):
            vertex.Ux = displacements[id][0]
            vertex.Uy = displacements[id][1]
            vertex.Uz = displacements[id][2]
        
        self._weights = axial_forces
    
    def EulerCriticalBucklingLimits(self, k, E, I):
        return [Edge.EulerCriticalBucklingLimit(edge, k, E, I) for edge in self.EdgeObjs]

    #-------------------------------------------------
    #        S T A T I C   M E T H O D S
    #-------------------------------------------------

    @staticmethod
    def ByJSONstring(data):
        # --- Node processing ---
        nodal_coordinates = []
        uxs = []
        uys = []
        uzs = []
        for node in data["nodes"]:
            nodal_coordinates.append([node["X"], node["Y"], node["Z"]])
            try:
                uxs.append(node["Ux"])
                uys.append(node["Uy"])
                uzs.append(node["Uz"])
            except:
                pass
        
        # --- Edge processing ---
        adj_list = []
        weights = []
        buckling_limits = []
        for edge in data["edges"]:
            adj_list.append((edge["start_id"], edge["end_id"]))
            weights.append(edge["axial_f"])
            try:
                buckling_limits.append(edge["euler_buckling_limit"])
            except:
                pass

        graph = None
    
        try:
            # --- Load processing
            load_vecs = []
            load_anc_ids = []
            load_labels = []
            for load in data["loads"]:
                load_vecs.append([load["X"], load["Y"], load["Z"]])
                load_anc_ids.append(load["anchor_id"])
                load_labels.append(load["label"])

            # Create the Graph instance
            graph = Graph(nodal_coordinates, adj_list, weights, load_components=load_vecs, anchored_nodes=load_anc_ids, load_labels=load_labels)
        except:
            # Create the Graph instance
            graph = Graph(nodal_coordinates, adj_list, weights)
        
        try:
            for id,node in enumerate(graph.VertexObjs):
                node.Ux = uxs[id]
                node.Uy = uys[id]
                node.Uz = uzs[id]

            for id,edge in enumerate(graph.EdgeObjs):
                edge.BucklingLimit = buckling_limits[id]
        except:
            pass
        
        return graph

class Edge():

    #-----------------------------------
    #      C O N S T R U C T O R
    #-----------------------------------
    
    def __init__(self, start_node, end_node, id, weight, directed=False):
        self._start_node = start_node
        self._end_node = end_node
        self._edge_id = id
        self._is_directed = directed
        self._weight = weight
        self._buckling_limit = 0 # euler buckling limit

    def __str__(self):
        return f"Edge: {self.Id}, Start node: {self.StartNode.Id}, End node: {self.EndNode.Id}, Weight: {self.ForceMagnitude}, Length: {self.Length}, Buckling Limit: {self.BucklingLimit}"
    
    #-----------------------------------
    #        G E T T E R S
    #-----------------------------------
    
    @property
    def Id(self):
        return self._edge_id
    
    @property
    def StartNode(self):
        return self._start_node
    
    @property
    def EndNode(self):
        return self._end_node
    
    @property
    def Length(self):
        start = self.StartNode.Coords
        end = self.EndNode.Coords
        return Geometry.EuclideanDistance(start, end)

    @property
    def ForceMagnitude(self):
        return self._weight
    
    @property
    def BucklingLimit(self):
        return self._buckling_limit
    
    #-----------------------------------
    #        S E T T E R S
    #-----------------------------------
    
    @ForceMagnitude.setter
    def ForceMagnitude(self, value):
        self._weight = value

    @BucklingLimit.setter
    def BucklingLimit(self, value):
        self._buckling_limit = value
    
    #-------------------------------------------------
    #        P U B L I C   M E T H O D S 
    #-------------------------------------------------

    def StartNode_AdjEdges(self):
        return [e_id for e_id in self.StartNode.IncidentEdges if e_id != self.Id]

    def EndNode_AdjEdges(self):
        return [e_id for e_id in self.EndNode.IncidentEdges if e_id != self.Id]

    def EulerCriticalBucklingLimit(self, k, E, I):
        if self.ForceMagnitude > 0:
            self._buckling_limit = 0
        else:
            P_cr = np.pi**2 * E * I / (k*self.Length)**2
            self._buckling_limit = P_cr
        return self.BucklingLimit

class Face():
    
    #-----------------------------------
    #      C O N S T R U C T O R
    #-----------------------------------
    
    def __init__(self, v_ids_seq, id, graph):
        self._vertex_ids_sequence = v_ids_seq
        self._face_id = id
        self._graph = graph

    def __str__(self):
        return f"Face: {self.Id}, Vertices sequence: {self.Vertices}, Edges sequence: {self.Edges}"
    
    #-----------------------------------
    #        G E T T E R S
    #-----------------------------------
    
    @property
    def Id(self):
        return self._face_id
    
    @property
    def Vertices(self):
        return self._vertex_ids_sequence    
    
    @property
    def Edges(self):
        edge_pairs = [[self._vertex_ids_sequence[i], self._vertex_ids_sequence[(i+1) % len(self._vertex_ids_sequence)]] for i in range(len(self._vertex_ids_sequence))]
        return [self._graph.EdgeByEndIds(pair).Id for pair in edge_pairs]
    
class Load():
    
    #-----------------------------------
    #      C O N S T R U C T O R
    #-----------------------------------
    
    def __init__(self, X, Y, Z, node_id, id, label):
        
        self._X = X
        self._Y = Y
        self._Z = Z
        self._load_id = id
        self._node_id = node_id
        self._label = label
    
    def __str__(self):
        return f"Force: {self.Id}, Anchor node: {self.NodeId}, Load vector: {self.Vec}, Label: {self.Label}"
    
    #-----------------------------------
    #        G E T T E R S
    #-----------------------------------
    
    @property
    def X(self):
        return self._X
    
    @property
    def Y(self):
        return self._Y
    
    @property
    def Z(self):
        return self._Z
    
    @property
    def Id(self):
        return self._load_id

    @property
    def NodeId(self):
        return self._node_id
    
    @property
    def Vec(self):
        return [self._X, self._Y, self._Z]
    
    @property
    def Label(self):
        return self._label

class Vertex():
    
    #-----------------------------------
    #      C O N S T R U C T O R
    #-----------------------------------
    
    def __init__(self, X, Y, Z, id, label):
        self._X = X
        self._Y = Y
        self._Z = Z
        self._coords = (X, Y, Z)
        self._node_id = id
        
        self._adjNodes = []
        self._incEdges = []
        self._label = label

        self._ux = 0
        self._uy = 0
        self._uz = 0
    
    def __str__(self):
        return f"Vertex: {self.Id}, Coordinates: {self.Coords}, Label: {self.Label}, Ux: {self.Ux}, Uy: {self.Uy}, Uz: {self.Uz}"
    
    #-----------------------------------
    #        G E T T E R S
    #-----------------------------------
    
    @property
    def AdjacentNodes(self):
        return self._adjNodes

    @property
    def Coords(self):
        return self._coords
    
    @property
    def Id(self):
        return self._node_id

    @property
    def IncidentEdges(self):
        return self._incEdges

    @property
    def Label(self):
        return self._label
        
    @property
    def X(self):
        return self._X
    
    @property
    def Y(self):
        return self._Y
    
    @property
    def Z(self):
        return self._Z
    
    @property
    def Ux(self):
        return self._ux
    
    @property
    def Uy(self):
        return self._uy
    
    @property
    def Uz(self):
        return self._uz
    
    #-----------------------------------
    #        S E T T E R S
    #-----------------------------------
    
    @AdjacentNodes.setter
    def AdjacentNodes(self, lst):
        self._adjNodes = lst

    @IncidentEdges.setter
    def IncidentEdges(self, lst):
        self._incEdges = lst

    @Ux.setter
    def Ux(self, value):
        self._ux = value
    
    @Uy.setter
    def Uy(self, value):
        self._uy = value

    @Uz.setter
    def Uz(self, value):
        self._uz = value
    