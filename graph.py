import numpy as np
import json
from collections import defaultdict
from scipy.special import comb

# load an index in json format
# output: name_to_pages: dict of name to list of pages
#         page_to_names: dict of page to list of names
def read_index_json(fn):
	with open(fn, 'r') as f:
		name_to_pages = json.load(f)
	page_to_names = defaultdict(list)
	for name, pages in name_to_pages.items():
		for page in pages:
			page_to_names[page].append(name)
	return name_to_pages, page_to_names

# construct the adjacency matrix representation of a text's graph
# input: pagesPerName: dict of name to list of pages
#        namesPerPage: dict of page to list of names
#        chapterBounds: if provided, only create edges for pages between these numbers
# Output: A: adjacency matrix (A[i, j] = 1 iff i and j appear on the same page)
#         allNames: list of names
#         idToName: dict of name to row/col index in A
def get_adjacency_matrix(pagesPerName, namesPerPage, chapterBounds=None):
	allNames = list(pagesPerName.keys())
	idToName = {}
	for i, name in enumerate(allNames):
		idToName[i] = name
	n = len(allNames)
	A = np.zeros((n, n))
	# pages are 1-indexed
	for page, names in namesPerPage.items():
		if chapterBounds != None and (page < chapterBounds[0] or page > chapterBounds[1]):
			continue
		for n1 in names:
			for n2 in names:
				if n1 == n2: continue
				i1 = allNames.index(n1)
				i2 = allNames.index(n2) 
				A[i1, i2] = 1
				A[i2, i1] = 1
	return A, allNames, idToName

# same as above but only inludes vertices that are listed in the nodes argument
def get_adjacency_matrix_on_specific_nodes(pagesPerName, namesPerPage, chapterBounds, nodes):
	allNames = list(pagesPerName.keys())
	idToName = {}
	for i, name in enumerate(allNames):
		idToName[i] = name
	nodeToId = {}
	for i, n in enumerate(nodes):
		nodeToId[i] = n
	A = np.zeros((len(nodes), len(nodes)))
	# pages are 1-indexed
	for page, names in namesPerPage.items():
		if page < chapterBounds[0] or page > chapterBounds[1]: continue
		#print(page, names)
		for n1 in names:
			if allNames.index(n1) not in nodes: continue
			for n2 in names:
				if allNames.index(n2) not in nodes: continue
				if n1 == n2: continue
				i1 = nodes.index(allNames.index(n1))
				i2 = nodes.index(allNames.index(n2)) 
				A[i1, i2] = 1
				A[i2, i1] = 1
	return A, allNames, idToName, nodeToId

#input: A is an adjacency matrix
#       1 - alpha is the probability of teleporting
# output: x is a vector of pagerank values
def pagerank(A, alpha):
	n = A.shape[0]
	P = np.transpose(A)
	P = P / P.sum(axis=0)
	v = np.ones((n)) / n
	x = np.copy(v)
	for i in range(0, 100):
		x_next = np.dot(alpha, np.dot(P, x)) + np.dot(1 - alpha, v)
		x_next = x_next / x_next.sum()
		x = x_next
	return x

# count triangles
# each is 
# input: A is an adjacency matrix
# output: count: number of triangles
#         triangles: set of triangles (each identified by the indices of the nodes within it)
def enumerate_triangles(A):
	n = A.shape[0]
	count = 0
	triangles = set()
	for u in range(0, n):
		for v in range(0, n):
			for w in range(0, n):
				if u == v or u == w or v == w: continue
				triangle = (A[u, v] or A[v, u]) and (A[u, w] or A[w, u]) and (A[v, w] or A[w, v])
				if triangle:
					count += 1
					triangles.add(tuple(sorted([u, v, w])))
	return count / 6, triangles

# input: A is a symmetric adjacency matrix (undirected graph)
#        k is number of samples
# output: estimate of the global clustering coefficient
def wedge_sampling(A, k):
	n = A.shape[0]
	degrees = A.sum(axis=0)
	# probability of selecting vertex v is proportion of wedges centered at v
	p_unnorm = [comb(d, 2) for d in degrees]
	total_wedges = sum(p_unnorm)
	p = [p_v / total_wedges for p_v in p_unnorm]
	vs = np.random.choice(n, k, p=p)
	count = 0
	for v in vs:
		# sample two neighbors uniformly without replacement
		neighbors = (np.where(A[v, :] == 1))[0]
		u, w = np.random.choice(neighbors, 2, replace=False)
		triangle = A[u, v] and A[u, w] and A[v, w]
		if triangle: count += 1
	return (count / k)

# enumerates triangle combinations of entities that appear on the same page
def triangles_from_same_page(namesPerPage, idToName, pages):
	page_triangles = set()
	nameToId = {}
	for i, name in idToName.items():
		nameToId[name] = i
	for page, names in namesPerPage.items():
		if page < pages[0] or page > pages[1]: continue
		# some names in namesPerPage may have been removed from idToName because they are disconnected!
		ids = [nameToId[name] for name in names]
		# enumerate all permutations of size 3--counts each 6 times (i.e. 3*2*1 more permutations than combinations)
		for u_i, u in enumerate(ids):
			for v_i, v in enumerate(ids):
				for w_i, w in enumerate(ids):
					if u_i == v_i or u_i == w_i or v_i == w_i: continue
					page_triangles.add(tuple(sorted([u, v, w])))
	return page_triangles

# output: global clustering coefficient for a graph
def global_clustering_coefficient(A, triangles):
	degrees = A.sum(axis=0)
	two_neighbor_pairs = [comb(d, 2) for d in degrees]
	return (3 * triangles / sum(two_neighbor_pairs))

# This function performs all the analysis for a graph.
# input: pagesPerName: dict of name to list of pages
#        namesPerPage: dict of page to list of names
#		 name: name of the textfile that will be saved
#	 	 pages: a tuple of the start and end page for this text
#        A: adjacency matrix of the graph
#        names: list of names
#        idToName: dict of name to row/col index in A
# output: a text file with graph statistics, triangle counts, and pageranks
def graph_all(pagesPerName, namesPerPage, name, pages, A, names, idToName):
	with open('%s.txt' % name, 'w')  as txtfile:

		idToNameWithDisconnectedVertices = idToName.copy()

		# remove disconnected vertices
		fullNames = names.copy()
		for x, dx in enumerate(np.sum(A, axis=0)):
			if dx == 0:	
				names.remove(fullNames[x])
				idToName.pop(x)

		d = A.sum(axis=0)
		txtfile.write('Vertices: %d\n' % len(idToName.keys()))
		txtfile.write('Edges: %d\n' % (A.sum() / 2))
		txtfile.write('Num vertices with degree 1: %d\n' % len((np.where(d == 1))[0]))
		txtfile.write('Max degree: %d\n' % max(d))
		txtfile.write('Vertices with max degree: ')
		txtfile.write(', '.join([idToName[i] for i in (np.where(d == max(d)))[0]]))
		txtfile.write('\n\n\n')

		# triangles!
		exact_triangles, triangles = enumerate_triangles(A)
		artifact_triangles = triangles_from_same_page(namesPerPage, idToNameWithDisconnectedVertices, pages)
		exact_triangles_different_pages = triangles.difference(artifact_triangles)
		gcce = global_clustering_coefficient(A, exact_triangles)
		approx_gcce = wedge_sampling(A, 380)
		txtfile.write('Triangles: %d\n' % exact_triangles)
		txtfile.write('Triangles from different pages: %d\n' %len(exact_triangles_different_pages))
		txtfile.write('Global clustering coefficient:\t\t\t%.16f\n' % gcce)
		txtfile.write('Approximate global clustering coefficient:\t%.16f\n' % approx_gcce)
		txtfile.write('\n\n')

		# pagerank
		# ignore the vertices with 0 edges
		A, names, idToName, nodeToId  = get_adjacency_matrix_on_specific_nodes(pagesPerName, namesPerPage, pages, list(idToName.keys()))
		original_ranks = pagerank(A, alpha=0.9)
		
		# sort by pagerank and print 
		ranks = [(x, idToName[nodeToId[i]]) for i, x in enumerate(original_ranks)]
		ranks.sort(reverse=True)
		txtfile.write('PageRank\n')
		for x, n in ranks:
			txtfile.write('%.16f\t%s\n' % (x, n))

def main():
	
	print('Creating graph: Theory and Design in the First Machine Age')
	pagesPerName, namesPerPage = read_index_json('theory-and-design-in-the-first-machine-age.json')
	A, names, idToName = get_adjacency_matrix(pagesPerName, namesPerPage)
	graph_all(pagesPerName, namesPerPage, 'banham-full', (1, 338), A, names, idToName)
	
	print('Creating graph: Toward an Architecture (intro)')
	pagesPerName, namesPerPage = read_index_json('toward-an-architecture.json')
	A, names, idToName = get_adjacency_matrix(pagesPerName, namesPerPage, (1, 77))
	graph_all(pagesPerName, namesPerPage, 'corbusier-intro', (1, 77), A, names, idToName)
	
	print('Creating graph: Toward an Architecture (full book)')
	A, names, idToName = get_adjacency_matrix(pagesPerName, namesPerPage, (80, 307))
	graph_all(pagesPerName, namesPerPage, 'corbusier-full', (80, 307), A, names, idToName)	

	pagesPerName, namesPerPage = read_index_json('histories-of-the-immediate-present.json')
	print('Creating graph: Histories of the Immediate Present')
	A, names, idToName = get_adjacency_matrix(pagesPerName, namesPerPage)
	graph_all(pagesPerName, namesPerPage, 'vidler-full', (1, 200), A, names, idToName)

if __name__ == '__main__':
	main()