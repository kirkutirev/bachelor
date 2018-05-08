#include <python3.5/Python.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <queue>
#include <set>

using namespace std;

class Graph {
public:
	Graph(PyObject* data) {
		int cnt = 0;
		for(Py_ssize_t i = 0; i < PyList_Size(data); i++) {
			PyObject* cur_vert = PyList_GetItem(data, i);
			init_matrix.push_back(vector<int>());
			for(Py_ssize_t j = 0; j < PyList_Size(cur_vert); j++) {
				PyObject* cur_edge = PyList_GetItem(cur_vert, j);
				init_matrix[cnt].push_back(PyLong_AsLongLong(cur_edge));
			}
			cnt++;
		}
		init_other_matrices();
	}

	void find_shortest_paths() {
		for(int i = 0; i < cur_edges.size(); i++) {
			dijkstra(i);
		}
	}

	void calc_node_betweenness_centrality(int max_dist) {
		vector<int> bc;
		rebuild_cur_edges(max_dist);
		find_shortest_paths();

		for(int v = 0; v < cur_edges.size(); v++) {
			double cur = 0;
			for(int i = 0; i < cur_edges[v].size(); i++) {
				for(int j = i + 1; j < cur_edges[v].size(); j++) {
					int s = cur_edges[v][i].first;
					int t = cur_edges[v][j].first;
					if(mindist_matrix[s][t] == mindist_matrix[s][v] + mindist_matrix[v][t]) {
						cur += (double) number_of_paths[s][v] * number_of_paths[v][t] / number_of_paths[s][t];
					}
				}
			}
			bc.push_back(round(cur));
		}
		node_betweenness_centrality = bc;
	}

	PyObject* get_current_matrix() {
		vector< vector<int> > cur(cur_edges.size(), vector<int>(cur_edges.size(), -1));
		for(int i = 0; i < cur_edges.size(); i++) {
			for(int j = 0; j < cur_edges[i].size(); j++) {
				cur[i][cur_edges[i][j].first] = cur_edges[i][j].second;
			}
		}
		return vector2d_to_pylist2d(cur);
	}

	PyObject* get_mindist_matrix() {
		return vector2d_to_pylist2d(mindist_matrix);
	}

	PyObject* get_node_betweenness_centrality(int max_dist) {
		calc_node_betweenness_centrality(max_dist);
		return vector_to_pylist(node_betweenness_centrality);
	}

private:
	const int inf = 1e8;

	PyObject* vector_to_pylist(vector<int> &v) {
		PyObject* pylist = PyList_New(v.size());
		for(int i = 0; i < v.size(); i++) {
			PyObject* cur = PyLong_FromLongLong((long long)v[i]);
			PyList_SetItem(pylist, i, cur);
		}
		return pylist;
	}

	PyObject* vector2d_to_pylist2d(vector< vector<int> > &v) {
		PyObject* pylist = PyList_New(v.size());
		for(int i = 0; i < v.size(); i++) {
			PyObject* cur = vector_to_pylist(v[i]);
			PyList_SetItem(pylist, i, cur);
		}
		return pylist;
	}

	void init_other_matrices() {
		int n = init_matrix.size();
		rebuild_cur_edges(inf);
		number_of_paths = vector< vector<int> > (n, vector<int>(n, 0));
		mindist_matrix = vector< vector<int> > (n, vector<int>(n, 0));
		prev_vert_matrix = vector< vector<int> > (n, vector<int>(n, 0));
	}

	void rebuild_cur_edges(int max_dist) {
		int n = init_matrix.size();
		cur_edges = vector< vector< pair<int, int> > >(n, vector< pair<int, int> >());
		for(int i = 0; i < n; i++) {
			for(int j = i + 1; j < n; j++) {
				if(init_matrix[i][j] != -1 && init_matrix[i][j] < max_dist) {
					cur_edges[i].push_back(make_pair(j, init_matrix[i][j]));
					cur_edges[j].push_back(make_pair(i, init_matrix[i][j]));
				}
			}
		}
	}

	void dijkstra(int source) {
		vector<int> d(cur_edges.size(), inf);
		vector<int> prev(cur_edges.size(), -1);
		vector<int> k(cur_edges.size(), 1);
		set< pair<int, int> > edges;
		d[source] = 0;
		edges.insert(make_pair(d[source], source));
		while(!edges.empty()) {
			int cur_vert = edges.begin()->second;
			edges.erase(edges.begin());
			for(int i = 0; i < cur_edges[cur_vert].size(); i++) {
				int target = cur_edges[cur_vert][i].first;
				int weight = cur_edges[cur_vert][i].second;

				if(d[target] > d[cur_vert] + weight) {
					edges.erase(make_pair(d[target], target));
					d[target] = d[cur_vert] + weight;
					prev[target] = cur_vert;
					k[target] = k[cur_vert];
					edges.insert(make_pair(d[target], target));
				} else if(d[target] == d[cur_vert] + weight) {
					k[target] += k[cur_vert];
				}
			}
		}

		number_of_paths[source] = k;
		mindist_matrix[source] = d;
		prev_vert_matrix[source] = prev;
	}

	vector<int> node_betweenness_centrality;
	vector< vector<int> >init_matrix;
	vector< vector< pair<int, int> > > cur_edges;
	vector< vector<int> > number_of_paths;
	vector< vector<int> > mindist_matrix;
	vector< vector<int> > prev_vert_matrix;
};

extern "C" {
	Graph* new_graph(PyObject* data) {
		return new Graph(data);
	}

	void find_graph_info(Graph* g) {
		g->find_shortest_paths();
	}

	PyObject* get_current_matrix(Graph* g) {
		return g->get_current_matrix();
	}

	PyObject* get_node_betweenness_centrality(Graph* g, int max_dist) {
		return g->get_node_betweenness_centrality(max_dist);
	}

	PyObject* get_mindist_matrix(Graph* g) {
		return g->get_mindist_matrix();
	}
}