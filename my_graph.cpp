#include <python3.5/Python.h>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <vector>
#include <queue>
#include <set>
#include <map>

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
		size = init_matrix.size();
		rebuild_cur_edges(inf);
		number_of_paths = vector< vector<int> > (size, vector<int>(size, 0));
		mindist_matrix = vector< vector<int> > (size, vector<int>(size, -1));
		prev_vert_matrix = vector< vector<int> > (size, vector<int>(size, -1));
	}

	void rebuild_cur_edges(int max_dist, bool to_sort = false) {
		paths_are_found = false;
		cur_edges = vector< vector< pair<int, int> > >(size, vector< pair<int, int> >());
		for(int i = 0; i < size; i++) {
			for(int j = i + 1; j < size; j++) {
				if(init_matrix[i][j] != -1 && init_matrix[i][j] < max_dist) {
					cur_edges[i].push_back(make_pair(j, init_matrix[i][j]));
					cur_edges[j].push_back(make_pair(i, init_matrix[i][j]));
				}
			}
			
			if(to_sort) {
				sort(cur_edges[i].begin(), cur_edges[i].end(), compare_for_sort);
			}
		}
	}

	void find_shortest_paths() {
		for(int i = 0; i < size; i++) {
			dijkstra(cur_edges, i);
		}
		paths_are_found = true;
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
		if(!paths_are_found) {
			find_shortest_paths();
		}
		return vector2d_to_pylist2d(mindist_matrix);
	}

	PyObject* get_node_betweenness_centrality() {
		calc_node_betweenness_centrality();
		return vector_to_pylist(node_betweenness_centrality);
	}

	PyObject* get_edge_betweenness_centrality() {
		calc_edge_betweenness_centrality();
		return vector2d_to_pylist2d(edge_betweenness_centrality);
	}

	void print_communities() {
		for(int i = 0; i < communities.size(); i++) {
			cout << "communitiy number " << i << endl;
			for(int j = 0; j < communities[i].size(); j++) {
				cout << communities[i][j] << " ";
			}
			cout << endl;
		}
	}

	PyObject* get_communities() {
		return vector2d_to_pylist2d(communities);
	}

	// void find_communities(int max_dist) {
	// 	// M. E. J. Newman and M. Girvan 2004 algorithm
	// 	set< pair<int, int> > all_edges;
	// 	for(int i = 0; i < size; i++) {
	// 		for(int j = 0; j < size; j++) {
	// 			if(init_matrix[i][j] != -1 && init_matrix[i][j] < max_dist) {
	// 				all_edges.insert(make_pair(i, j));
	// 			}
	// 		}
	// 	}

	// 	steps.push_back(all_edges);

	// 	while(!all_edges.empty()) {
	// 		build_communities_edges(all_edges);
	// 		find_shortest_paths(comm_edges);
	// 		calc_edge_betweenness_centrality();
	// 		auto max_edge = find_edge_with_max_centrality();
	// 		all_edges.erase(max_edge);
	// 		all_edges.erase(make_pair(max_edge.second, max_edge.first));
	// 		steps.push_back(all_edges);
	// 	}
	// }

	void dbscan(int max_dist, int eps = 50, int m = 4) {
		rebuild_cur_edges(max_dist, true);
		vector<int> numbers(size, -1);
		int cur_group_nmb = 0;
		for(int i = 0; i < size; i++) {
			if(numbers[i] == -1) {
				int nmb_of_nbrs = lower_bound(cur_edges[i].begin(), cur_edges[i].end(), max_dist, compare_for_lb) - cur_edges[i].begin();
				if(nmb_of_nbrs >= m) {
					numbers[i] = cur_group_nmb;
					extend(i, max_dist, eps, numbers, cur_group_nmb);
					cur_group_nmb++;
				}
			}
		}
		build_communities(numbers, cur_group_nmb);
	}

	void calc_edge_betweenness_centrality() {
		if(!paths_are_found) {
			find_shortest_paths();
		}
		edge_betweenness_centrality = vector< vector<int> > (size, vector<int>(size, 0));
		vector<int> cnt(size, 0);
		for(int i = 0; i < size; i++) {
			vector< pair< int, pair<int, int> > > tmp;
			for(int j = 0; j < size; j++) {
				if(prev_vert_matrix[i][j] != -1) {
					tmp.push_back(make_pair(mindist_matrix[i][j], make_pair(j, i)));
				}
			}

			sort(tmp.begin(), tmp.end(), greater< pair< int, pair<int, int> > >());
			for(auto cur: tmp) {
				int x = cur.second.first;
				int y = cur.second.second;
				edge_betweenness_centrality[x][y] += (cnt[x] + 1);
				edge_betweenness_centrality[y][x] += (cnt[x] + 1);
				cnt[y] += (cnt[x] + 1);
			}
		}
	}

private:
	const int inf = 1e8;
	int size;
	
	void extend(int vert, int max_dist, int eps, vector<int> &numbers, int cur_group_nmb) {
		queue<int> q;
		q.push(vert);
		while(!q.empty()) {
			int cur = q.front();
			q.pop();
			for(int i = 0; i < cur_edges[cur].size(); i++) {
				int to = cur_edges[cur][i].first;
				if(cur_edges[cur][i].second <= eps && numbers[to] == -1) {
					numbers[to] = cur_group_nmb;
					q.push(to);
				}
			}
		}
	}

	void build_communities(vector<int> &numbers, int cur_group_nmb) {
		communities = vector< vector<int> >(cur_group_nmb, vector<int>());
		for(int i = 0; i < numbers.size(); i++) {
			if(numbers[i] != -1) {
				communities[numbers[i]].push_back(i);
			}
		}
	}
	
	bool check_modularity() {
		//to do
		return true;
	}

	void calc_node_betweenness_centrality() {
		if(!paths_are_found) {
			find_shortest_paths();
		}
		vector<int> bc;
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

	pair<int, int> find_edge_with_max_centrality() {
		pair<int, int> cur(0, 0);
		for(int i = 0; i < size; i++) {
			for(int j = 0; j < size; j++) {
				if(edge_betweenness_centrality[i][j] > edge_betweenness_centrality[cur.first][cur.second]) {
					cur = make_pair(i, j);
				}
			}
		}
		return cur;
	}

	void build_communities_edges(const set< pair<int, int> > &edges) {
		comm_edges = vector< vector< pair<int, int> > >(size, vector< pair<int, int> >());
		for(auto cur: edges) {
			comm_edges[cur.first].push_back(make_pair(cur.second, init_matrix[cur.first][cur.second]));
			comm_edges[cur.second].push_back(make_pair(cur.first, init_matrix[cur.first][cur.second]));
		}
	}

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

	static bool compare_for_sort(const pair<int, int> &a, const pair<int, int> &b) {
		return a.second < b.second;
	}

	static bool compare_for_lb(const pair<int, int> &a, const int &b) {
		return a.second < b;
	}

	void dijkstra(const vector< vector< pair<int, int> > > &data, int source) {
		vector<int> d(data.size(), inf);
		vector<int> prev(data.size(), -1);
		vector<int> k(data.size(), 1);
		set< pair<int, int> > edges;
		d[source] = 0;
		prev[source] = source;
		edges.insert(make_pair(d[source], source));
		while(!edges.empty()) {
			int cur_vert = edges.begin()->second;
			edges.erase(edges.begin());
			for(int i = 0; i < data[cur_vert].size(); i++) {
				int target = data[cur_vert][i].first;
				int weight = data[cur_vert][i].second;

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

	bool paths_are_found;
	vector<int> node_betweenness_centrality;
	vector< vector<int> > edge_betweenness_centrality;
	vector< vector<int> > init_matrix;
	vector< vector<int> > communities;
	vector< vector< pair<int, int> > > cur_edges;
	vector< vector< pair<int, int> > > comm_edges;
	vector< vector<int> > number_of_paths;
	vector< vector<int> > mindist_matrix;
	vector< vector<int> > prev_vert_matrix;
};

extern "C" {
	Graph* new_graph(PyObject* data) {
		return new Graph(data);
	}

	PyObject* get_current_matrix(Graph* g) {
		return g->get_current_matrix();
	}

	PyObject* get_mindist_matrix(Graph* g) {
		return g->get_mindist_matrix();
	}

	PyObject* get_communities(Graph* g) {
		return g->get_communities();
	}

	PyObject* get_node_betweenness_centrality(Graph* g) {
		return g->get_node_betweenness_centrality();
	}

	PyObject* get_edge_betweenness_centrality(Graph* g) {
		return g->get_edge_betweenness_centrality();
	}

	void calc_edge_betweenness_centrality(Graph* g) {
		g->calc_edge_betweenness_centrality();
	}

	void rebuild_cur_edges(Graph* g, int max_dist) {
		g->rebuild_cur_edges(max_dist);
	}

	void find_shortest_paths(Graph* g) {
		g->find_shortest_paths();
	} 

	void dbscan(Graph* g, int max_dist, int eps, int m) {
		g->dbscan(max_dist, eps, m);
	}

	void print_communities(Graph* g) {
		g->print_communities();
	}
}