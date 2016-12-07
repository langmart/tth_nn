#include <iostream>
using namespace std;

void load(string my_model)
{
    auto load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), my_model, &graph_def);
    auto session_status = session->Create(graph_def);

    vector<tensorflow::Tensor> out;
    vector<string> vNames;

    int node_count = graph_def.node_size();
    
    for (int i = 0; i < node_count; i++)
    {
        auto n = graph_def.node(i);

        if (n.name().find("nWeights") != string::npos)
        {
            vNames.push_back(n.name());
        }
    }
    session->Run({}, vNames, {}, &out);
}

int main()
{
    string fname = "graph/my_graph.pb";
    load(fname);
    cout << "Successfully restored " << fname << "." << endl;
    return 0;
}
