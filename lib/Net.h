#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>


using namespace std;

class Network{
private:
    std::vector<int> sizes; // 一维数组下标表示层数，值表示层网络节点数
    std::vector<std::vector<double>> biases; // 二维数组下表表示第i层第j个结点，值表示偏置值
    std::vector<std::vector<std::vector<double>>> weights; // 三维数组，下标表示第i层第j个结点到下一层第k个结点的权重(全连接)
public:
// 构造函数
Network(const std::vector<int>& sizes);

//前向传播
std::vector<double> feedforward(const std::vector<double>& a);

//反向传播
void backprop(const std::vector<double>& x, int y, std::vector<std::vector<double>>& gradient_biases,
                  std::vector<std::vector<std::vector<double>>>& gradient_weights);

//计算输出层的误差
std::vector<double> cost_derivative(const std::vector<double>& output_activation, int y);

// 更新权重和偏置
void update_mini_batch(const std::vector<std::pair<std::vector<double>, int>>& mini_batch, double eta);

//评估网络
int evaluate(const std::vector<std::pair<std::vector<double>, int>>& test_data);


//训练参数
//void train(const std::vector<std::pair<std::vector<double>, int>>& training_data, int epochs,
//          double eta, const std::vector<std::pair<std::vector<double>, int>>& test_data);
void train(const std::vector<std::pair<std::vector<double>, int>>& training_data,
                    int epochs, double eta,
                    const std::vector<std::pair<std::vector<double>, int>>& test_data);
//保存网络当前参数
void save(const std::string& file_name);
//导入训练好的网络参数
void load(const std::string& file_name);
//打印网络结构
void print_network () const;

void initialize_gradient_accumulators(std::vector<std::vector<double>>& gradient_biases,
                                 std::vector<std::vector<std::vector<double>>>& gradient_weights);

void update_parameters( std::vector<std::vector<double>>& gradient_biases,
                     std::vector<std::vector<std::vector<double>>>& gradient_weights,double eta);

vector<int> predict(vector<vector<double>> test_data);

vector<double> softmax(const std::vector<double>& input);
};
