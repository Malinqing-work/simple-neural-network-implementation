#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <random>
#include <ctime>
#include <cstdlib>
#include <functional>
using namespace std;

namespace Utils{

template<typename T>
constexpr const T& min(const T& a, const T& b) {
    return b < a ? b : a;
}

template<typename T, typename U>
void my_swap(std::pair<T, U>& lhs, std::pair<T, U>& rhs) {
    using std::swap;
    swap(lhs.first, rhs.first);
    swap(lhs.second, rhs.second);
    }

// 读取测试集
void getTestData(const string &filename,vector<vector<double>>&data);

//读取训练集
void getTrainData(const string &filename, vector<int> &labels, vector<vector<double>> &data);

// 将输出压缩成激活值的函数
double sigmoid(double z);

//激活函数的导数
double sigmoid_prime(double z);


// 用于初始化权重的辅助函数
 double he_init_weight(int fan_in);

 double random_weight();

}
