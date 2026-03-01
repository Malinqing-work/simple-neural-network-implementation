#include "Utils.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace std;
namespace Utils{
// 将输出压缩成激活值的函数
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}
//激活函数的导数
double sigmoid_prime(double z){
    return sigmoid(z)*(1-sigmoid(z));
}


 // 生成 [-0.05, 0.05] 范围内的随机权重
double random_weight() {
        return (rand() / static_cast<double>(RAND_MAX)) * 0.1 - 0.05;
    }

    // He 初始化权重
double he_init_weight(int fan_in) {
        return std::sqrt(2.0 / fan_in) * (rand() / static_cast<double>(RAND_MAX)) - 0.5;
    }
// 读取CSV文件中的所有行数据，并将其转换为double类型的二维vector
// 二维数组中的每一行保存一个图像的数据
void getTestData(const string &filename,vector<vector<double>>&data) {

    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    while (getline(file, line)) { // 按行读取
        vector<double> row;
        stringstream ss(line);
        string value;
        double num;

        while (getline(ss, value, ',')) {
            num = stod(value);
            row.push_back(num);
        }

        data.push_back(row);
    }

    file.close();
}
void getTrainData(const string &filename, vector<int> &labels, vector<vector<double>> &data) {
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    while (getline(file, line)) { // 按行读取
        vector<double> row;
        stringstream ss(line);
        string value;
        int label;
        double num;

        // 读取标签
        getline(ss, value, ',');
        label = stoi(value);
        labels.push_back(label);
        // 跳过第一个字符（标签）
        getline(ss, value,','); // 读取并丢弃第一个值（标签）
        // 读取图像数据
        while (getline(ss, value, ',')) {
            num = stod(value);
            row.push_back(num/255);
        }

        data.push_back(row);
    }

    file.close();
}

}


//int main() {
//    string filename = "mnist_train.csv";
//    vector<int> labels;
//    vector<vector<double>> data;
//
//    // 获取训练数据
//    getTrainData(filename, labels, data);
//
//    if (data.empty()) {
//        cerr << "No data read from file." << endl;
//        return 1;
//    }
//
//
//    // 输出归一化后的第一行图像数据
//    int index=0;
//    for(int i=0;i<labels.size();i++)
//   {
//       if(labels[i]==6)
//       {
//           index=i;
//           break;
//       }
//   }
//
//    cout << "Normalized first row data:" << endl;
//
//        for(int j=index;j<784;j++)
//        {
//            if(data[index][j]!=0) data[index][j]=1;
//            cout<<data[index][j]<<" ";
//            if(j%28==0) cout<<endl;
//        }
//        cout<<endl;
//
////    for (const auto &value : data[0]) {
////        cout << value << " ";
////    }
////    cout << endl;
//
//    return 0;
//}
