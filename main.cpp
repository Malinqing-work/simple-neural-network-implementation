#include <iostream>
#include "lib/Net.h"
#include "lib/Utils.h"

using namespace std;

int main() {
    vector<int> train_labels,test_labels;
    vector<vector<double>>train_data,test_data;
    vector<pair<vector<double>, int>> training_data,testing_data;

    Utils::getTrainData("./data/small_train.csv",train_labels,train_data);

    Utils::getTrainData("./data/small_test.csv",test_labels,test_data);
    // 构造像素数据和标签数据的映射
    for (size_t i = 0; i < train_data.size(); ++i) {
        training_data.push_back(make_pair(train_data[i], train_labels[i]));
    }
    for(size_t i =0;i<test_data.size();++i){
        testing_data.push_back(make_pair(test_data[i],test_labels[i]));
    }

    Network net({784, 16,16, 10});



    // 训练网络
    net.train(training_data,30,0.2,testing_data);  // 30 为训练次数;batch_size为50;0.3为学习率；



    //打印网络结构
    net.print_network();

/*上述是训练过程并保存网络参数  下述模块是利用得到的网络参数进行预测*/
//    vector<int> train_labels;
//    vector <vector<double>> train_data;
//    vector <int> result;
//    Utils::getTrainData("./data/small_test.csv",train_labels,train_data);
//
//    Network net({784, 16,16, 10});
//
//    net.load("./result");
//
//    result=net.predict(train_data);
//
//    for(int i=0;i<train_labels.size();i++)
//    {
//       cout<<"标签:"<<train_labels[i]<< "  预测:"<<result[i]<<endl;
//    }
//
//    return 0;
}
