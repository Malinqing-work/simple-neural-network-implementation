#include "Net.h"
#include "Utils.h"
/*明天的工作计划：修改反向传播和计算误差函数
 *
 *从反向传播开始查改
 *
 */
using namespace std;
// 构造函数
Network::Network(const std::vector<int>& sizes) {
        this->sizes = sizes;
        this->biases.resize(sizes.size() - 1);
        this->weights.resize(sizes.size() - 1);

        std::srand(static_cast<unsigned int>(std::time(nullptr))); // 设置随机种子

        // 初始化权重和偏置
        for (size_t i = 1; i < sizes.size(); ++i) {
            biases[i - 1].resize(sizes[i]);
            weights[i - 1].resize(sizes[i - 1]);

            for (int j = 0; j < sizes[i]; ++j) {
                biases[i - 1][j] = 0.0; // 偏置初始化为0

                for (int k = 0; k < sizes[i - 1]; ++k) {
                    // 使用 He 初始化权重
                    weights[i - 1][j].push_back(Utils::he_init_weight(sizes[i - 1]));
                }
            }
        }
}
// 前向传播
vector<double> Network:: feedforward(const vector<double>& a) {
        vector<double> activation = a;   // 输入的列向量
        for (size_t i = 1; i < sizes.size(); ++i) {
            vector<double> next_activation;
            for (int j = 0; j < sizes[i]; ++j) {
                double sum = 0.0;
                for (int k = 0; k < sizes[i - 1]; ++k) {
                    sum += weights[i - 1][j][k] * activation[k];
                }
                sum += biases[i - 1][j];
                next_activation.push_back(Utils::sigmoid(sum));
            }
            activation = next_activation;
        }
        return activation;
}

void Network::backprop(const std::vector<double>& x, int y,
                       std::vector<std::vector<double>>& gradient_biases,
                       std::vector<std::vector<std::vector<double>>>& gradient_weights) {

    // 存储每一层的激活值和未激活的z值
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> z_values;

    // 初始化第一层的激活值
    std::vector<double> activation = x;
    activations.push_back(activation);

    // 前向传播得到每一层的激活值和z值
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> next_activation;
        std::vector<double> z_i;
        for (int j = 0; j < sizes[i + 1]; ++j) {
            double sum = biases[i][j];
            for (int k = 0; k < sizes[i]; ++k) {
                sum += weights[i][j][k] * activation[k];
            }
            z_i.push_back(sum);
            next_activation.push_back(Utils::sigmoid(sum));
        }
        activation = next_activation;
        activations.push_back(next_activation);
        z_values.push_back(z_i);
    }

    // 初始化梯度
    gradient_biases.clear();
    gradient_weights.clear();
    for (size_t i = 0; i < weights.size(); ++i) {
        gradient_biases.push_back(std::vector<double>(sizes[i + 1], 0.0));
        gradient_weights.push_back(std::vector<std::vector<double>>(
            sizes[i], std::vector<double>(sizes[i + 1], 0.0)));
    }

    // 计算输出层的误差梯度
    std::vector<double> output_delta = cost_derivative(activations.back(), y);
    // 反向传播计算梯度
    for (int l = weights.size() - 1; l > 0; --l) {
        for (int j = 0; j < sizes[l]; ++j) {
            // 计算当前层第j个神经元的delta值
            double delta = (l == weights.size() - 1) ?
                output_delta[j] * Utils::sigmoid_prime(z_values.back()[j]) :
                0.0; // 对于隐藏层，delta初始为0，将在下面的循环中累加

            // 累加来自下一层的delta值
            for (int k = 0; k < sizes[l + 1]; ++k) {
                delta += gradient_weights[l][k][j] * output_delta[k];
            }

            // 更新偏置和权重梯度
            gradient_biases[l - 1][j] = delta;
            for (int k = 0; k < sizes[l - 1]; ++k) {  // 这里的循环变量k 和 j 和 l 分别是什么意思

            gradient_weights[l - 1][k][j] = delta * activations[l - 1][k];
        }
        }
    }

}

std::vector<double> Network::cost_derivative(const std::vector<double>& output_activation, int y) {
    std::vector<double> delta(output_activation.size(), 0.0);  // 初始化为0
    for (size_t i = 0; i < output_activation.size(); ++i) {
        if (i == static_cast<size_t>(y)) {
            delta[i] = -(1.0 - output_activation[i]); // 期望类别的梯度
        } else {
            delta[i] = -output_activation[i]; // 其他类别的梯度
        }
    }
    return delta;
}

void Network::update_mini_batch(const std::vector<std::pair<std::vector<double>, int>>& mini_batch, double eta) {
    // 初始化或重置梯度累加器
    std::vector<std::vector<double>> gradient_biases;
    std::vector<std::vector<std::vector<double>>> gradient_weights;
    initialize_gradient_accumulators(gradient_biases, gradient_weights);
    // 如果mini_batch只有一个样本，直接调用backprop
    if (mini_batch.size() == 1) {
        backprop(mini_batch[0].first, mini_batch[0].second, gradient_biases, gradient_weights);
    }
    else {

        for (const auto& sample : mini_batch) {
        std::vector<std::vector<double>> local_gradient_biases;
        std::vector<std::vector<std::vector<double>>> local_gradient_weights;

        backprop(sample.first, sample.second, local_gradient_biases, local_gradient_weights);
        // 将这些梯度累加到总梯度中
        for (size_t i = 0; i < gradient_biases.size(); ++i) {
            for (size_t j = 0; j < gradient_biases[i].size(); ++j) {
                gradient_biases[i][j] += local_gradient_biases[i][j];
                for (size_t k = 0; k < gradient_weights[i][j].size(); ++k) {
                    gradient_weights[i][j][k] += local_gradient_weights[i][j][k];

                }
            }
        }
      }
    }
     // 更新网络的权重和偏置
    // 这里的更新应该是基于整个mini-batch的平均梯度
    update_parameters(gradient_biases, gradient_weights, eta / mini_batch.size());
}



// 辅助函数，用于更新网络参数
void Network::update_parameters( std::vector<std::vector<double>>& gradient_biases,
                                 std::vector<std::vector<std::vector<double>>>& gradient_weights,double eta) {

    // 根据梯度更新权重和偏置
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        for (size_t j = 0; j < sizes[i + 1]; ++j) {
            //cout<<"flag4"<<endl;
            biases[i][j] -= eta * gradient_biases[i][j];
            for (size_t k = 0; k < sizes[i]; ++k) {
                weights[i][j][k] -= eta * gradient_weights[i][j][k];
            }
        }
    }
}

// 辅助函数，用于初始化梯度累加器
void Network::initialize_gradient_accumulators(std::vector<std::vector<double>>& gradient_biases,
                                              std::vector<std::vector<std::vector<double>>>& gradient_weights) {
    // 根据网络结构初始化梯度累加器
    gradient_biases.clear();
    gradient_weights.clear();
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        gradient_biases.push_back(std::vector<double>(sizes[i + 1], 0.0));
        gradient_weights.push_back(std::vector<std::vector<double>>(sizes[i], std::vector<double>(sizes[i + 1], 0.0)));
    }
}

void Network::train(const std::vector<std::pair<std::vector<double>, int>>& training_data,
                    int epochs, double eta,
                    const std::vector<std::pair<std::vector<double>, int>>& test_data) {
     vector<int> predicts;
     int maxNumber = 0;
     int num  =0;
    for (int j = 0; j < epochs; ++j) {
        // 遍历整个训练数据集
        for (const auto& sample : training_data)
        {
            update_mini_batch({sample}, eta) ;
        }
        // 每训练一轮就评估一次
        if (!test_data.empty()) {
            num = evaluate(test_data);
            predicts.push_back(num);

            if (num > maxNumber) {
                maxNumber = num;   // 更新最大准确率
                save("result");    // 保存当前网络参数

            }
             std::cout << "Epoch: " << j + 1 << "  "
                      << num << "/" << test_data.size()
                      << " " << (num / static_cast<double>(test_data.size()) * 100) << "%" << std::endl;

            }
        }
        cout << "Test accuracy: " << maxNumber << "/" << test_data.size() << endl;
}


/*假设 output 是 [0.1, 0.7, 0.2]，那么：
 *max_element(output.begin(), output.end()) 将返回指向 0.7 的迭代器。
 *distance(output.begin(), max_element(output.begin(), output.end()))
 *将计算从 output.begin() 到指向 0.7 的迭代器的距离，结果是 1（因为 0.7 是第二个元素）
*/
int Network::evaluate(const vector<pair<vector<double>, int>>& test_data) {
    int correct = 0;
    for (const auto& sample : test_data) {
        vector<double> output = feedforward(sample.first);
        int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
        //cout<<"标签数据:"<<sample.second<< "  预测数据:"<<predicted<<endl;
        if (predicted == sample.second) {
            correct++;
        }
    }
    return correct;
}
vector<int> Network::predict(vector<vector<double>> test_data)
{
    vector<int> result;
    for(const auto &sample:test_data)
    {
        vector<double> output = feedforward(sample);
        int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
        result.push_back(predicted);
    }
    return result;
}
void Network::save(const std::string& file_name) {
    std::string filename = file_name + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to create file: " << filename << std::endl;
        return;
    }

    // 写入层数
    int num_layers = sizes.size();
    file << num_layers << std::endl;

    for (size_t i = 1; i < sizes.size(); ++i) {
        int num_neurons = sizes[i];
        file << num_neurons << std::endl;

        for (int j = 0; j < num_neurons; ++j) {
            // 写入偏置
            file << biases[i - 1][j] << std::endl;
            // 写入权重
            size_t num_inputs = sizes[i - 1];
            for (size_t k = 0; k < num_inputs; ++k) {
                file << weights[i - 1][j][k] << " ";
            }
            file << std::endl;
        }
    }

    file.close();
}

void Network::load(const std::string& file_name) {
    std::string filename = file_name + ".txt";
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    // 读取层数
    int num_layers;
    file >> num_layers;
    sizes.resize(num_layers);
    biases.resize(num_layers - 1);
    weights.resize(num_layers - 1);

    for (size_t i = 1; i < sizes.size(); ++i) {
        int num_neurons;
        file >> num_neurons;
        sizes[i] = num_neurons;
        biases[i - 1].resize(num_neurons);
        weights[i - 1].resize(num_neurons);

        for (int j = 0; j < num_neurons; ++j) {
            // 读取偏置
            file >> biases[i - 1][j];
            // 读取权重
            size_t num_inputs = sizes[i - 1];
            for (size_t k = 0; k < num_inputs; ++k) {
                file >> weights[i - 1][j][k];
            }
        }
    }

    file.close();
}
void Network::print_network() const {
        std::cout << "Network structure:" << std::endl;
        for (size_t i = 0; i < sizes.size(); ++i) {
            std::cout << "Layer " << i << " size: " << sizes[i] << std::endl;
        }
}
