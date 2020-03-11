#include <vector>
#include <random>

class HiddenMarkovModel {
private:
    int L;
    int D;
    std::vector<std::vector<double>> A;
    std::vector<std::vector<double>> O;
    std::vector<double> A_start;
public:
    HiddenMarkovModel(int L, int D) {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        double number = distribution(generator);

    }
};