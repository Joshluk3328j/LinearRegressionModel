#include <iostream>
#include <vector>
#include <cmath>
// #include <matplotlibcpp.h>

class LinearRegression {
    protected:
        double x, y, m, c, numerator, denominator, score, x_avg, y_avg;
        std::pair <double, double> avgs = {};
        std::vector <double> preds = {};
        int arr_length = 0;

        std::pair <double, double> getAverage(const std::vector <double> &x_t, const std::vector <double> &y_t) {
            double x_sum = 0;
            double y_sum = 0;
            arr_length = x_t.size();
            // calculate average of x and y train

            for (int val : x_t) {
                x_sum += val;
            }

            for (int val : y_t) {
                y_sum +=val;
            }
            return std::make_pair(x_sum / arr_length , y_sum / arr_length);

        }
    public:
        LinearRegression() {
            std::cout << "The model has been instantiated" << std::endl;
        }
        
        void fit(const std::vector <double> &x_train, const std::vector <double> &y_train) {
            std::cout << "Fitting model ..." << std::endl;
            avgs = getAverage(x_train, y_train);
            x_avg = avgs.first;
            y_avg = avgs.second;
            for (int i = 0; i < x_train.size(); ++i) {
                numerator += (x_train[i] - x_avg) * (y_train[i] - y_avg);
                denominator += (x_train[i] - x_avg) * (x_train[i] - x_avg);
            }
            m = numerator / denominator;
            c = y_avg - m * x_avg;
        }

        std::vector <double> make_pred(const std::vector <double> &x_test) {
            std::cout << "Making predictions ..." << std::endl;
            preds = {};
            for (int i=0; i < x_test.size() ; ++i) {
                preds.push_back(m * x_test[i] + c);
            }
            return preds;
        }

        double R2_score (std::vector <double> &y_test) {
            std::cout << "Scoring model ..." << std::endl;
            double ssr = 0;
            double sst = 0;
            for (int i = 0; i < y_test.size(); ++i) {
                ssr += (y_test[i] - preds[i]) * (y_test[i] - preds[i]);
                sst += (y_test[i] - y_avg) * (y_test[i] - y_avg);
            }
            score = 1 - (ssr / sst);
            std::cout << "The model did " << score << " out of 1" << std::endl;
            return score;
        }
};


int main() {
    LinearRegression model;
    // train dataset
    std::vector<double> x_train = {1, 2, 3, 4, 5, 6};
    std::vector<double> y_train = {1.2, 1.9, 3.2, 3.8, 5.1, 6.3};
    
    // test dataset
    std::vector<double> x_test = {2.5, 3.5, 5.5, 7};
    std::vector<double> y_test = {2.3, 3.5, 5.7, 7.2};    

    model.fit(x_train,y_train);

    std::vector <double> predictions = model.make_pred(x_test);
    const double model_score = model.R2_score(y_test);

    std::cout << "predictions: " << std::endl;
    for (double i :predictions) {std::cout << i << std::endl;} 



    return 0;
}