#include "include.h"

namespace Impulse {

    namespace Recommender {

        Trainer::Trainer(Model &model) : model(model) {
        }

        void Trainer::setLearningRate(double value) {
            this->learningRate = value;
        }

        void Trainer::setNumOfIterations(T_Size value) {
            this->numOfIterations = value;
        }

        void Trainer::setVerbose(bool value) {
            this->verbose = value;
        }

        void Trainer::setVerboseStep(T_Size value) {
            this->verboseStep = value;
        }

        void Trainer::train() {
            T_Size step = 0;
            Model model = this->model;
            Math::T_Matrix y = model.getY();
            Math::T_Matrix means = model.getMeans();
            double error = model.getError();

            if (this->verbose) {
                std::cout << "Starting training with error: [" << error << "], [" << this->numOfIterations
                          << "] iterations." << std::endl;
            }

            while (step < this->numOfIterations) {
                model.calculatePredictions();
                Math::T_Matrix predictions = model.getPredictions();
                Math::T_Matrix theta = model.getTheta();
                Math::T_Matrix x = model.getX();

                Math::T_Matrix newX(x.rows(), x.cols());
                Math::T_Matrix newTheta(theta.rows(), theta.cols());

                for (T_Size i = 0; i < x.cols(); i++) {
                    //std::cout << "TEST: " << i << std::endl;
                    for (T_Size j = 0; j < x.rows(); j++) {
                        //std::cout << "TEST2: " << j << std::endl;
                        double gradientSum = 0.0;
                        T_Size k = 0;

                        for (T_Size l = 0; l < y.cols(); l++) {
                            //std::cout << "TEST3: " << l << std::endl;
                            if (!std::isnan(y(i, k))) {
                                //std::cout << "RESULT:" << ((predictions(i, k) - means(i)) - y(i, k)) << std::endl;
                                //std::cout << "RESULT2:" << (y(i, k)) << std::endl;
                                //std::cout << predictions << std::endl;
                                //return;
                                //std::cout << "THETA:" << theta(j, k) << std::endl;
                                //std::cout << "TMP:" << (predictions(i, k) - means(i)) << std::endl;
                                //std::cout << "TMP:" << ((predictions(i, k) - means(i) - y(i, k))) << std::endl;
                                gradientSum += ((predictions(i, k) - means(i) - y(i, k)) * theta(j, k));

                                //std::cout << k << "," << j << std::endl;
                            }
                            k++;
                        }

                        // std::cout << "TMP:" << gradientSum << std::endl;

                        newX(j, i) = x(j, i) - (this->learningRate * gradientSum);
                    }
                }
                /*std::cout << "NEW X: " << std::endl << newX << std::endl;
                return;*/

                for (T_Size i = 0; i < theta.cols(); i++) {
                    //std::cout << "TEST: " << i << std::endl;
                    for (T_Size j = 0; j < theta.rows(); j++) {
                        //std::cout << "TEST2: " << j << std::endl;
                        double gradientSum = 0.0;
                        T_Size k = 0;

                        for (T_Size l = 0; l < y.rows(); l++) {
                            if (!std::isnan(y(k, i))) {
                                //std::cout << "TEST3: " << l << std::endl;
                                //std::cout << "PREDICTIONS: " << (predictions(j, i) - y(k, i)) << std::endl;
                                //std::cout << "Y: " << y(k, i) << std::endl;
                                //std::cout << "NEW X:" << newX(j, k) << std::endl;
                                gradientSum += (predictions(k, i) - means(k) - y(k, i)) * newX(j, k);
                                //std::cout << "PREDICTIONS: " << (predictions(k, i) - means(k)) << std::endl;
                            }
                            k++;
                        }

                        newTheta(j, i) = theta(j, i) - (this->learningRate * gradientSum);
                    }
                }

                //return;

                model.setX(newX);
                model.setTheta(newTheta);

                error = model.getError();

                if ((step + 1) % this->verboseStep == 0) {
                    std::cout << "Step: [" << (step + 1) << "] with error:[" << error << "]." << std::endl;
                }

                step++;
            }

            if (this->verbose) {
                std::cout << "Training ended with error: [" << error << "]." << std::endl;
            }
        }
    }
}