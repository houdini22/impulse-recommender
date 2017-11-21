#include <utility>

#include "include.h"

namespace Impulse {

    namespace Recommender {

        Model::Model(Impulse::Dataset::Dataset &dataset, T_Size numFeatures) : dataset(dataset) {
            this->numFeatures = numFeatures;
            this->datasetMatrix = this->dataset.exportToEigen();
            this->initialize();
        }

        void Model::initialize() {
            T_Size uniqueItemsCount = this->datasetMatrix.col(0).maxCoeff() + 1;
            T_Size uniqueCategoriesCount = this->datasetMatrix.col(1).maxCoeff() + 1;

            this->means.resize(uniqueItemsCount);
            this->means.setZero();

            Math::T_Vector meanCounts(uniqueItemsCount);
            meanCounts.setZero();

            this->y.resize(uniqueItemsCount, uniqueCategoriesCount);
            this->y.setZero();

            for (T_Size i = 0; i < this->datasetMatrix.rows(); i++) {
                this->y(this->datasetMatrix(i, 0), this->datasetMatrix(i, 1)) = this->datasetMatrix(i, 2);
                if (!std::isnan(this->datasetMatrix(i, 2))) {
                    this->means(this->datasetMatrix(i, 0)) += this->datasetMatrix(i, 2);
                    meanCounts(this->datasetMatrix(i, 0)) += 1;
                }
            }

            this->means = this->means.array() / meanCounts.array();
            this->y = this->y.array() - this->means.replicate(1, uniqueCategoriesCount).array();

            this->x.resize(this->numFeatures, uniqueItemsCount);
            this->x.setRandom();
            this->x = this->x.unaryExpr([](const double x) {
                return abs(x) / 3.99;
            });

            this->theta.resize(this->numFeatures, uniqueCategoriesCount);
            this->theta.setRandom();
            this->theta = this->theta.unaryExpr([](const double x) {
                return abs(x) / 3.99;
            });

            this->predictions.resize(uniqueItemsCount, uniqueCategoriesCount);
            this->predictions.setZero();

            /*this->x << 1.1048966816963, 1.1048966816962, 0.88391734535708, -0.3234194309528, -1.1048966816963,
                    1.0757658273392, 1.0757658273393, 0.86061266187134, -1.5226708965577, -1.0757658273392;

            this->theta << -1.4418356320263, -0.63569876479206, 1.0387671984091, 1.0387671984091,
                    -0.84304648053928, -1.6710128715178, 1.2570296760286, 1.2570296760286;*/
        }

        void Model::calculatePredictions() {
            this->predictions.setZero();

            for (T_Size i = 0; i < this->predictions.rows(); i++) {
                for (T_Size j = 0; j < this->predictions.cols(); j++) {
                    Math::T_Vector x = this->x.col(i);
                    Math::T_Vector theta = this->theta.col(j);
                    for (T_Size k = 0; k < x.rows(); k++) {
                        this->predictions(i, j) += (x(k) * theta(k));
                    }
                }
            }
        }

        Math::T_Matrix Model::getPredictions() {
            return this->predictions;
        }

        Math::T_Matrix Model::getTheta() {
            return this->theta;
        }

        Math::T_Matrix Model::getX() {
            return this->x;
        }

        Math::T_Matrix Model::getY() {
            return this->y;
        }

        Math::T_Matrix Model::getMeans() {
            return this->means;
        }

        void Model::setX(Math::T_Matrix x) {
            this->x = std::move(x);
        }

        void Model::setTheta(Math::T_Matrix theta) {
            this->theta = std::move(theta);
        }

        double Model::getError() {
            this->calculatePredictions();
            double sum = 0.0;

            for (T_Size i = 0; i < this->y.cols(); i++) {
                for (T_Size j = 0; j < this->y.rows(); j++) {
                    if (!std::isnan(this->y(j, i))) {
                        sum += pow(this->predictions(j, i) - this->y(j, i), 2.0);
                    }
                }
            }

            return sum / 2;
        }

        double Model::predict(T_Size itemId, T_Size categoryId) {
            return this->predictions(itemId, categoryId) + this->means(itemId);
        }

        void Model::debug() {
            this->calculatePredictions();
            std::cout << "PREDICTIONS: " << std::endl << this->predictions << std::endl << "===" << std::endl;
            //std::cout << "Y: " << std::endl << this->y << std::endl << "===" << std::endl;
        }
    }
}
