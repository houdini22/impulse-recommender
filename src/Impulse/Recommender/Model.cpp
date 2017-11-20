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
            this->x = this->x / 3.99;

            this->theta.resize(this->numFeatures, uniqueCategoriesCount);
            this->theta.setRandom();
            this->theta = this->theta / 3.99;

            this->predictions.resize(uniqueItemsCount, uniqueCategoriesCount);
            this->predictions.setZero();

            /*this->x << -0.63090354072446, -0.6309035407243, -0.50472283257967, 1.3761809198022, 0.63090354072446,
                    1.40066790037295, 1.400667900373, 1.1205343202982, -0.76526705206508, -1.4006679003729;

            this->theta << 0.3723803072231, 1.3418598314896, -0.85712006935633, -0.85712006935633,
                    -1.6171313307557, -1.1804481852672, 1.3987897580115, 1.3987897580115;*/
        }

        void Model::calculatePredictions() {
            for (T_Size i = 0; i < this->predictions.rows(); i++) {
                for (T_Size j = 0; j < this->predictions.cols(); j++) {
                    this->predictions(i, j) = this->means(i);
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
            this->x = x;
        }

        void Model::setTheta(Math::T_Matrix theta) {
            this->theta = theta;
        }

        double Model::getError() {
            this->calculatePredictions();
            double sum = 0.0;

            for (T_Size i = 0; i < this->y.cols(); i++) {
                for (T_Size j = 0; j < this->y.rows(); j++) {
                    if (!std::isnan(this->y(j, i))) {
                        sum += pow(this->predictions(j, i) - (this->y(j, i) + this->means(j)), 2.0);
                    }
                }
            }

            return sum / 2;
        }
    }
}
