#include "include.h"

namespace Impulse {

    namespace Recommender {

        Serializer::Serializer(Model &model) : model(model) {

        }

        void Serializer::toJSON(T_String path) {
            nlohmann::json result;

            result["numberOfFeatures"] = this->model.getNumberOfFeatures();

            Math::T_Matrix x = this->model.getX();
            Math::T_Matrix y = this->model.getY();
            Math::T_Matrix theta = this->model.getTheta();
            Math::T_Matrix means = this->model.getMeans();
            Math::T_Matrix predictions = this->model.getPredictions();

            result["x"] = this->getRolled(x);
            result["y"] = this->getRolled(y);
            result["theta"] = this->getRolled(theta);
            result["means"] = this->getRolled(means);
            result["predictions"] = this->getRolled(predictions);

            result["x_size"] = {x.cols(), x.rows()};
            result["y_size"] = {y.cols(), y.rows()};
            result["theta_size"] = {theta.cols(), theta.rows()};
            result["means_size"] = {means.cols(), means.rows()};
            result["predictions_size"] = {predictions.cols(), predictions.rows()};

            std::ofstream out(path);
            out << result.dump();
            out.close();
        }

        Math::T_RawVector Serializer::getRolled(Math::T_Matrix &var) {
            Math::T_RawVector result;
            result.reserve((unsigned long) (var.cols() * var.rows()));

            for (T_Size i = 0; i < var.cols(); i++) {
                for (T_Size j = 0; j < var.rows(); j++) {
                    if (std::isnan(var(j, i))) {
                        result.push_back(-1337.0);
                    } else {
                        result.push_back(var(j, i));
                    }
                }
            }

            return result;
        }
    }
}

