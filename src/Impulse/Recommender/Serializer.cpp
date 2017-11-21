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

            std::ofstream out(path);
            out << result.dump();
            out.close();
        }

        Math::T_RawVector Serializer::getRolled(Math::T_Matrix &var) {
            Math::T_RawVector result;
            result.reserve((unsigned long) (var.cols() * var.rows()));

            for (T_Size i = 0; i < var.cols(); i++) {
                for (T_Size j = 0; j < var.rows(); j++) {
                    result.push_back(var(j, i));
                }
            }

            return result;
        }
    }
}
