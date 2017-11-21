#include "include.h"

namespace Impulse {

    namespace Recommender {

        Model Builder::buildFromJSON(T_String path) {
            std::ifstream fileStream(path);
            nlohmann::json jsonFile;

            fileStream >> jsonFile;
            fileStream.close();

            Model model((T_Size) jsonFile["numberOfFeatures"]);

            Math::T_RawVector x = jsonFile["x"];
            model.setX(createMatrix(x, jsonFile["x_size"][0], jsonFile["x_size"][1]));
            x.clear();

            Math::T_RawVector y = jsonFile["y"];
            model.setY(createMatrix(y, jsonFile["y_size"][0], jsonFile["y_size"][1]).unaryExpr([](const double x) {
                if (x == -1337.0) { // ugly hack
                    return nan("");
                }
                return x;
            }));
            y.clear();

            Math::T_RawVector theta = jsonFile["theta"];
            model.setTheta(createMatrix(theta, jsonFile["theta_size"][0], jsonFile["theta_size"][1]));
            theta.clear();

            Math::T_RawVector means = jsonFile["means"];
            model.setMeans(createMatrix(means, jsonFile["means_size"][0], jsonFile["means_size"][1]));
            means.clear();

            Math::T_RawVector predictions = jsonFile["predictions"];
            model.setPredictions(
                    createMatrix(predictions, jsonFile["predictions_size"][0], jsonFile["predictions_size"][1]));
            predictions.clear();

            return model;
        }

        Math::T_Matrix createMatrix(Math::T_RawVector &vec, T_Size cols, T_Size rows) {
            Math::T_Matrix result(rows, cols);
            T_Size k = 0;

            for (T_Size i = 0; i < cols; i++) {
                for (T_Size j = 0; j < rows; j++) {
                    result(j, i) = vec.at(k++);
                }
            }

            return result;
        }
    }
}