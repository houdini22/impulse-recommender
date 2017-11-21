#include <iostream>

#include "./src/Impulse/Recommender/include.h"

using namespace Impulse::Dataset;
using namespace Impulse::Recommender;

int main() {
    initialize();

    DatasetBuilder::CSVBuilder builder("/home/hud/CLionProjects/recommender/data/data.csv");
    Dataset dataset = builder.build();
    dataset.out();

    DatasetModifier::Modifier::CategoryId categoryId(dataset);
    categoryId.applyToColumn(0);
    categoryId.applyToColumn(1);
    dataset.out();

    Model model(dataset, 2);
    model.calculatePredictions();
    std::cout << model.getPredictions() << std::endl;
    std::cout << model.getError() << std::endl;

    std::cout << "PREDICTION: " << model.predict(1, 1) << std::endl;
    std::cout << "PREDICTION: " << model.predict(1, 2) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 0) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 3) << std::endl;

    Trainer trainer(model);
    trainer.setLearningRate(0.01);
    trainer.setNumOfIterations(5000);
    trainer.setVerbose(true);
    trainer.setVerboseStep(100);
    trainer.train();

    model.calculatePredictions();
    std::cout << model.getPredictions() << std::endl;
    std::cout << "PREDICTION: " << model.predict(1, 1) << std::endl;
    std::cout << "PREDICTION: " << model.predict(1, 2) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 0) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 3) << std::endl;
    std::cout << "PREDICTION (mean): " << model.predict(2) << std::endl;

    return 0;
}