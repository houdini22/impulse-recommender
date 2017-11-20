#include <iostream>

#include "./src/Impulse/Recommender/include.h"

using namespace Impulse::Dataset;

int main() {
    DatasetBuilder::CSVBuilder builder("/home/hud/CLionProjects/recommender/data/data.csv");
    Dataset dataset = builder.build();
    dataset.out();

    DatasetModifier::Modifier::CategoryId categoryId(dataset);
    categoryId.applyToColumn(0);
    categoryId.applyToColumn(1);
    dataset.out();

    Impulse::Recommender::Model model(dataset, 2);
    model.calculatePredictions();
    std::cout << model.getPredictions() << std::endl;
    std::cout << model.getError() << std::endl;

    // return 0;

    Impulse::Recommender::Trainer trainer(model);
    trainer.setLearningRate(0.01);
    trainer.setNumOfIterations(1000);
    trainer.setVerbose(true);
    trainer.setVerboseStep(50);
    trainer.train();

    model.calculatePredictions();
    std::cout << model.getPredictions() << std::endl;

    return 0;
}