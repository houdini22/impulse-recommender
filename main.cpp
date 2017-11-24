#include <iostream>

#include "./src/Impulse/Recommender/include.h"

using namespace Impulse::Dataset;
using namespace Impulse::Recommender;

void test_movie_lens() {
    DatasetBuilder::CSVBuilder builder("/home/hud/CLionProjects/recommender/data/movie_lens.csv");
    Dataset dataset = builder.build();
    dataset.out();

    DatasetModifier::Modifier::CategoryId categoryId(dataset);
    categoryId.applyToColumn(0);
    categoryId.applyToColumn(1);
    dataset.out();

    Model model(dataset, 75);
    model.calculatePredictions();
    std::cout << model.getError() << std::endl;

/*
    std::cout << "PREDICTION: " << model.predict(1, 1) << std::endl;
    std::cout << "PREDICTION: " << model.predict(1, 2) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 0) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 3) << std::endl;
*/

    SmartTrainer trainer(model);
    trainer.setLearningRate(0.1);
    trainer.setNumOfIterations(10000);
    trainer.setVerbose(true);
    trainer.setVerboseStep(1);
    trainer.train();

    Serializer serializer(model);
    serializer.toJSON("/home/hud/CLionProjects/recommender/saved/movie_lens5.json");
}

void test_test() {
    DatasetBuilder::CSVBuilder builder("/home/hud/CLionProjects/recommender/data/test.csv");
    Dataset dataset = builder.build();
    dataset.out();

    DatasetModifier::Modifier::CategoryId categoryId(dataset);
    categoryId.applyToColumn(0);
    categoryId.applyToColumn(1);
    dataset.out();

    Model model(dataset, 2);

    return;

    model.calculatePredictions();
    //std::cout << model.getPredictions() << std::endl;
    std::cout << model.getError() << std::endl;

    std::cout << "PREDICTION: " << model.predict(1, 1) << std::endl;
    std::cout << "PREDICTION: " << model.predict(1, 2) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 0) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 3) << std::endl;

    Trainer trainer(model);
    trainer.setLearningRate(0.01);
    trainer.setNumOfIterations(5000);
    trainer.setVerbose(true);
    trainer.setVerboseStep(1);
    trainer.train();

    model.calculatePredictions();
    std::cout << model.getPredictions() << std::endl;
/*
    Serializer serializer(model);
    serializer.toJSON("/home/hud/CLionProjects/recommender/saved_test.json");*/
}

void test_my() {
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
    trainer.setNumOfIterations(300);
    trainer.setVerbose(true);
    trainer.setVerboseStep(20);
    trainer.train();

    model.calculatePredictions();
    std::cout << model.getPredictions() << std::endl;
    std::cout << model.getError() << std::endl;
    std::cout << "PREDICTION: " << model.predict(1, 1) << std::endl;
    std::cout << "PREDICTION: " << model.predict(1, 2) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 0) << std::endl;
    std::cout << "PREDICTION: " << model.predict(2, 3) << std::endl;
    std::cout << "PREDICTION (mean): " << model.predict(2) << std::endl;

    Serializer serializer(model);
    serializer.toJSON("/home/hud/CLionProjects/recommender/saved_my.json");
}

void test_load() {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    Model model = Builder::buildFromJSON("/home/hud/CLionProjects/recommender/saved/movie_lens5.json");
    std::cout << model.getError() << std::endl;
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Time: " << duration << std::endl;
    std::cout << "PREDICTION: " << model.predict(0, 0) << std::endl;
    std::cout << "PREDICTION: " << model.predict(20, 0) << std::endl;

    if (false) {
        Trainer trainer(model);
        trainer.setLearningRate(0.001);
        trainer.setNumOfIterations(5000);
        trainer.setVerbose(true);
        trainer.setVerboseStep(1);
        trainer.train();

        Serializer serializer(model);
        serializer.toJSON("/home/hud/CLionProjects/recommender/saved/movie_lens2.json");
    }
}

int main() {
    //initialize();
    //test_movie_lens();
    //test_test();
    test_load();
    //test_my();
    return 0;
}