cmake_minimum_required(VERSION 3.8)
project(recommender)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetBuilder/CSVBuilder.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetBuilder/CSVBuilder.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetBuilder/Abstract.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetBuilder/Abstract.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Abstract.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Abstract.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Callback.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Callback.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Category.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Category.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/CategoryId.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/CategoryId.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/MinMaxScaling.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/MinMaxScaling.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/MissingData.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/MissingData.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/ZScoresScaling.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/ZScoresScaling.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/common.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/common.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/Dataset.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/Dataset.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetSample.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetSample.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/include.h
        main.cpp src/Impulse/Recommender/Model.h src/Impulse/Recommender/include.h src/Impulse/Recommender/Model.cpp src/Impulse/Recommender/common.h
        src/Impulse/Recommender/Math/common.cpp
        src/Impulse/Recommender/Math/common.h
        src/Impulse/Recommender/Trainer.h
        src/Impulse/Recommender/Trainer.cpp)

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -m64 -flto -march=native -funroll-loops -mavx -msse4.2 -fopenmp")

add_executable(recommender ${SOURCE_FILES})