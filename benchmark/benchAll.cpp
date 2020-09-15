#include <benchmark/benchmark.h>
#include <QVector>
#include <vector>
#include <boost/numeric/odeint.hpp>

#include "DynamicSystems/DynamicSystem.hpp"
#include "DynamicSystemWrapper.hpp"
#include "DynamicSystemParser/DynamicSystemParser.hpp"
#include "Model/Model.hpp"


BENCHMARK_MAIN();


static void BM_DefaultSystemLorenz(benchmark::State &state) {
    QVector<QVector3D> vector;
    float normalize = 7;
    auto pushBacker = DynamicSystemWrapper_n::getPushBackAndNormalizeLambda(vector, normalize);
    auto system = DynamicSystems::AllSystems::getSystemLorenz<decltype(pushBacker)>();

    int pointsCount = 10000000;
    vector.reserve(pointsCount);
    auto constants = system.getInterestingConstants()[0].second;

    for (auto _ : state) {
        system.compute(DynamicSystemWrapper_n::getPushBackAndNormalizeLambda(vector, normalize),
                       {1, 1, 1}, pointsCount, 0.01, constants);
        vector.clear();
        vector.reserve(pointsCount);
    }
}

BENCHMARK(BM_DefaultSystemLorenz)->Unit(benchmark::kMillisecond);


static void BM_PureSystemLorenz(benchmark::State &state) {
    QVector<QVector3D> vector;
    float normalize = 7;
    auto pushBacker = [&vector, normalize](const Model::Point &point) {
        vector.push_back(
                QVector3D(static_cast<float>(point.x) / normalize,
                          static_cast<float>(point.y) / normalize,
                          static_cast<float>(point.z) / normalize)
        );
    };
    //auto pushBacker = DynamicSystemWrapper_n::getPushBackAndNormalizeLambda(vector, normalize);
    std::vector<long double> constValues{10 + static_cast<long double>(rand()) / RAND_MAX,
                                         28 + static_cast<long double>(rand()) / RAND_MAX,
                                         (8.0 / 3.0) + static_cast<long double>(rand()) / RAND_MAX};


    int pointsCount = 10000000;
    vector.reserve(pointsCount);

    for (auto _ : state) {
        auto derivatives = [sigma = constValues[0], r = constValues[1], b = constValues[2]](
                const Model::Point &values) {
            return Model::Point{
                    sigma * (values.y - values.x),
                    values.x * (r - values.z) - values.y,
                    values.x * values.y - b * values.z
            };
        };
        Model::generatePoints(pushBacker, {1, 1, 1}, pointsCount, 0.01, derivatives);
        vector.clear();
        vector.reserve(pointsCount);
    }
}

BENCHMARK(BM_PureSystemLorenz)->Unit(benchmark::kMillisecond);


static void BM_PureGetterSystemLorenz(benchmark::State &state) {
    QVector<QVector3D> vector;
    float normalize = 7;
    auto pushBacker = DynamicSystemWrapper_n::getPushBackAndNormalizeLambda(vector, normalize);
    std::vector<long double> constValues{10, 28, 8.0 / 3.0};
    auto derivativesFunctionGetter = [](std::vector<long double> constValues) {
        return [sigma = constValues[0], r = constValues[1], b = constValues[2]](const Model::Point &values) {
            return Model::Point{
                    sigma * (values.y - values.x),
                    values.x * (r - values.z) - values.y,
                    values.x * values.y - b * values.z
            };
        };
    };

    int pointsCount = 10000000;
    vector.reserve(pointsCount);

    for (auto _ : state) {
        Model::generatePoints(pushBacker, {1, 1, 1}, pointsCount, 0.01, derivativesFunctionGetter(constValues));
        vector.clear();
        vector.reserve(pointsCount);
    }
}

BENCHMARK(BM_PureGetterSystemLorenz)->Unit(benchmark::kMillisecond);


static void BM_DefaultSystemLorenzParser(benchmark::State &state) {
    QVector<QVector3D> vector;
    float normalize = 7;
    auto pushBacker = DynamicSystemWrapper_n::getPushBackAndNormalizeLambda(vector, normalize);
    auto system = DynamicSystemParser::getDynamicSystem<decltype(pushBacker)>("Lorenz Attractor",
                                                                              {"10*(y - x)",
                                                                               "x*(28 - z) - y",
                                                                               "x*y - 2.66*z"});

    int pointsCount = 2000000;
    vector.reserve(pointsCount);
    std::vector<long double> constants{};

    for (auto _ : state) {
        system.compute(DynamicSystemWrapper_n::getPushBackAndNormalizeLambda(vector, normalize),
                       {1, 1, 1}, pointsCount, 0.01, constants);
        vector.clear();
        vector.reserve(pointsCount);
    }
}

BENCHMARK(BM_DefaultSystemLorenzParser)->Unit(benchmark::kMillisecond);

static void BM_BoostLorenz(benchmark::State &state) {
    QVector<QVector3D> vector;
    float normalize = 7;
    auto pushBacker = [&vector, normalize](const boost::array<long double, 3> &point, long double) {
        vector.push_back(
                QVector3D(static_cast<float>(point[0]) / normalize,
                          static_cast<float>(point[1]) / normalize,
                          static_cast<float>(point[2]) / normalize)
        );
    };
    std::vector<long double> constValues{10, 28, 8.0 / 3.0};
    auto derivatives = [sigma = constValues[0], r = constValues[1], b = constValues[2]]
            (const boost::array<long double, 3> &values, boost::array<long double, 3> &deriv, long double) {
        deriv[0] = sigma * (values[1] - values[0]);
        deriv[1] = values[0] * (r - values[2]) - values[1];
        deriv[2] = values[0] * values[1] - b * values[2];
    };

    int pointsCount = 10000000;
    vector.reserve(pointsCount);

    boost::array<long double, 3> start{1, 1, 1};
    for (auto _ : state) {
        boost::numeric::odeint::runge_kutta4<boost::array<long double, 3>, long double> stepper{};
        boost::numeric::odeint::integrate_n_steps(stepper, derivatives, start, static_cast<long double>(0),
                                                  static_cast<long double>(0.01), pointsCount, pushBacker);
        vector.clear();
        vector.reserve(pointsCount);
    }
}

BENCHMARK(BM_BoostLorenz)->Unit(benchmark::kMillisecond);
