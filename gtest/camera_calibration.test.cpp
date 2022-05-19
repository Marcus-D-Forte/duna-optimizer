#include "duna/generic_optimizator.h"
#include "duna/cost/calibration_cost.hpp"
#include "duna/duna_log.h"

#include <gtest/gtest.h>

#define MODEL_PARAMS 4
#define FLOAT_TOL 0.0025

using namespace duna;

using VectorN = GenericOptimizator<MODEL_PARAMS>::VectorN;
using PointT = pcl::PointXYZ;

// Prepare model
camera_calibration_data_t data;

CalibrationCost<MODEL_PARAMS> *cost;
GenericOptimizator<MODEL_PARAMS> *optimizator;

const float matlab_solution[] = {
    -0.010075911761110,
    0.020714594988011,
    -0.058274626693636,
    0.009185665242934,
    -0.000659168270130,
    0.013700587828461,
};

inline void addNoise(camera_calibration_data_t& data, const float noise){

    for (int i = 0; i < data.point_list.size(); ++i)
    {
        float noise_increment = (float)noise * (std::rand() - RAND_MAX / 2) / (float)(RAND_MAX / 2);
        // DUNA_DEBUG("noise: %f\n",noise_increment);
        data.point_list[i].x += noise_increment;
        data.point_list[i].y += noise_increment;
        data.point_list[i].z += noise_increment;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    

    for (int i = 1; i < argc; ++i)
    {
        printf("arg: %2d = %s\n", i, argv[i]);
    }

    // // Initialize dataset
    data.CameraModel << 586.122314453125, 0, 638.8477694496105, 0, 0,
        722.3973388671875, 323.031267074588, 0,
        0, 0, 1, 0;

    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ());
    data.camera_lidar_frame.block<3, 3>(0, 0) = rot;

    // // Point
    data.point_list.push_back(PointT(2.055643, 0.065643, 0.684357));
    data.point_list.push_back(PointT(1.963083, -0.765833, 0.653833));
    data.point_list.push_back(PointT(2.927500, 0.707000, 0.125250));
    data.point_list.push_back(PointT(2.957833, 0.384667, 0.123667));
    data.point_list.push_back(PointT(2.756000, 0.712000, -0.298000));

    // // Pixel
    data.pixel_list.push_back(camera_calibration_data_t::pixel_pair(621, 67));
    data.pixel_list.push_back(camera_calibration_data_t::pixel_pair(878, 76));
    data.pixel_list.push_back(camera_calibration_data_t::pixel_pair(491, 279));
    data.pixel_list.push_back(camera_calibration_data_t::pixel_pair(559, 282));
    data.pixel_list.push_back(camera_calibration_data_t::pixel_pair(481, 388));

    // std::cerr << data.CameraModel << std::endl;
    // std::cerr << data.camera_lidar_frame << std::endl;

    cost = new CalibrationCost<MODEL_PARAMS>(&data);
    optimizator = new GenericOptimizator<MODEL_PARAMS>(cost); // Ptr

    optimizator->setMaxOptimizationIterations(10);

    return RUN_ALL_TESTS();
}

TEST(LaserCameraCalibration, NoNoise)
{
    
    VectorN x0;
    x0.setZero();
    optimizator->minimize(x0);

    Eigen::Map<const Eigen::Matrix<float,MODEL_PARAMS,1>> v_sol(matlab_solution, MODEL_PARAMS);
    Eigen::Matrix<float,MODEL_PARAMS,2> compare;
    compare.col(0) = v_sol;
    compare.col(1) = x0;
    std::cerr << "Solution: " << compare << "\n";

    for (int i = 0 ; i < 6; ++ i){
        EXPECT_NEAR(x0[i], matlab_solution[i], FLOAT_TOL);
    }
}


TEST(LaserCameraCalibration, SmallNoise)
{
    // Add noise
    float noise = 0.0001;

    addNoise(data,noise);

    VectorN x0;
    x0.setZero();
    optimizator->minimize(x0);

    Eigen::Map<const Eigen::Matrix<float,MODEL_PARAMS,1>> v_sol(matlab_solution, MODEL_PARAMS);
    Eigen::Matrix<float,MODEL_PARAMS,2> compare;
    compare.col(0) = v_sol;
    compare.col(1) = x0;
    std::cerr << "Solution: " << compare << "\n";
    EXPECT_NEAR((v_sol-x0).norm(),0.0f,FLOAT_TOL);
}

TEST(LaserCameraCalibration, MediumNoise)
{
    // Add noise
    float noise = 0.0005;

    addNoise(data,noise);

    VectorN x0;
    x0.setZero();
    optimizator->minimize(x0);

    Eigen::Map<const Eigen::Matrix<float,MODEL_PARAMS,1>> v_sol(matlab_solution, MODEL_PARAMS);
    Eigen::Matrix<float,MODEL_PARAMS,2> compare;
    compare.col(0) = v_sol;
    compare.col(1) = x0;
    std::cerr << "\nSolution: " << compare << "\n";
    EXPECT_NEAR((v_sol-x0).norm(),0.0f,FLOAT_TOL);
}

TEST(LaserCameraCalibration, HeavyNoise)
{
    // Add noise
    float noise = 0.0007;

    addNoise(data,noise);

    VectorN x0;
    x0.setZero();
    optimizator->minimize(x0);

    Eigen::Map<const Eigen::Matrix<float,MODEL_PARAMS,1>> v_sol(matlab_solution, MODEL_PARAMS);
    Eigen::Matrix<float,MODEL_PARAMS,2> compare;
    compare.col(0) = v_sol;
    compare.col(1) = x0;
    std::cerr << "\nSolution: " << compare << "\n";

    EXPECT_NEAR((v_sol-x0).norm(),0.0f,FLOAT_TOL);
    
}